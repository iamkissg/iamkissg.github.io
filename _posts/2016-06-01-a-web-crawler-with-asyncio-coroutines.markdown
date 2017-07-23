---
layout:	    post
title:      "[译]基于异步协程的网络爬虫"
subtitle:   "换种姿势写爬虫~"
date:       2016-06-01
author:     "kissg"
header-img: "img/2016-06-01-a-web-crawler-with-asyncio-coroutines/covery.jpg"
comment:    true
tags:
    - python
    - 菜鸟成长日记
    - 译
---

## 译者的话

最近英文资料看多了, 手很痒, 很想翻译一篇高质量的英文文章. 这不, 马上就去跟原作者要了翻译权. 下面开始吧~\\
(我是不会告诉你们, 期末将至, 我要落地, 要忙着应付各种实验, 课程设计, 报告, 期末考了. 之后的几篇如无意外, 都会以翻译的形式出现, 我尽量挑高质量的文章, 并尽量使译文能有原文80%的好.)

### 简介

经典的计算机科学理论强调算法的高效性, 即计算应当尽可能快地完成. 但对于许多网络程序而言, 耗时的并非计算, 而是打开过多网络连接带来的时延, 或者突发事件造成的网络延迟. 这些程序引出的一个问题是: 对大量网络事件的等待. 解决此问题的时新方法是`异步I/O`, 或者称为`async`.

本章用一个网络爬虫程序进行演示说明. 该爬虫是一个典型的异步程序, 它等待大量的响应, 却几乎不做计算. 爬虫一次性爬到的页面越多, 程序就能越快地完成. 如果采用一个线程发起一个请求的方式来实现快速爬取, 当并发数量上升时, 在套接字耗尽之前, 内存或其他线程相关的资源就已经耗尽了. 而采用异步I/O的方式, 就避免了对多线程的需求.

我们用三步来实现这个例子. 首先, 我们将展示一个异步消息循环(async event loop)的例子, 并写出爬虫的大概样子, 它以回调的方式使用消息循环(这种方式的效率很高, 但若对其扩展以运用于更复杂的问题, 将会导致代码不可收拾). 其次, 我们将展示Python的协程的高效性与可扩展性. 我们通过生成器函数来实现简单的协程. 最后, 我们用Python标准库"asyncio"[^foot1], 实现全功能的协程, 并用一个异步队列来这些协程.

### 任务描述

网络爬虫发现并下载一个网站上的所有网页, 可能会对这些网页进行归档或建立索引. 以一个URL的根路径为起点, 爬虫抓取每张页面, 解析得到新的链接, 并将它们加入一个链接的队列. 当抓取的页面没有任何链接, 并且链接队列为空时, 爬虫程序也就停止了.

我们可以通过同时下载多张页面来加快爬取的速度, 当爬虫找到新的链接时, 它用分离的套接字实现对新页面的同步获取. 爬虫会解析收到的响应, 并将新的链接加入链接队列. 过高的并发性可能会降低性能, 因此需要为并发请求的数量设一个上限, 等到请求完成, 才从链接队列中取链接, 发起新的请求.

### 传统方法

我们如何让爬虫并发执行? 传统的方法是, 创建一个线程池. 每个线程通过一个套接字, 每次下载一张页面. 举个例子, 从`xkcd.com`下载一张页面:

```
def fetch(url):
    sock = socket.socket()
    sock.connect(('xkcd.com', 80))
    request = 'GET {} HTTP/1.0\r\nHost: xkcd.com\r\n\r\n'.format(url)
    sock.send(request.encode('ascii'))
    response = b''
    chunk = sock.recv(4096)
    while chunk:
        response += chunk
        chunk = sock.recv(4096)

    # Page is now downloaded.
    links = parse_links(response)
    q.add(links)
```

默认地, 套接字的操作是阻塞的: 当线程调用一个方法时, 比如`connect`或`recv`, 该线程将被暂停, 直到调用的操作完成[^foot2]. 因此, 为了一次性下载多个页面, 我们需要多个线程. 一个成熟的应用程序, 通过维持一个线程池, 复用线程的方式来抵消频繁创建进程的代价; 同样地, 它通过维持一个连接池, 实现套接字的复用.

然而, 线程的代价是高昂的, 操作系统通常会强制为进程, 用户或机器设置不同的线程上限. 在我的系统中(Jesse's system), 一个Python线程占用大约50k的内存资源, 并且启动成千上万的线程将导致故障. 如果我们将并发的套接字上的同步操作也按比例增加到上万的数量级, 在套接字耗尽之前, 就线程就已经耗尽了. 单个线程的开销或者系统对于线程数量的限制是瓶颈所在.

Dan Kegel在他的著作"The C10K problem"[^foot3]中, 概括了多线程I/O并发的局限性. 开篇, 他写到:

> It's time for web servers to handle ten thousand clients simultaneously, don't you think? After all, the web is a big place now.

Kegel在1999年的时候, 创造了"C10K"一词. 一万个连接, 现在听起来似乎挺不足道的, 但到今天, C10K问题也仅仅只是量变, 并没有本质的改变. 在当时, 用一个线性一条连接的方法解决C10K问题是不切实际的, 而现在, 也仅仅是天花板更高了而已. 讲真, 我们的爬虫能与线程配合得相当好. 然而, 对于那些维持着成千上万的连接的超大规模应用, 天花板依然存在: 大多数系统在耗尽线程之后, 仍然能创建套接字. 怎么才能解决这个问题呢?

## 异步

异步I/O框架在单线程中使用非阻塞(non-blocking)套接字, 从而实现并发操作. 我们的异步爬虫, 在连接到服务器之前, 先设置套接字为非阻塞的:

```python
sock = socket.socket()
sock.setblocking(False)
try:
    sock.connect(('xkcd.com', 80))
except BlockingIOError:
    pass
```

令人恼火的是, 即使是正常工作时, 非阻塞的套接字也会在`connect`时抛出异常. 这个异常会反复调用底层的C函数, 为`EINPROGRESS`设置`errno`, 以此来提醒你, 连接已经开始.

现在, 我们的爬虫需要一种方式来获知连接何时建立, 以便它发送HTTP请求. 可以简单地用一个紧密循环(tight loop)来实现:

```python
request = 'GET {} HTTP/1.0\r\nHost: xkcd.com\r\n\r\n'.format(url)
encoded = request.encode('ascii')

while True:
    try:
        sock.send(encoded)
        break  # Done.
    except OSError as e:
            pass

print('sent')
```

然而, 这个方法不仅浪费电(囧), 还不能实现基于多套接字(multuple socket)事件的有效异步等待. 在"远古时代", BSD Unix对此的解决方法是用`select`, 一个C函数, 用于等待一个非阻塞套接字事件的发生, 或者也可以是几个套接字的数组. 如今, 为了应付互联网应用超大连接数的需求, 采用如`poll`的代替方案, BSD上采用`kqueue`, Linux上采用`epoll`. 这些API与`select`类似, 但能更好地应付超大数量的连接数.

Python 3.4的`DefaultSelector`使用基于系统的最优的类`select`函数. 为了实现网络I/O的通知, 我们创建一个非阻塞套接字, 并用`DefaultSelector`注册:

```python
from selectors import DefaultSelector, EVENT_WRITE

selector = DefaultSelector()

sock = socket.socket()
sock.setblocking(False)
try:
    sock.connect(('xkcd.com', 80))
except BlockingIOError:
    pass

def connected():
    selector.unregister(sock.fileno())
    print('connected!')

selector.register(sock.fileno(), EVENT_WRITE, connected)
```

我们忽略伪错误, 并调用`selector.register`, 传入套接字的文件描述符和一个表示我们正在等待的事件的常数. 为了在连接建立时得到通知, 我们需要传入一个`EVENT_WRITE`: 也就是说, 我们希望获知何时套接字是"可写的". 我们还需要传入一个Python函数, `connected`. 当事件发生时, 执行该函数. 这样的函数就被称为*回调函数(callback)*

当选择器(selector)收到I/O通知时, 我们用一个循环来处理它:

```python
def loop():
    while True:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback()
```

回调函数`connected`被存储为`event_key.data`. 因此, 一旦非阻塞套接字连通, 我们可以立即检索数据并回调函数.

不同与之前的紧密循环, 此处对`select`的调用会暂停, 以异步等待下一个I/O事件. 接着, 在循环中执行对于发生的事件的回调. 未完成的操作会被挂起, 直到下一个时间循环时钟的到来, 才继续执行.

所以到目前为止, 我们演示了什么呢? 我们展示了如何启动一个操作, 以及当操作准备就绪时执行回调函数. 异步框架就是建立在我们之前提到的**非阻塞套接字**和**事件循环**两个特征之上的, 这就在单线程中实现了并发操作.

此处, 我们实现了"并发", 但不是传统意义上所谓的"并行". 也就是说, 我们创建了一个能重叠I/O的微型系统. 这意味着, 可以在其他操作期间创建并执行新的操作. 这并不是利用多核实现并行操作. 也就是说, 该系统是为I/O密集型问题设计的, 而不是CPU密集型问题.

我们的事件循环是高效并发I/O的, 因为并没有为每一条连接都分配线程资源. 在真正开始之前, 有必要纠正一个普遍的误解——异步比多线程更快. 坦白说, 通常并不是这样的. 在Python中, 像我们所创建的事件循环, 在处理数量不大但交互频繁的连接时, 要比多线程方式慢. 在没有全局解释器锁(Global Interpreter Lock)的运行时环境下, 多线程同样表现得更出色. 而异步I/O的真正用武之处在于, 充满大量慢连接和不频繁事件的应用场景.

### 回调编程

基于我们刚刚建立的简短的异步框架, 我们该如何创建一个网络爬虫呢? 即使是一个简单的URL捕捉器难于编写.

我们从已经获取的全局URL集合开始, 如下所示:

```python
urls_todo = set(['/'])
seen_urls = set(['/'])
```

集合`seen_urls`包含了`urls_todo`以及访问过的URL. 两个集合都用根URL"/"进行初始化.

抓取一张页面需要一系列的回调. 当一个套接字连接成功时, `connected`回调函数被执行, 用于向服务器发送一个GET请求. 但之后它就必须异步地等待响应, 因此它会回调另一个函数. 当回调函数被执行后, 只有再次被调用, 它才能读取完整的响应.

我们将这些回调函数封装进一个`Fetcher`对象. 因此, 它需要一个URL, 一个套接字对象, 和一个累积存储响应字节的地方:

```python
class Fetcher:
    def __init__(self, url):
        self.response = b''  # Empty array of bytes.
        self.url = url
        self.sock = None
```

我们以`Fetcher.fetch`开始:

```python
    # Method on Fetcher class.
    def fetch(self):
        self.sock = socket.socket()
        self.sock.setblocking(False)
        try:
            self.sock.connect(('xkcd.com', 80))
        except BlockingIOError:
            pass

        # Register next callback.
        selector.register(self.sock.fileno(),
                          EVENT_WRITE,
                          self.connected)
```

`fetch`方法启动一个连接. 但请注意, 在连接建立之前, 该方法就返回了. 它必须将控制权返还给事件循环, 以等待连接的建立. 为了理解, 请想象完整的应用结构, 是这样的:

```python
# Begin fetching http://xkcd.com/353/
fetcher = Fetcher('/353/')
fetcher.fetch()

while True:
    events = selector.select()
    for event_key, event_mask in events:
        callback = event_key.data
        callback(event_key, event_mask)
```

所有的事件通知都是通过调用`select`函数, 然后在事件循环中进行处理. 因此, `fetch`方法必须将控制权交由事件循环, 如此, 程序才能知晓套接字已经建立. 然后, 在循环中回调`connected`函数, 在上述`fetch`方法的最后已经完成该回调函数的注册.

以下是`connected`的实现:

```python
    # Method on Fetcher class.
    def connected(self, key, mask):
        print('connected!')
        selector.unregister(key.fd)
        request = 'GET {} HTTP/1.0\r\nHost: xkcd.com\r\n\r\n'.format(self.url)
        self.sock.send(request.encode('ascii'))

        # Register the next callback.
        selector.register(key.fd,
                          EVENT_READ,
                          self.read_response)
```

该方法发送一个GET请求. 一个真实的应用会检查`send`的返回值, 以防完整的消息不能一次性发送完毕. 但是我们的请求很简短, 我们的应用也很简单. 因此, 就"轻率"地调用`send`, 并等待响应. 当然, 必须注册另一个回调函数, 从而将控制权上交给事件循环. 下一个也是最后一个回调函数是`read_response`, 用于处理服务器的响应:

```python
    # Method on Fetcher class.
    def read_response(self, key, mask):
        global stopped

        chunk = self.sock.recv(4096)  # 4k chunk size.
        if chunk:
            self.response += chunk
        else:
            selector.unregister(key.fd)  # Done reading.
            links = self.parse_links()

            # Python set-logic:
            for link in links.difference(seen_urls):
                urls_todo.add(link)
                Fetcher(link).fetch()  # <- New Fetcher.

            seen_urls.update(links)
            urls_todo.remove(self.url)
            if not urls_todo:
                stopped = True
```

每当选择器发现套接字是"可读的", 就会执行回调函数. 此处"可读"的意思是, 套接字是有数据的, 或它已经被关闭.

回调函数一次最多从套接字读取4kb的数据. 若数据少于4kb, `chunk`将读取所有数据; 若数据量超过4kb, `chunk`读取4kb的数据, 并保持套接字依旧可读, 下一个时钟到来时, 事件循环将再次调用回调函数读取数据. 响应结束时, 服务器将关闭套接字, `chunk`变空.

上述代码中出现的`parse_links`方法, 返回一个URL的集合. 我们为每一个新的URL都启动一个新的抓取器, 并且不设置并发上限. 使用回调函数的异步编程的一个显著特征就是: 不需要为共享数据设置排他锁, 向`seen_urls`集合添加链接就是一个例子. 因为没有抢占式多任务, 我们的代码并不能随意地打断.

我们用一个全局变量`stopped`来控制循环:

```python
stopped = False

def loop():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback()
```

当所有页面都下载完毕, 抓取器停止全局事件循环, 程序结束.

This example makes async's problem plain: spaghetti code. 我们需要某种方式去实现一系列的计算与I/O操作, 并调度这一系列的操作并发地执行. 然后, 不适用多线程, 这一系列的操作就不能用一个单一函数完成: 启动I/O操作, 存储状态, 返回结果. 因此, 需要再思考并写一段状态存储代码.

让我们来解释. 考虑下我们是如何在一个线程里通过常规阻塞套接字抓取URL的:

```python
# Blocking version.
def fetch(url):
    sock = socket.socket()
    sock.connect(('xkcd.com', 80))
    request = 'GET {} HTTP/1.0\r\nHost: xkcd.com\r\n\r\n'.format(url)
    sock.send(request.encode('ascii'))
    response = b''
    chunk = sock.recv(4096)
    while chunk:
        response += chunk
        chunk = sock.recv(4096)

    # Page is now downloaded.
    links = parse_links(response)
    q.add(links)
```

这个函数是如何记住一个套接字操作与下一个操作之间的状态的呢? 它包含了套接字, 一个URL, 可累积的响应对象. 单线程的函数通过使用编程语言提供的基本特征存储局部变量的临时状态, 通过一个局部栈. 该函数又有一点"附加部分"——即代码将在I/O完成时被执行. 通过线程的指令指针, 运行时会记得这个附加部分. 因此, I/O完成后, 你不需要考虑重新存储局部变量和附加部分. 而这一切都是建立在语言基础上的.

但是使用基于回调的异步框架, 这些语言特征毫无帮助. 当等待I/O时, 函数必须显式地保存其状态, 因为在I/O完成前, 函数返回会丢失栈帧. 为了替代局部变量, 基于回调的例子将`sock`和`response`存储为Fetcher实例的实例属性. 为了替代指令指针, 通过注册回调函数`connected`和`read_response`来存储附加部分. 随着应用特征的增长, 我们通过手动回调的方式存储的状态复杂性也增加. 程序员并不乐意从事这样繁重的工作.

更坏的情况是, 在执行下一个回调函数前, 当前的回调函数抛出了异常. 比如, 我们写了一个差劲的`parse_links`方法, 它在解析HTML文档时会抛出异常:

```shell
Traceback (most recent call last):
  File "loop-with-callbacks.py", line 111, in <module>
    loop()
  File "loop-with-callbacks.py", line 106, in loop
    callback(event_key, event_mask)
  File "loop-with-callbacks.py", line 51, in read_response
    links = self.parse_links()
  File "loop-with-callbacks.py", line 67, in parse_links
    raise Exception('parse error')
Exception: parse error
```

以上栈回溯只是显示了事件循环正在执行一个回调函数. 我们并不能知道是错误的原因. 有2种可能会导致出错: 我们忘了将要去哪, 以及何时来. 这种上下文信息的丢失称为"stack ripping", 它容易引发混淆. Stack ripping还会妨碍我们为回调函数链建立异常处理器, 采用"try/except"块包裹函数调用及其子树.

因此, 暂且不说多线程与异步的相对效率, 两者在易错性上
