---
layout:	    post
title:      "这不是一篇爬虫教程"
subtitle:   "\"Python网络数据采集\"笔记整理 + 个人心得"
date:       2016-06-02
author:     "kissg"
header-img: "img/2016-06-02-not-a-tutorial-on-crawler/cloud_with_mountain_with_lake.jpg"
tags:
    - python
    - 菜鸟成长日记
---

> 白马非马

## 前言

**郑重声明: 这不是一篇爬虫教程! 但是, 目的是学习爬虫的看官, 不妨驻足, 也许开卷有益.**

**既然不是爬虫的教程, 还能写些什么呢? 我的本意是给诸位理一理写爬虫的思路. 给你一个大局观, 你敢不敢要?**


## 正文

首先, 问一个问题: 爬虫(crawler)是什么, 或者说, 爬虫是做什么的?

诸位可能想写或已经写过一些爬虫程序了, 应该有这样的感受: **互联网, 就是一张巨大无比的"网", 上面挂着无数的资源.** 而我们所谓的爬虫, 就是负责将我们想要的资源采集到本地的自动化程序, 由于采集的过程与蜘蛛在网上到处爬很像, 因此得名"网络蜘蛛", 也称"网络爬虫". 这里我用到一个词——**自动化**, 自动是相对于手动而言的. 当我浏览网页时, 看到一张很喜欢的图片, 想把它保存到本地, 我可以简单地"右击"-"save image as..."; 但是有100张甚至更多这样的图片, 手动下载就不太合适. 这种"脏活累活"就可以交给爬虫来做. "喂, 爬虫老弟, https://xxx.xxx.xxx 这个网页上的图片麻烦你给我全部下载到本地", 然后你就可以美美地去看"权力的游戏"最新一集了. 因此, 爬虫有时也叫"网络机器人".

说了这么多, 讲白了, 爬虫就是一个自动化采集工具.

> 插入语: 我越来越觉得, 编程的终极目标之一就是实现各种自动化.

再来讲一个很重要的概念——网络. 我在另一篇博客"Python之socket初见"中提到`网络通信, 归根结底是进程间通信`. 同样地, 只要你愿意, 完全可以将网络看作是计算机的另一块无限容量的硬盘. 想想如今的各种网盘服务, 不就是最好的证明吗? 当然, 我所讲的"将网络看作硬盘"并不局限于网盘的概念. 试想一下, 用浏览器打开一张网页, 与用文本编辑器打开一个文本文件又有多大差别呢? 网速够快, 加载够快的话, 就仿佛页面上的所有资源都存储在本地一样. (如果你与我一样, 还在用着机械硬盘, 用word打开一个文本文档, 不一定比用Google Docs打开一个网盘上的文本文档更快.)

同样地, 使用爬虫, 也可以将网络看作远端的硬盘来看待. 无怪乎, `urlopen()`与`open()`长得这么像, 用起来也这么像:

```python
>>> from urllib.request import urlopen
>>> with open("a.txt") as f: ...   f.read() ...  'Hello, World\n' >>> with urlopen("http://kissg.me") as f:
...   f.read()
...
b'<!DOCTYPE html>...
```

---

预备知识介绍完毕, 下面来看看"爬虫"的几种不同实现.

### API采集数据

你可能不认同: 这也能算爬虫? 但是, 假设你要采集的是一个天气网站的某地天气, 而它恰好提供了API, 于是你通过调用API马上就获得了需要的数据, 这难道不算爬虫吗? 非要千辛万苦爬网页, 分析文档获取数据才叫爬虫? 爬虫的根本目的是采集数据或者爬取资源, 至于如何实现, 重要吗? 还是重要的(- -!), 有简单实用的方法, 没道理不用对吧?

使用API来采集数据的好处很明显, 在上文已有所体现:

1. 使用简单
2. 数据使用标准格式(比如JSON), 不同开发者, 不同架构, 不同语言都可使用, 且易使用

但是缺点也很明显:

1. 不存在!
2. 请求内容和次数有限制, 无法满足需求

关于通过API采集数据就讲这么多, 我就想告诉诸位: 原来还有一种这么简单的数据采集方式!

```python
# "Python网络数据采集"中的一个例子, 有删改
>>> import json
>>> from urllib.request import urlopen
>>> myIpAddress = "115.236.9.89"
>>> response = urlopen("http://freegeoip.net/json/" + myIpAddress)
>>> response = urlopen("http://freegeoip.net/json/" + myIpAddress).read().decode("utf-8")
>>> responseJson = json.loads(response)
>>> print(responseJson.get("country_code"))
CN
```

### 非API数据采集

我将所有不是通过API获取数据的爬虫统称为一类, 意味着想要从其中解析出目标数据, 还得下一番功夫.

前文提到过, 爬虫的自动化是相对手动而言的. 这手动的过程就是我们在浏览器内的各种操作: 打开链接, 右击保存, 复制粘贴... 因此, 在我看来, 爬虫的终极目标就是模拟浏览器, 或者说模拟浏览器的行为. [图灵测试](https://zh.wikipedia.org/wiki/%E5%9B%BE%E7%81%B5%E6%B5%8B%E8%AF%95)的目的是测试计算机能否表现出与人等价的无法区分的智能; 而现在许多网站都有反爬虫的手段, 会判断接入的是爬虫, 还是网络浏览器, 此时, 如果用赤果果的爬虫去访问网站, 也许你的IP立马就被封了, 这就偷鸡不成蚀把米了. 因此, 写爬虫的第一步, 就是`爬虫浏览器化`(这是突发奇想的一个词, 能明白就行).

当浏览器向服务器发起请求时, 都会带有请求头部(Request Headers). 请求头部字段`User-Agent`会带有浏览器的信息, 比如我的是这样的:

```text
user-agent:Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.63 Safari/537.36
```

当爬虫使用urllib标准库, 却没有设置`User-Agent`时, 默认使用`Python-urllib/3.X`(X取决使用的Python版本), 这不一下子就暴露了嘛.

因此`爬虫浏览器化`, 一个简单有效的方法就是为请求设置头部, 最重要的可能要数`User-Agent`. 某些网站可能还会检查`Referer`是否指向网站本身, 如果不是, 网站可能不会响应. 当然, 你可能通过Chrome浏览器的开发者工具查看Request Headers, 根据实际情况设置需要的头部字段.

`爬虫浏览器化`的另一个要点是模拟登录与处理cookie. 如果要采集的数据只有在登录之后才能看到(如知乎, 你只有登录了才能看到内容), 这一点就显得至关重要了.

针对上述2点, 用第三库`Requests`, 很容易就实现了请求头设置, 表单发送, cookie跟踪等, 下面是简单的使用方法:

```python
# 代码依旧来自"Python网络数据采集", 有删改
import requests
from bs4 import BeautifulSoup

# 创建一个会话对象, 可跨请求使用参数, 并能保持通过该会话实例创建的请求的cookies
session = requests.Session()
headers = {"Users-Agent": "kissg@engine"}  # 胡乱设的User-Agent
# www.whatismybrowser.com 是个不错的开发者网站,值得收藏, 可查看详细浏览器信息
url1 = "https://www.whatismybrowser.com/developers/what-http-headers-is-my-browser-sending"
response1 = session.get(url1, headers=headers)
bsObj = BeautifulSoup(response1.text)
print(bsObj.find("table",{"class":"table-striped"}).get_text)
# ...
# <th>USER_AGENT</th>
# <td>kissg@engine</td>
# ...

data = {"username": "username", "password": "password"}
response2 = session.post("http://pythonscraping.com/pages/cookies/welcome.php", data=data)
print(response2.cookies.get_dict())
# {'loggedin': '1', 'username': 'username'}
```

有时候, 过快的(非人的)爬取速度, 可能也会招致网站的怀疑. 因此, 如果对于采集速度不是要求太高, 可适当放缓(比如加个time.sleep()), 不仅更"人性化"了, 更减小了目标网站的负担, 何乐而不为? 我很喜欢"Python网络数据采集"书中的一句话:

> 真的没有理由去伤害别人的网站

在真正进行采集之前, 还有一个可选的步骤是使用`设置代理`, 一般这样做的目的是避免IP地址被封杀, 而一个小小的副作用是, 网速会变慢, 这似乎挺符合上述放缓采集速度的要求的, 一举两得? 推荐配合[Tor](https://www.torproject.org/)使用, 以下是`Tor`与`PysSocks`搭配的一个例子:

```python
import socks  # 导入了PySocks模块
import socket
from urllib.request import urlopen

socks.set_default_proxy(socks.SOCKS5, "localhost", 9150)  # tor 服务必须运行在9150端口
socket.socket = socks.socksocket
print(urlopen("http://icanhazip.com").read())
# 77.247.181.162, 通过 freegeoip.net 验证, 这是荷兰(Netherlands)的一个IP地址
```

我测试了下, 打开Tor浏览器(启用Tor服务)之后, 不管运行多少次爬虫, IP地址都没有再发生变化. 重启Tor浏览器(重启服务)或点击`New Identity`之后, 再运行爬虫, IP地址就刷新了. 如果希望隔几分钟就重置代理, 写一段代码重启Tor浏览器即可. 对此我没有更好的解决办法, 如果你有更好的办法, 请不吝赐教.

关于爬虫的前期准备, 我知道的就这么多了. 下面, 是时候让爬虫跑(爬)起来了!

我一直觉得写爬虫, 是有套路的, 无非就是: 给定入口, 给定路线, 给定一些规则, 然后爬虫从入口爬入互联网, 沿着路线爬行, 将符合规则的资源采集到本地, 并将按规则发现新的路线, 采集新的符合规则的资源... 我突然觉得, 爬虫就跟**红警的采矿车**很像——采矿车踩到矿之后, 运回精炼厂, 之后接着出去采矿, 当一处矿采完之后, 还会自己探索未知领域发现新矿. 你可能听说过`分布式爬虫`, 可以想象, 那就是在一张网上, 撒了更多的爬虫, 或者一张地图上, 同时有多辆采矿车在采矿(玩过红警的同学可能更有体会).

下面是爬虫的一般性框架(伪代码), 根据[知乎上的回答](https://www.zhihu.com/question/20899988), 有删改:

```python
entrypoint = "http://kissg.me"  # 入口

seen = set()  # 已知的网页, 控制爬虫不重复采集, 一般用set, 利用其互异与速查的特性
wait_to_crawl = queue()   # 等待爬取的网页, 对应上述的"路线"

seen.insert(entrypoint)
wait_to_crawl.put(entrypoint)

while True:
    if wait_to_crawl.size > 0:
        current_url = wait_to_crawl.get()  # get 方法从队列中取出头元素, 带删除效果
        store(current_url)                 # 储存页面
        for next_url in extract_urls(current_url):  # 发现新的链接
            if next_url not in seen:
                seen.insert(next_url)
                wait_to_crawl.put(next_url)         # 用新发现的链接扩展路线
    else:
        break
```

对于简单的爬虫, `urlopen()`函数就够用了. 用它打开一个url(可以是str, 或`Request`对象), 返回一个`Response`对象, 这是一个类文本(file-like)对象, 可以轻松地读取, 在此基础上进行数据采集. 若还对`urlopen()`函数使用了`data`关键字参数, 请求方法将从默认的`GET`变为`POST`, 也就是说, 用`urlopen()`也能实现表单提交. 另, 若使用`Request`对象, 还能设置请求头部与代理. 此处不再多说, 具体使用还请参考[文档](https://docs.python.org/3/library/urllib.request.html)

既然标准`urllib`库已经能做这么多事情了, 为什么还需要`Requests`呢? 因为相对于`urllib`, `Requests`更高级, `Requests`之于`urllib`就好比`Python`之于`C语言`一样, 抽象层级更高, 更易用, 上文的cookie处理就是很好的证明. 三言两语说不尽, 详情还请看文档[Requests: HTTP for Humans](https://requests.readthedocs.io/en/master/)

虽然`urllib`很强大, `Requests`更强大, 但**道高一尺, 魔高一丈**, 面对动态网页的时候, 它们也只能低下高贵的头颅, 表示无能为力. 所谓动态网页, 就是显示的内容(页面元素)会随着时间, 环境或用户操作而发生变化, 可分为服务器端与客户端的, 此处专指客户端动态网页, 因为服务器端的行为, 我们控制不了. 客户端动态网页使用以`JavaScript`为代表的客户端脚本语言来控制网页的展示, 具体表现就是**你在浏览器中看到的网内容, 与用爬虫采集的内容不一样**. 这是因为爬虫并不能执行`JavaScript`代码, 也就不能使页面产生变化. 解决办法是, 使用`selenium` + `phantomjs`.

`selenium`最初是一个为网站自动化测试开发的工具, 它能直接控制浏览器的行为, 可模拟用户的各种操作, 比如键盘输入, 鼠标点击等. 这就为在爬虫里执行`JavaScript`代码提供了可能. 更多selenium的说明, 请看[官网](http://www.seleniumhq.org/); python的selenium库文档, 请看[这里](http://seleniumhq.github.io/selenium/docs/api/py/index.html).

`phantomjs`则是无头的(headless)浏览器, 它会将网站加载到内存并执行页面上的`JavaScript`代码, 但不会向用户展示图形界面, 即你看不到实体的浏览器. [官网](http://phantomjs.org/)

因此, `selenium` + `phantomjs`的爬虫, 实际就是将一个可控的浏览器装入了爬虫程序. 以浏览器无异, 它可以设置request header, 处理表单, 跟踪cookie, 执行JavaScript... 你可以想象, 这是一只变种的自带浏览器的巨型爬虫!

```python
from selenium import webdriver
import time
# 将webdriver.PhantomJS()改成webdriver.Chrome(), 就打开了一个Chrome浏览器
driver = webdriver.PhantomJS(executable_path='/home/kissg/Tools/phantomjs/bin/phantomjs')
driver.get("http://pythonscraping.com/pages/javascript/ajaxDemo.html")  # 打开url
time.sleep(3)  # 延迟3s. 该动态使用ajax设置了一个2s的延迟, 2s之后页面会发生变化
print(driver.find_element_by_id('content').text)
driver.close()
```

前面所做的这一切, 只是采集了网页, 为了获得需求的数据, 还需要进行数据清洗. 当然, 如果爬虫只是用来下载图片等资源, 就没这个必要了.

一般情况下, 采集到的都是html文档, 或者xml文档. 因此, 可以用`BeautifulSoup`和`正则表达式(regex)`进行初步数据清洗.

`BeautifulSoup`是一个可以从html或xml文件中提取数据的Python第三方库. 它的强大之处在于将html或xml文档解析成一个树, 可以很方便地进行导航和搜索, 从而从原始文档中提取出原始数据. 使用`BeautifulSoup`, 最好是有一点html(xml)和css知识, 不用多, 会导航和搜索就可以了. [文档详情](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

`regex`就不说了吧, 一两句真的说不清. 我之前试过, 分别用`regex`和`BeautifulSoup`提取数据, `regex`效率更高, 但`BeautifulSoup`胜在简单. 学习`regex`的话, 建议看一遍[re标准库](https://docs.python.org/3/library/re.html), 很详细.

至于如何对已获得的原始数据进行加工处理, 我只能说, 请便~ 对于不同的数据需求, 可以做统计, 或数据融合, 或者万事休提, 先存到数据库.

## 总结

虽然本文以python为例, 但个人认为许多内容是放之四海皆准的, 我很努力地往这个方向上靠了.

简单地回顾下.

- 本文先是介绍了我所理解的爬虫和网络的概念;
- 然后提供了利用API进行数据采集的思路;
- 再然后按编写爬虫程序的一般过程分别讲了前期准备, 采集方式和后期处理的一些内容:
 - 前期准备: 伪装(浏览器化), 处理登录和cookie, 设置代理
 - 采集方式: `urllib`标准库, `Requests`库采集静态网页; `selenium` + `phantomjs`采集动态网页
 - 后期处理: 初步数据清洗的2大利器: `BeautifulSoup`和`regex`

就这么多了, 之前承诺的对[先睹为快]放出来笔记再进行梳理总结, 我做到了.

不过这篇得算6.10那一档的, 期末了, 我要闭关了.
