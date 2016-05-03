---
layout:     post
title:      "Python之socket初见"
subtitle:   "用python3实现简单抓包器"
date:       2016-05-01
author:     "kissg"
header-img: "img/python-socket-packet-sniffer/road.jpg"
tags:
    - 菜鸟成长日记
    - python
    - socket
---

> 不给跬步，无以至千里

## 引文

因为学业要求，需要编程实现一个抓包器，具体要求如下：

> 捕获本机网卡的IP包，对捕获的IP包进行解析。\\
要求必须输出以下字段：版本号、总长度、标识位、片偏移、协议、原地址与目的地址。

由于编程语言不做要求，我绝对当然就选择了很好很强大的**python**，然后恶补了下**socket**与**python socket programming**的相关知识，这也正是本文的主要内容。我发现，用python来实现抓包器真的太简单了(至少达到实验要求是这样的)，因此将这部分内容放在文章最后([先睹为快](#packet-sniffer))。

## 正文

按照往常的习惯，我还是从最基础的概念讲起——什么是socket？

socket实际上是[进程间通信(Inter-process communication，IPC)](https://en.wikipedia.org/wiki/Inter-process_communication)的一种形式。单机模式下，你有许多方式可以实现IPC，比如文件、管道、共享内存等等，因此这里也并不是socket的用武之地。socket真正的强大在于**跨平台的通信**，即网络通信(换言之，网络通信归根结底是进程间的通信)，几乎是“只此一家，别无分号”。因此，我们常说的socket，又称**netwok socket**。

> A network socket is an endpoint of a connection across a computer network.——[维基百科](https://en.wikipedia.org/wiki/Network_socket)

根据以上定义可知，socket是一个网络连接的一端。其实这样说，socket的概念还是很抽象，但它指明了一点：***是端，即通信的起点或终点***，而不是这整条通信链。socket的本意是插座，我觉得挺形象的：我们不考虑也并不用甚至不能知道数据是如何从一端发送到另一端的，就像我们不用知道插座里的电怎样来的一样，只要它在那里，而我们能通过它实现通信的目的(对于插座是用电)就行了。

有了基本概念之后，我们再来讲讲socket的结构。一个socket至少由以下两部分组成：

1. 本地socket地址：由本地主机的ip地址和端口号组成。
2. 传输协议：例如TCP、UDP、raw IP等。

当一个socket连接到另一个socket时，它还需要加上远程socket的地址。

还是用插座模型来加深理解：每个插座都有一个物理位置属性，而且一定是全球唯一的，比如隔壁老王家二楼卧室左边床头的插座，你再也找不出第二个与它物理位完全相同的插座了，这就是IP地址的作用；要用电，则插座里的电总是要输送到某个用电器的，电灯还是电视机，这就由端口号来指定了，此处的用电器就相当于应用进程；而传输协议则相当于规定了输送到插座的电的电压，是110V还是220V还是380V高压电；远程socket地址就相当于指明了电的来源，是秦山核电站还是葛洲坝水电站。

以上便是对socket结构的比喻描述，需要强调的是，每一项都可以唯一地确定一个socket，比如一个TCP服务器可能同时为多个客户端提供服务，因此就会有多个socket——它们的本地socket地址完全相同，但远程socket地址却不一样；再比如本地与远程socket地址都相同，但使用的协议不一样，一个使用TCP协议，一个使用UDP协议，那就是两个不同的socket。

前文简单地提到了端口号与应用进程之间的关系，不知道大家对端口的概念有没有产生疑惑，比如为什么不直接用进程标识符(Process Identifier，简称PID)来表征一个应用进程呢？以下是我为了更好地理解端口的概念，所查资料的摘要。

> - **A port is a logical construct that identifies a specific process or a type of service.**(端口是能确定一个特定进程或一类服务的逻辑结构。)\\
- **Ports are unnecessary on direct point-to-point links when the computers at each end can only run one program at a time. Ports became necessary after computers became capable of executing more than one program at a time and were connected to modern packet-switched networks.**(在直接以点对点形式连接，并且通信双方每次只能运行一个程序的计算机上，端口是非必要的。当计算机具备了同时运行多个程序的能力，并连接到现代分组交换网上时，端口就显得很必要了。)\\
——[维基百科](https://en.wikipedia.org/wiki/Port_(computer_networking))

事实上，以上内容并不能为我解惑，但我大致明白了端口的功能与PID类似，这让我更困惑了——port与PID有什么不同?

> - 把一个特定机器上运行的特定进程指明为因特网上通信的终点是不可行的，因为进程的创建与撤销都是动态的，通信的一方几乎无法识别对方机器上的进程。另外，往往需要利用目的主机提供的**功能**来识别终点，而不需要知道具体实现这个功能的进程是哪一个。\\
- 虽然通信的终点是应用进程，但只要把传送的报文交到目的主机的某个合适的目的端口，最后的交付到目的进程的工作由TCP来完成就可以了。此处的端口为软件端口，有别于路由器或交换机上的硬件端口，它是应用层的各种协议进程与运输实体进行层间交互的一种地址.不同的操作系统具体实现端口的方法可以不同.\\
- 端口号只具有本地意义，它只是为了标识本地计算机应用层中各进程在和运输层交互时的层间接口。\\
——《计算机网络(第六版)(谢希仁)》(有删改)

这下清晰多了，让我们再把重点拎一拎：

- port是标识本地计算机应用层中各进程和运输层交互的层间接口
- port是运输层的概念，而进程是应用层的概念
- 通信的终点虽然是进程，但数据只要传输到目的主机的某个合适的端口就行了。

再用一个比喻来更形象地解释这个概念：如果用一个学校来比喻一台计算机，那应用层就应该是学校里的各个部门，运输层是传达室，而端口则是传达室里的存放各部门信件的信箱，网络通信的数据就是信件。当有某部门的信件达到时，并不需要直接送到那个部门，送件人也并不知道那个部门怎么走，就将信件投递到传达室内该部门的信箱里，之后的事情与送件人再没有任何关系了。部门里的人需要自己去取件再处理。发件的过程与收件类似。

有了前面的知识铺垫之后，socket编程就变得简单多了，因为说白了就2步：

1. 创建socket
2. 使用socket

在python中创建一个socket，与上述的socket的结构又有点不同，具体格式如下：

```python
# socket.socket(family=AF_INET, type=SOCK_STREAM, proto=0, fileno=None)
s = socket.socket(AF_INET, SOCK_STREAM)
```

参数解释(后2者一般使用默认即可，有用到再讲解):

- family - 用于指定socket的地址类型，前缀AF即Address Family的缩写。最常见的有`AF_INET`和`AF_INET6`两种，分别表示使用IPv4和IPv6协议
- type   - 用于指定socket的类型，使用SOCK前缀。常见的有以下几种类型：
 1. `SOCK_STREAM`：使用TCP或STCP协议，面向连接的socket
 2. `SOCK_DGRAM`：使用UDP协议，无连接的socket
 3. `SOCK_RAW`：这类socket，数据包将绕过运输层，因此可以在应用层得到数据包的头部

一般，使用`AF_INET`和`SOCK_STREAM`/`SOCK_DGRAM`就能应付绝大多数情况了。因为INET socket几乎占了socket总数的99%，而stream socket和datagram socket又是这绝大多数中的绝大多数。

socket的使用，根据情况的不同而有所不同。按使用的协议不同，主要可分为TCP通信和UDP通信；按使用场景的不同，可分为服务器端和客户端两类。但是套路却很简单，如下所示：

```python
# TCP通信的基本步骤：
# TCP server: socket > bind > listen > while True > {accept > recv > send} > close
# TCP client: socket -----------------------------> connect > send > recv  > close

# UDP通信的基本步骤:
# UDP server: socket > bind > recvfrom > sendto > close
# UDP client: socket -------> sendto > recvfrom > close

```

基本套路就是如此，简单说下几个函数：

- `socket.bind(adress)` - 该函数将socket绑定到一个地址上。要注意2点：1. 一个socket只能绑定到一个地址上，不能重复绑定；2. 地址格式必须与创建socket时指定的一致，例如不能将INET socket绑定到INET6 socket上
- `socket.listen([backlog])` - 启动监听，即允许服务器接受连接。backlog参数用于指定最大连接数，一旦服务器达到最大连接数，之后的连接都将被拒绝。
- `socket.connect(address)` - 与指定地址的远程socket建立连接。建立连接之后就可以发送或接受数据了。只有使用TCP协议才需要建立连接，UDP协议是无连接的(connectless)
- `socket.accept()` - 接受一个连接。其返回值是`(conn, address)`，其中conn是是一个新的socket对象，用于与建立连接的socket收发数据，address则作为远程socket地址绑定到该socket上。注意，此时母服务器socket仍然在兢兢业业地负责监听。
- `socket.send(bytes)`/`socket.sendto(bytes[,flags], address)` - 向远程socket发送数据，注意，必须以bytes格式发送。而使用sendto时，要指名目的socket的地址
- `socket.recv(bufsize[, flags])`/`socket.recvfrom(bufsize[, flags])` - 从socket接收数据，通过bufsize指定一次性接收数据的最大量，recv的返回值是bytes对象，而recvfrom的返回值是`(bytes, address)`，address即为发送数据的socket地址。

在此，在此举且仅举一个例子，如下：

```python
# Echo server program
import socket

# 创建socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 56789))  # 绑定地址，如前所述，socket地址由ip地址与port组成
    s.listen(1)  # 启动监听，并设置最大连接数为1
    conn, addr = s.accept()  # 接受连接，并将返回的新socket与addr info分别储存
    with conn:
        print('Connected by', addr)  # 在本地打印信息，测试用
        while True:
            data = conn.recv(65565)  # 65535是一次性能接收的最大值
            if not data:
                break
            conn.send(data)  # 将数据原样发回
```

```python
# Echo client program
import socket

# 创建socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 56789))  # 连接到另一个socket
    s.send(b'Hello, world')  # 向对方socket发送数据
    data = s.recv(65565)
print('Received', str(data))  # 打印收到的数据
```

结果如图所示：

![echo-back-demo](/img/python-socket-packet-sniffer/socket_demo.png)

其实这就是一个socket用于单机模式下IPC的例子，`127.0.0.1`是`localhost`的IP地址。你只要将此处本地socket地址改成远程的socket地址，就可以与远程计算机通信了。(当然，一般情况下，你要发些有意义的消息，人家才会响应你)

*注，运行脚本需要使用管理员权限，因为socket的创建实际是系统调用。*

<p id="packet-sniffer"></p>

接下来我们讲讲抓包器的python3实现。

与前面的例子不同，这里我们使用`RAW socket`，除此之外，我们还需要指定协议，即`socket.socket(family, type, proto, fileno)`中的`proto`，这相当于为指定协议在运输层打开一个缺口，从而在运输层放行该协议的包。另外，由于事先不需要连接到某个远程socket，因此我们使用`recvfrom`来捕获所有本机收到的数据包。以TCP协议为例，最初的程序应该是这样的：

```python
import socket

with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP) as s:
    while True:
        print(s.recvfrom(65565))
```

```text
➜  /home/kissg/Tests sudo python3 socket_demo.py
(b'E\x00\x00<Na@\x00@\x06\xeeX\x7f\x00\x00\x01\x7f\x00\x00\x01\xb9\xa6\x048G\xd4\x95\xff\x00\x00\x00\x00\xa0\x02\xaa\xa$\xfe0\x00\x00\x02\x04\xff\xd7\x04\x02\x08\n\x00\xc8\xdb\x9e\x00\x00\x00\x00\x01\x03\x03\x07', ('127.0.0.1', 0))
...
```

`recvfrom`的返回值是`(data, addrinfo)`，因此上述结果中的`b'E\x00...'`就是一个原始的IP数据报，其由如下3部分组成：`IP header` + `TCP header` + `data`。我们只要从数据报中取出想要解析的头部，再做解析即可。以`IP header`为例：一个IP数据报的头部由20字节的固定部分与长度可变的可选字段组成，如下所示：

```python
# 0                   1                   2                   3
# 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |Version|  IHL  |Type of Service|          Total Length         |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |         Identification        |Flags|      Fragment Offset    |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |  Time to Live |    Protocol   |         Header Checksum       |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                       Source Address                          |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                    Destination Address                        |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# |                    Options                    |    Padding    |
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

通常只解析固定部分即可，因此通过切片(slice)取得数据包的前20个字节，即为所需的IP数据报头部。之后再对各字段分别处理。此处为了方便取得各字段，用到`struct`模块的`unpack`函数([使用方法详情](https://docs.python.org/3/library/struct.html#struct.unpack))，具体见代码注释：

```python
import socket
from struct import unpack

with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP) as s:
    while True:
        raw_packet = s.recvfrom(65565)  # (data, addr)
        packet = raw_packet[0]  # data
        raw_iph = packet[0:20]  # 尚未解析的IP数据报头部固定部分
        # unpack(fmt, buffer) - 根据指定的格式化字符串来拆解给定的buffer
        # B 单字节的整型
        # H 双字节的整型
        # s bytes，前加数字表示取4字节的bytes
        iph = unpack("!BBHHHBBH4s4s", raw_iph)
        fields = {}
        fields["Version"] = iph[0] >> 4  # 版本字段与IP数据报头部共享一个字节，通过右移操作取得单独的版本字段
        fields["IP Header Length"] = (iph[0] & 0xF) * 4  # 首部长度字段的1代表4个字节
        fields["Type of Service"] = iph[1]  # 区分服务，一般情况下并不使用
        fields["Total Length"] = iph[2]  # IP首部+数据的总长度，即len(packet)
        fields["Identification"] = iph[3]  # 标识
        flags = iph[4] >> 13  # 标识位与片偏移共享2个字节，且最高位并且未使用
        fields["MF"] = 1 if flags & 1 else 0  # 测试最低位
        fields["DF"] = 1 if flags & 1 else 0  # 测试中间位
        fields["Fragment Offset"] = iph[4] & 0x1FFF  # 位与操作取得片偏移
        fields["Time to Live"] = iph[5]  # 生存时间，单位是跳数
        fields["Protocol"] = iph[6]  # 数据报携带的数据使用的协议，TCP为6
        fields["Header Checksum"] = iph[7]  # 首部校验和
        # socket.inet_ntoa(..)
        # - convert an ip address from 32-bit packed binary format to string format
        fields["Source Address"] = socket.inet_ntoa(iph[8])
        fields["Destination Address"] = socket.inet_ntoa(iph[9])

        for k, v in fields.items():  # 遍历打印，由于是dict，因此打印是无序的
            print(k, ':', v)
        print("")
```

结果如下所示：

```text
Identification : 58009
Total Length : 40
Header Checksum : 23092
Time to Live : 64
IHL : 20
Source Address : 127.0.0.1
Protocol : 6
Version : 4
Destination Address : 127.0.0.1
Fragment Offset : 0
MF : 0
Type of Service : 0
DF : 0
```

TCP首部的解析与此类似，不再赘述。但是需要注意，不能简单地取`packet[20:40]`的片段作为TCP首部(固定部分也为20字节)，因为由于可选字段的存在，IP首部长度是可变的。

更进一步，当使用

```python
socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
```

来创建socket时，将可以捕获本机所有收到或发出的以太网帧。这意味着我们可以实现功能更加强大的抓包器！现在数据包的格式就变成了：`ethernet frame header` + `IP header` + `inner-Protocol header` + `data`。这`inner-Protocol header`表示IP数据报的数据部分使用协议的首部，具体视数据包类型而定。各首部解析方法都类似，基本上算是重复劳动，不再赘述。以下文字对为何使用这几个参数做了权威的说明：

> - **Packet sockets are used to receive or send raw packets at the device driver (OSI Layer 2) level.**(packet sockets允许我们在OSI模型的第二层(即数据链路层)收发raw packets。)
- **The socket_type is either SOCK_RAW for raw packets including the link level header or SOCK_DGRAM for cooked packets with the link level header removed.**(通过指定socket类型为`SOCK_RAW`或`SOCK_DGRAM`，我们可以分别获得数据链路层头部在内的raw packet或去除了数据链路层头部的packet。)
- **Protocol is the IEEE 802.3 protocol number in network order.**(协议是网络顺序的IEEE802.3协议)(注：网络字节顺序，按从高到低的顺序存储，在网络上使用统一的网络字节顺序以避免兼容性问题；主机字节顺序，不同的主机的主机可能不同，与CPU设计有关，x86结构与PowerPC结构字节顺序相反)
- **All incoming packets of that protocol type will be passed to the packet socket before they are passed to the protocols implemented in the kernel.**(即上文提到的放行)\\
——Linux Programmer's Manual

> - **#define ETH_P_ALL 0x0003      /* Every packet (be careful!!!) */**(0x0003表示匹配所有的包)\\
——<linux/if_ether.h>

看了上面的说明之后，相信大家对此处参数的使用已经有所了解了：`AF_PACKET`表示创建一个packet socket，从而使我们能在数据链路层*收发*raw packets，而`SOCK_RAW`则让系统保留了数据链路层头部，`0x0003`放行了所有包，`socket.ntohs(..)`函数将收到的所有包的字节顺序从网络顺序转换到本机顺序。如此一层层下来，我们就实现了对本机所有收发的以太网帧的捕获。

参考代码,请看[这里](https://github.com/Engine-Treasure/python-practise/blob/master/packet_sniffer.py)

## 小结

我一直觉得将原理、概念等理清了非常重要。编程的手段是多变的，但万变不离其宗，掌握了基本原理，不管是用Python来编程也好，用C/C++也罢，流程都是类似的。因此本文花了大量的篇幅来介绍这些原理/概念，总结一下：

- socket是进程间通信的一种形式，尤其适用于跨平台的进程间通信，即网络通信；
- socket至少由本地socket地址与传输协议组成，若建立了连接，那么还应有远程socket地址；
- port是标识本地计算机应用层中各进程和运输层交互的层间接口(因此，也有说包含了port的socket是应用层与运输层的层间接口的)；
- python socket programming，需要根据客户端/服务器，使用的协议(TCP/UDP)不同来选择不同的编程模式；
- 抓包器要使用raw socket，并指定放行的协议，根据首部的不同，解析方法有不同，但套路类似
