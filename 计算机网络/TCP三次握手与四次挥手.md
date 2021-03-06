## TCP简介
* TCP（传输控制协议）：是一种面向连接的、可靠的、基于字节流的传输层协议
* 在一个 TCP 连接中，仅有两方进行彼此通信，因此广播和多播不能用于 TCP
* TCP 使用校验和，确认和重传机制来保证可靠传输
* TCP 给数据分节进行排序，并使用累积确认保证数据的顺序不变和非重复
* TCP 使用滑动窗口机制来实现流量控制，通过动态改变窗口的大小进行拥塞控制

## TCP三次握手与四次挥手
* 首先了解TCP的报文格式
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/TCP%E5%A4%B4%E9%83%A8%E6%A0%BC%E5%BC%8F.png)

* 重要字段及意义
  * 原端口号（source port）-目的端口号(destination port)：同IP数据报中的源IP与目的IP，唯一确定一条TCP连接
  * 序列号(sequence number)：占4字节，TCP是面向字节流的，在一个TCP连接中传送的字节流的每一个字节都按顺序编号.如：传送1000个字节的字节流，其中每个字节都是有编号的，比如0-999，这里的序号是指发送的时候本报文段的第一个字节的序。
  * 确认号(acknowledgement number)：占4字节，是期望收到下一个报文段的第一个数据字节的序号。如果确认号为N，表示到序号N-1为止的所有的数据都已经正确收到
  * 首部长度(header length)：4位，首部长度也称为数据偏移，其代表的意思是本报文的数据起始处距离本报文段的起始处有多远，因为TCP首部中存在可选字段，所以首部长度不固定，所以这个字段是必要的，可以明确指出TCP报文的首部长度。因为其是按4字节为单位的，所以4位二进制数能表示的最大数是15，也就是首部最大长度是60字节
  * 保留(resv)：占6位，目前未使用，置0
  * 码元比特（Control Bits）：有6个控制位，其中包括：URG（Urgent，ACK(Acknowledgment),PSH(Push)，RST(Reset)，SYN，FIN；
    * ACK：确认号有效性标志，一旦一个连接被建立起来，该位就为1，请求连接的时候该位为0
    * SYN：取值1代表是连接请求或者连接接受的报文，取值为0就代表是其他包
    * FIN：传输数据结束标志，取值1表示此报文段的发送方的数据已经发送完毕，并要求释放连接
  * 窗口(window size)：占16个字节，这里的窗口不是指发送方的窗口，而是指接收端此时还能接收多少数据，因为接收方接收数据的缓冲区的大小是有限的。此处的窗口值是作为发送方设置其发送窗口大小的依据，指出了现在允许对方发送的数据量，
  * 校验和(checksum)：占2个字节，校验首部和数据部分
  * 紧急指针(urgent pointer)：占2个字节，只有在URG=1(表示紧急数据需加速传递)的时候才有效
  
* TCP三次握手：
  * 所谓三次握手(Three-way Handshake)，是指建立一个 TCP 连接时，需要客户端和服务器总共发送3个包。三次握手的目的是连接服务器指定端口，建立 TCP 连接，并同步连接双方的序列号和确认号，交换 TCP 窗口大小信息。
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/%E4%B8%89%E6%AC%A1%E6%8F%A1%E6%89%8B.png)
    * 第一次握手：
    客户端发送一个 TCP 的 SYN 标志位置1的包，指明客户端打算连接的服务器的端口，以及初始序号 X,保存在包头的序列号(Sequence Number)字段里。不携带数据，但是要消耗一个序号。
    发送完毕后，客户端进入 SYN_SEND 状态；

    * 第二次握手：
    服务器发回确认包(ACK)应答。即 SYN 标志位和 ACK 标志位均为1。服务器端选择自己 ISN 序列号，放到 Seq 域里，同时将确认序号(Acknowledgement Number)设置为客户的 ISN 加1，即X+1。 同样不携带数据，但是要消耗一个序号。
    发送完毕后，服务器端进入 SYN_RCVD 状态。

    * 第三次握手
    客户端再次发送确认包(ACK)，SYN 标志位为0，ACK 标志位为1，并且把服务器发来 ACK 的序号字段+1，放在确定字段中发送给对方，并且在数据段放写ISN的+1
    发送完毕后，客户端进入 ESTABLISHED 状态，当服务器端接收到这个包时，也进入 ESTABLISHED 状态，TCP 握手结束。

* TCP四次挥手：
  * TCP 的连接的拆除需要发送四个包，因此称为四次挥手(Four-way handshake)，也叫做改进的三次握手。客户端或服务器均可主动发起挥手动作。
![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/%E5%9B%9B%E6%AC%A1%E6%8C%A5%E6%89%8B.png)
    * 第一次挥手：
    假设客户端想要关闭连接，客户端发送一个 FIN 标志位置为1的包，表示自己已经没有数据可以发送了，但是仍然可以接受数据。
    发送完毕后，客户端进入 FIN_WAIT_1 状态。
    * 第二次挥手：
    服务器端确认客户端的 FIN 包，发送一个确认包，表明自己接受到了客户端关闭连接的请求，但还没有准备好关闭连接。
    发送完毕后，服务器端进入 CLOSE_WAIT 状态，客户端接收到这个确认包之后，进入 FIN_WAIT_2 状态，等待服务器端关闭连接。
    * 第三次挥手
    服务器端准备好关闭连接时，向客户端发送结束连接请求，FIN 置为1。
    发送完毕后，服务器端进入 LAST_ACK 状态，等待来自客户端的最后一个ACK。
    * 第四次挥手：
    客户端接收到来自服务器端的关闭请求，发送一个确认包，并进入 TIME_WAIT状态，等待可能出现的要求重传的 ACK 包。
    服务器端接收到这个确认包之后，关闭连接，进入 CLOSED 状态。
    客户端等待了某个固定时间（两个最大段生命周期，2MSL，2 Maximum Segment Lifetime）之后，没有收到服务器端的 ACK ，认为服务器端已经正常关闭连接，于是自己也关闭连接，进入 CLOSED 状态。

* 面试题：为什么time——wait状态必须等待两个最大段生命周期？
  1. 为了保证A发送的最后一个确认包可以到达B， 最后一个包可能丢失， 那么 B 就会让 A 重传，如果 A 直接关闭的话，就无法收到 B 重传的包
  2. 为了防止已失效的连接请求报文段出现在本连接中。A 在发送完最后一个 ACK 报文段后，再经过时间 2MSL，就可以使得本连接持续时间内说产生的所有报文段都从网络中消失。这样就可以使下一个新的连接中不会出现这种旧的连接请求报文段。
 （MSL指一个片段在网络中最大的存活时间，2MSL就是一个发送和一个回复所需的最大时间。如果直到2MSL，Client都没有再次收到FIN，那么Client推断ACK已经被成功接收，则结束TCP连接。）

* 面试题：为什么握手需要三次，挥手需要四次？
  * 三次握手的原因：为了防止已失效的连接请求到达服务器，让服务器错误打开连接。
    客户端发送的连接请求如果在网络中滞留，那么就会隔很长一段时间才能收到服务器端发回的连接确认。客户端等待一个超时重传时间之后，就会重新请求连接。但是这个滞留的连接请求最后还是会到达服务器，如果不进行三次握手，那么服务器就会打开两个连接，服务端就会等待客户端发数据，但客户端此时并没有请求建立连接，就不会发数据，服务端资源产生了浪费。
    如果有第三次握手，客户端会知道之前已经发过了，忽略服务器之后发送的对滞留连接请求的连接确认，不进行第三次握手，因此就不会再次打开连接。

  * 四次挥手的原因：确保数据能够完成传输
    TCP 释放连接是单向释放，A 没有数据发给 B 了，A 就向 B 发出连接释放报文段，然后停止发送数据。 在三次握手中，同步和确认可以一起发送，但是在连接释放中，可能 B 还要给 A发送数据，所以只能先发出一个确认包，等 B 的数据发送完了， 再发送一个关闭连接的包。

* TCP 长连接和短连接
  * TCP 长连接
  长连接是在一个 TCP 连接上可以连续发送多个数据包。
  长连接在没有数据通信时，定时发送数据包(心跳)，以维持连接状态。如果收到对方回应的 ACK，则连接保持；在超过一定重试次数后都没有收到对方回应，则认为连接丢失，断开 tcp 连接。
  优点： 省去较多的 TCP 建立和关闭的操作，减少浪费，节约时间。 适用于频繁请求资源的用户。
  过程： 连接 -> 数据传输 -> 保持连接(心跳) -> 数据传输 -> 保持连接(心跳) -> ... -> 关闭连接
  应用场景： 操作频繁（读写），点对点的通讯，而且连接数不能太多情况

  * TCP 短连接
  短连接是通信双方有数据交互时，就建立一个 TCP 连接，数据发送完成后，断开此 TCP 连接
  优点： 管理起来比较简单，存在的连接都是有用的连接，不需要额外的控制手段。
  过程： 连接 -> 数据传输 -> 关闭连接
  应用场景： WEB 网站的 http 服务（HTTP1.0 只支持短连接）

* HTTP 和 TCP 的 keep-alive
  HTTP keep-alive： 目的是 tcp 连接复用，避免建立过多的 tcp 连接，可以在同一个 tcp 连接上传送多个 http
  TCP keep-alive： 目的是保持 tcp 连接的存活，也就是通过发送一个数据为空的心跳包来确认连接是否存活。
  
## SYN攻击
* 什么是 SYN 攻击（SYN Flood）？
在三次握手过程中，服务器发送 SYN-ACK 之后，收到客户端的 ACK 之前的 TCP 连接称为半连接(half-open connect)。此时服务器处于 SYN_RCVD 状态。当收到 ACK 后，服务器才能转入 ESTABLISHED 状态.
SYN 攻击指的是，攻击客户端在短时间内伪造大量不存在的IP地址，向服务器不断地发送SYN包，服务器回复确认包，并等待客户的确认。由于源地址是不存在的，服务器需要不断的重发直至超时，这些伪造的SYN包将长时间占用未连接队列，正常的SYN请求被丢弃，导致目标系统运行缓慢，严重者会引起网络堵塞甚至系统瘫痪。
SYN 攻击是一种典型的 DoS/DDoS 攻击。
* 如何检测 SYN 攻击？
检测 SYN 攻击非常的方便，当在服务器上看到大量的半连接状态时，特别是源IP地址是随机的，基本上可以断定这是一次SYN攻击。在 Linux/Unix 上可以使用系统自带的 netstats 命令来检测 SYN 攻击。
* 如何防御 SYN 攻击？
SYN攻击不能完全被阻止，除非将TCP协议重新设计。所做的是尽可能的减轻SYN攻击的危害，常见的防御 SYN 攻击的方法有如下几种：
  * 缩短超时（SYN Timeout）时间
  * 增加最大半连接数
  * 过滤网关防护
  * SYN cookies技术



