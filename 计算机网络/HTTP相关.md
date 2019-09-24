## HTTP简介：
HTTP协议是Hyper Text Transfer Protocol（超文本传输协议）的缩写,是用于从万维网（WWW:World Wide Web ）服务器传输超文本到本地浏览器的传送协议；基于TCP/IP通信协议来传递数据（HTML 文件, 图片文件, 查询结果等）
## HTTP工作原理：
用于客户端-服务端架构上。浏览器作为HTTP客户端通过URL向HTTP服务端即WEB服务器发送所有请求；Web服务器根据接收到的请求后，向客户端发送响应信息；

HTTP默认端口号为80，但可以改为8080或者其他端口；
Web服务器有：Apache服务器，IIS服务器（Internet Information Services）等。

## HTTP三点注意：
* 无连接：限制每次连接只处理一个请求。服务器处理完客户的请求，并收到客户的应答后，即断开连接。采用这种方式可以节省传输时间。
* 媒体独立：只要客户端和服务器知道如何处理的数据内容，任何类型的数据都可以通过HTTP发送。
* 无状态：协议对于事务处理没有记忆能力。缺少状态意味着如果后续处理需要前面的信息，则它必须重传，这样可能导致每次连接传送的数据量增大；但在服务器不需要先前信息时它的应答就较快。

[HTTP消息结构](#http消息结构)

## HTTP八种请求方法（常用的是GET和POST）：
* GET：最常见的一种请求方式，当客户端要从服务器中读取文档时，当点击网页上的链接或者通过在浏览器的地址栏输入网址来浏览网页的，使用的都是GET方式。GET方法要求服务器将URL定位的资源放在响应报文的数据部分，回送给客户端。使用GET方法时，请求参数和对应的值附加在URL后面，利用一个问号（“?”）代表URL的结尾与请求参数的开始，传递参数长度受限制。
GET方式的请求一般不包含”请求内容”部分，请求数据以地址的形式表现在请求行。地址链接如下：
```html
<a href="http://www.google.cn/search?hl=zh-CN&source=hp&q=domety&aq=f&oq=">http://www.google.cn/search?hl=zh-CN&source=hp
&q=domety&aq=f&oq=</a> 
```
地址中”?”之后的部分就是通过GET发送的请求数据，我们可以在地址栏中清楚的看到，各个数据之间用”&”符号隔开。显然，这种方式不适合传送私密数据。另外，由于不同的浏览器对地址的字符限制也有所不同，一般最多只能识别1024个字符，所以如果需要传送大量数据的时候，也不适合使用GET方式。
* POST：对于上面提到的不适合使用GET方式的情况，可以考虑使用POST方式，因为使用POST方法可以允许客户端给服务器提供信息较多。POST方法将请求参数封装在HTTP请求数据中，以名称/值的形式出现，可以传输大量数据，这样POST方式对传送的数据大小没有限制，而且也不会显示在URL中。
* HEAD：类似于get请求，只不过返回的响应中没有具体的内容，用于获取报头
* PUT：传输文件，报文主体中包含文件内容，保存到对应URI位置
* DELETE：删除文件，与PUT方法相反，删除对应URI位置的文件
* TRACE：回显服务器收到的请求，主要用于测试或诊断
* OPTIONS：允许客户端查看服务器的性能
* CONNECT：HTTP/1.1协议中预留给能够将连接改为管道方式的代理服务器。

## HTTP中GET和POST区别（重点）：
GET 和 POST 本质上没有区别，都是 TCP 连接
* GET提交的数据会在地址栏中显示出来，而POST提交不会
  * GET提交，请求的数据会附在URL之后（就是把数据放置在HTTP协议头＜request-line＞中），以?分割URL和传输数据，多个参数用&连接
例如：login.action?name=hyddd&password=idontknow&verify=%E4%BD%A0%E5%A5%BD
如果数据是英文字母/数字，原样发送；如果是空格，转换为+，如果是中文/其他字符，则直接把字符串用BASE64加密，得出如： %E4%BD%A0%E5%A5%BD，其中％XX中的XX为该符号以16进制表示的ASCII；
  * POST提交：提交的数据放置在请求数据＜request-body＞中；
* 传输数据的大小：
虽然HTTP协议没有对传输的数据大小及URL长度进行限制， 而在实际开发中存在的限制主要有：
GET:特定浏览器和服务器对URL长度有限制，因此传输数据就会受到URL长度的限制，因此Get传输的数据量小，但效率较高；
POST:由于不是通过URL传值，理论上数据不受限；但实际各个WEB服务器会规定对post提交数据大小进行限制；
* 安全性：
GET是不安全的，因为URL是可见的，可能会泄露私密信息，如密码等；POST较GET安全性高
* 侧重点
get重点在从服务器上获取资源，post重点在向服务器发送数据；
* 编码问题
GET方式只能支持ASCII字符，向服务器传的中文字符可能会乱码；POST支持标准字符集，可以正确传递中文字符。

## HTTP消息结构
* 请求报文Request：
  * 组成： 请求行（request-line）；请求头部(header)；空行(blank-line)；请求数据(request-data)
  ![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/Request%E6%B6%88%E6%81%AF%E7%BB%93%E6%9E%84.png)
    * 请求行由请求方法、URI和HTTP协议版本3个字段组成，用空格分隔。如：GET /index.html HTTP/1.1
    * [HTTP协议的请求方法有八种](## HTTP八种请求方法)
    * 请求头部通知服务器有关于客户端请求的信息，由关键字/值对组成。每行一对，关键字和值用英文冒号“:”分隔。
    * 最后一个请求头之后是一个空行，发送回车符和换行符，通知服务器以下不再有请求头。
    * 请求数据在POST方法中使用，POST方法适用于需要客户填写表单的场合。
  * 例子：以get和post方法为例（“---------请求行---------”  类似的信息是我手动添加的，实际报文里面没有）：
  ```html
  GET /search?hl=zh-CN&source=hp&q=domety&aq=f&oq= HTTP/1.1  ---------请求行---------
  ---------请求头---------
  Accept: image/gif, image/x-xbitmap, image/jpeg, image/pjpeg, application/vnd.ms-excel, application/vnd.ms-powerpoint, 
  application/msword, application/x-silverlight, application/x-shockwave-flash, */*  
  Referer: <a href="http://www.google.cn/">http://www.google.cn/</a>  
  Accept-Language: zh-cn  
  Accept-Encoding: gzip, deflate
  User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727; TheWorld)  
  Host: <a href="http://www.google.cn">www.google.cn</a>  
  Connection: Keep-Alive  
  Cookie: PREF=ID=80a06da87be9ae3c:U=f7167333e2c3b714:NW=1:TM=1261551909:LM=1261551917:S=ybYcq2wpfefs4V9g; 
  NID=31=ojj8d-IygaEtSxLgaJmqSjVhCspkviJrB6omjamNrSm8lZhKy_yMfO2M4QMRKcH1g0iQv9u-2hfBW7bUFwVh7pGaRUb0RnHcJU37y-
  FxlRugatx63JLv7CWMD6UB_O_r
  ```
  字段     | 含义    
  ---------- | :-----------:  
  Accept     | 客户端可识别的内容类型      
  Referer     | 从哪个页面链接过来的    
  user-Agent     | 产生请求的浏览器类型    
  Host     | 请求的主机名     
  Connection     | 是否需要持久连接 
  Cookie     | 浏览器保存的cookie信息，发送请求时会自动把该域名下的cookie值一起发给wdb服务器     
  
  ```html
  POST /search HTTP/1.1  ---------请求行---------
  ---------请求头---------
  Accept: image/gif, image/x-xbitmap, image/jpeg, image/pjpeg, application/vnd.ms-excel, application/vnd.ms-powerpoint, 
  application/msword, application/x-silverlight, application/x-shockwave-flash, */*  
  Referer: <a href="http://www.google.cn/">http://www.google.cn/</a>  
  Accept-Language: zh-cn  
  Accept-Encoding: gzip, deflate  
  User-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727; TheWorld)  
  Host: <a href="http://www.google.cn">www.google.cn</a>  
  Connection: Keep-Alive  
  Cookie: PREF=ID=80a06da87be9ae3c:U=f7167333e2c3b714:NW=1:TM=1261551909:LM=1261551917:S=ybYcq2wpfefs4V9g; 
  NID=31=ojj8d-IygaEtSxLgaJmqSjVhCspkviJrB6omjamNrSm8lZhKy_yMfO2M4QMRKcH1g0iQv9u-2hfBW7bUFwVh7pGaRUb0RnHcJU37y-
  FxlRugatx63JLv7CWMD6UB_O_r  
  ---------空行---------
  hl=zh-CN&source=hp&q=domety ---------请求数据---------
  ```
* 响应报文Response：
  * 组成：与请求报文类似，包括：状态行、消息报头、空行、响应正文
    * 状态行：通过提供一个状态码来说明所请求的资源情况，格式：HTTP-Version Status-Code Reason-Phrase 
      * HTTP-Version表示服务器HTTP协议的版本；
      * Status-Code表示服务器发回的响应状态代码；
      * Reason-Phrase表示状态代码的文本描述。状态代码由三位数字组成，第一个数字定义了响应的类别，且有五种可能取值
      * 常用的状态码及状态描述对应的含义：
1xx：表示通知消息的，如请求收到了或者正在进行处理
2xx：表示成功
3xx：表示重定向
4xx：表示客户的差错
5xx：表示服务器的差错

状态码      | 文本描述     | 含义     
---------- | :-----------:  | :-----------:  
100     | CONTINUE     | 继续，一般是在发送post请求时，已发送了http header之后服务器返回此信息，表示确认，之后发送
200     | OK     | 客户端请求成功   
301     | Moved Permanently     | 永久性转移了  
302     | Found               | 暂时重定向  
400     | Bad Request     | 客户端请求有语法错误，不能被服务器所理解     
401     | Unauthorized     | 请求未经授权，这个状态代码必须和WWW-Authenticate报头域一起使用    
403     | Forbidden     | 服务器收到请求，但是拒绝提供服务     
404     | Not Found     | 请求资源不存在     
500     | Internal Server Error | 服务器内部错误，无法完成请求  
502     | Bad Gateway     | 作为网关或者代理工作的服务器尝试执行请求时，从远程服务器接收到了一个无效的响应 
503     | Server Unavailable     | 服务器当前不能处理客户端的请求，一段时间后可能恢复正常 
504     | Gateway Time-out   | 充当网关或代理的服务器，未及时从远端服务器获取请求

  * 例子：（“---------状态行---------”  类似的信息是我手动添加的，实际报文里面没有）：
  ```html
  HTTP/1.1 200 OK   ---------状态行---------
  ---------消息报文---------
  Date: Sat, 31 Dec 2005 23:59:59 GMT
  Content-Type: text/html;charset=ISO-8859-1
  Content-Length: 122
  ---------空行---------
  ---------响应正文---------
  ＜html＞
  ＜head＞
  ＜title＞Wrox Homepage＜/title＞
  ＜/head＞
  ＜body＞
  ＜!-- body goes here --＞
  ＜/body＞
  ＜/html＞
  ```

## 页面输入URL，按下回车发生了什么？
1. DNS解析：将域名解析为对应的 IP 地址。解析的过程是现在本地的 host 文件里查找，没有找到就查找本地DNS缓存。如果都没有找到，就查询本地DNS服务器，此服务器收到查询时，如果要查询的域名，包含在本地配置区域资源中，就返回解析结果给客户机，如果不包含的话就迭代地向根服务器查找。当根服务器收到本地域名服务器发出的请求后，要么就给出IP地址，要么给出下一次应该向哪一个权限域名服务器进行查询，知道返回IP地址为止。本地DNS会把映射关系缓存到本地，以便下一次使用。
2. TCP连接：三次握手、其中的流量控制、拥塞控制等
3. 发送HTTP请求：建HTTP请求报文并通过TCP协议中发送到服务器指定端口，网络层 IP 协议查询 MAC 地址： IP 协议的作用是把 TCP 分割好的各种数据包传送给接收方。而要保证确实能传到接收方还需要接收方的 MAC 地址，也就是物理地址。IP 地址和 MAC 地址是一一对应的，一个网络设备的 IP 地址可以更换，但是 MAC 地址一般固定不变。ARP 协议可以将 IP 地址解析成对应的 MAC 地址。当通信的双方不在同一个局域网时，需要多次中转才能到达最终的目标，在中转的过程中需要通过下一个中转站的 MAC 地址来搜索下一个中转目标。
数据到达数据链路层。 客户端发送请求完成
4. 服务器处理请求并返回HTTP报文
5. 浏览器解析渲染页面：浏览器会解析 HTML 生成 DOM Tree，然后解析 CSS 文件构建渲染树， 等到渲染树构建完成， 浏览器开始布局渲染树并将其绘制到屏幕上。JS 的解析是由浏览器中的 js 解析引擎完成。
6. 连接结束：四次挥手的方式拆除TCP连接

* 面试题：cookie 和 session 区别
HTTP 协议是无状态的。一旦数据交换完毕，客户端与服务器端的连接就会关闭，再次交换数据需要建立新的连接。因此服务器无法从连接上跟踪会话，session 和 cookie 就是常用的会话跟踪技术。

cookie： 为了识别用户身份，进行 session 跟踪而存储在用户本地中的数据，属于客户端技术。
客户端向服务器发送 http 请求，服务器的响应会有 set-cookie，下一次访问时，客户端就带上这个 cookie 向服务器请求，这样服务器就能从 cookie（根据 cookie 中的 sessionID） 中识别用户身份。

session： Session 对象存储特定用户会话所需的属性及配置信息，session 产生的 sessionID 存放在 cookie 中，禁用 cookie 后，还可以通过其他方式（url 或表单）获取 sessionID

区别：
  cookie 数据存在客户端，session 数据存在服务端
  cookie 数据大小有限制，且浏览器对每个站点的 cookie 个数有限制； session 没有大小限制，和服务器的内存大小有关。
  cookie 是明文，有安全隐患，通过拦截或本地文件找得到你的 cookie 后可以进行攻击; 而 session 是加密保存在服务端，相对更安全
  Session 是保存在服务器端上会存在一段时间才会消失，如果 session 过多会增加服务器的压力
  
联系： 
  session 基于 cookie 技术， 客户端 Session 默认是以 cookie 的形式来存储的
可以只用 session 或只用 cookie 吗？ 不能，只用 cookie 不安全，只用 session 会占用过多的服务器资源，存放重要的安全的信息（如登录）放 session 中，不重要的放 cookie 中。

* HTTP 缓存
服务器向客户端返回 HTML 文件时，需要考虑这个文件是否已经有缓存过，缓存过是否有变化。

浏览器缓存主要针对静态资源（js、css、图片），也就是第一次请求资源时会将服务器返回的数据写入本地缓存中，下次请求就不用与服务端建立连接请求，直接从本地获取。 优势： 减少请求时间，减少网络带宽消耗。

与缓存相关的首部：
Expires： 响应首部，表示资源过期的时间
Cache-Control： 通用首部，控制缓存策略
If-Modified-Since： 请求首部，资源最近修改时间
Last-Modified： 响应首部，资源最近修改时间
If-None-Match（请求）和 Etag（响应）： 相当于文件 id，用于对比文件内容是否变化

## HHTP1.1版本新特性
* 缓存处理： HTTP1.1 有多种缓存控制策略
* 默认持久连接节省通信量，只要客户端服务端任意一端没有明确提出断开TCP连接，就一直保持连接，可以发送多次HTTP请求
* 管线化，客户端可以同时发出多个HTTP请求，而不用一个个等待响应
* [断点续传原理](#断点续传)

## HTTP1.X 和 HTTP2.0 区别
HTTP1.X 需要多个连接才能实现并发和缩短延迟，而 HTTP2.0 比 HTTP1.X 大幅度提升了 web 性能。
  1. 多路复用
  允许单一的 HTTP/2 连接同时发起多重的请求-响应消息，HTTP1.X 只能顺序执行，回复后再请求下一个资源。
  实现多路复用： 二进制分帧，HTTP2.0 将报文分成 HEADERS 帧和 DATA 帧，它们都是二进制格式的。来自不同数据流的帧可以交错发送，然后再根据每个帧头的数据流标识符重新组装，这样就能保证接收到的资源都是正确的。
  HTTP2 通过让所有数据流共用同一个连接，可以更有效地使用 TCP 连接，让高带宽也能真正的服务于 HTTP 的性能提升
  2. 首部压缩
  HTTP1.X 的头部带有大量信息，没有经过压缩，而且每次都要重复发送，给网络带来额外的负担。
  HTTP2.0 在客户端和服务器同时维护和更新一个包含之前见过的首部字段表，从而避免了重复传输。且使用 Huffman 编码对首部字段进行压缩。
  3. 服务器推送
  服务端推送是一种在客户端请求之前发送数据的机制。
  HTTP2.0 在客户端请求一个资源时，会把相关的资源一起发送给客户端，客户端之后就不需要再次发起请求了。


## 面试题：HTTP 与 HTTPS 区别：
* 通信使用明文不加密，内容可能被窃听
* 不验证通信方身份，可能遭到伪装
* 无法验证报文完整性，可能被篡改
* HTTP使用80端口，HTTPS使用443端口
* https协议需要到ca申请证书，一般免费证书较少，因而需要一定费用。

HTTPS就是HTTP加上加密处理（一般是SSL安全通信线路）+认证+完整性保护

## 断点续传
* 在HTTP头部定义Range（request）和Content-Range（response）字段，标志文件已经传输的数据大小和传输范围；且服务端返回的状态码是206
* 需要注意的是：判断文件是否发生了变化，可采用标志文件唯一性的方法，如Last-Modified 或者MD5值
* 服务端在收到续传请求时，通过If-Range中的内容进行校验，校验一致时返回206的续传回应；不一致时服务端则返回200回应，回应的内容为新的文件的全部数据。
