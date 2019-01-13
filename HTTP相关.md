 [Github标题1](#github标题1)
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

## [HTTP消息结构](#http消息结构)

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
* PUT：从客户端向服务器传送的数据取代指定的文档的内容
* DELETE：请求服务器删除指定的页面
* TRACE：回显服务器收到的请求，主要用于测试或诊断
* OPTIONS：允许客户端查看服务器的性能
* CONNECT：HTTP/1.1协议中预留给能够将连接改为管道方式的代理服务器。

## HTTP中GET和POST区别（重点）：
* GET提交的数据会在地址栏中显示出来，而POST提交不会
  * GET提交，请求的数据会附在URL之后（就是把数据放置在HTTP协议头＜request-line＞中），以?分割URL和传输数据，多个参数用&连接
例如：login.action?name=hyddd&password=idontknow&verify=%E4%BD%A0%E5%A5%BD
如果数据是英文字母/数字，原样发送；如果是空格，转换为+，如果是中文/其他字符，则直接把字符串用BASE64加密，得出如： %E4%BD%A0%E5%A5%BD，其中％XX中的XX为该符号以16进制表示的ASCII；
  * POST提交：提交的数据放置在请求数据＜request-body＞中；
* 传输数据的大小：
虽然HTTP协议没有对传输的数据大小及URL长度进行限制， 而在实际开发中存在的限制主要有：
GET:特定浏览器和服务器对URL长度有限制，因此传输数据就会受到URL长度的限制；
POST:由于不是通过URL传值，理论上数据不受限；但实际各个WEB服务器会规定对post提交数据大小进行限制；
* 安全性：
POST的安全性要比GET的安全性高。

## HTTP消息结构
* 请求报文Request：
  * 组成： 请求行（request-line）；请求头部(header)；空行(blank-line)；请求数据(request-data)
  ![image](图片链接)
    * 请求行由请求方法、URL和HTTP协议版本3个字段组成，用空格分隔。如：GET /index.html HTTP/1.1
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
      | 状态码      | 文本描述     | 含义     |
      | ---------- | :-----------:  | :-----------: |
      | 200     | OK     | 客户端请求成功     |
      | 400     | Bad Request     | 客户端请求有语法错误，不能被服务器所理解     |
      | 401     | Unauthorized     | 请求未经授权，这个状态代码必须和WWW-Authenticate报头域一起使用     |
      | 403     | Forbidden     | 服务器收到请求，但是拒绝提供服务     |
      | 404     | Not Found     | 请求资源不存在     |
      | 500     | Internal Server Error | 服务器发生不可预期的错误     |
      | 503     | Server Unavailable     | 服务器当前不能处理客户端的请求，一段时间后可能恢复正常     |
    * 消息报头
    * 空行
    * 响应正文
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


	### Github标题1
