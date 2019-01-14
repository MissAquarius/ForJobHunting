## 简介
Charles是一款网络代理服务器，通过成为电脑或者浏览器的代理，然后截取请求和请求结果达到分析抓包的目的；常用在MAC平台上。

## 使用前配置
* 电脑端：

![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/charles-1.png)

但是现在只能抓取到http的请求，抓取不到https的请求：
需要打开【Proxy-SSL Proxying Settings】勾选Enable SSL Proxying，add上*:*或者*:443表示允许抓到所有请求或https请求，443端口是https的

![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/charles-2.png)

* 手机端：
  * iOS端第一次使用，需要下载Charles根安装证书：在手机上用Safari打开chls.pro/ssl这个地址下载并安装证书
  * 手机端打开设置，连接到与电脑同一局域网下
  * 点击网络名字后面的感叹号，打开配置代理。 
  * 将配置代理设置为手动，将服务器设置为电脑连接的网络的ip（如何查看电脑的ip地址？ifconfig en0），端口为8888(charles的默认端口)
  
  ![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/charles-3.png)
  
  * 安装之后随便打开一个网页，电脑上的charles就会弹出是否允许连接的信息，点allow即可
* 例子：
  * 利用iOS端的正式版美团app，登陆自己的账号，点击某个外卖订单的在线客服入口，随便点击一个FAQ，抓到的包如下。
  * 页面右边显示的Request和Response的信息，可以以不同的方式展示。比如Raw展示原始的Request和Response消息结构。
  
    ![image](https://github.com/MissAquarius/ForJobHunting/blob/master/image/charles-4.png)
    
## 配置步骤总结：
* 电脑端：Charles -> Proxy -> SSL Proxying settings -> Add 添加抓取HTTPS的请求
* iOS手机端：安装证书:打开chls.pro/ssl -> 安装证书，信任证书:设置 -> 通⽤ -> 关于本机 -> 证书信任设置

## 使用步骤总结：
* 电脑端：[Proxy->macOS Proxy] ，将charles设置成系统代理；查看电脑的IP地址；
* 手机端：与电脑端连接在同一局域网中，设置HTTP代理
* 抓包
