## 装饰器
* 理解：
  1. Python中，函数也是一个对象，fun表示函数的地址，fun()表示调用该函数
  2. 函数中可以嵌套函数，但是这些嵌套函数在外部不能直接访问
  3. 函数可以作为参数传递给某个函数，也可以作为某个函数的返回值
* 装饰器：装饰器，可以抽离出大量与函数本身功能无关的雷同代码到装饰器中并继续重用，也就是能够给函数增加额外的功能。装饰器的返回值是一个函数对象。 常用场景：性能测试、插入日志、事务处理、缓存、权限校验等。

* 首先来看一个简单的例子： 如果有两个函数 foo1 和 foo2 都要实现在进入函数主体前，先处理日志的功能，那么我们可能会直接修改原函数，但是要将所有需要该功能的函数都修改一遍；  或者考虑 新定义一个函数 use_logging 来专门处理，这样 foo1 和 foo2 就只需要调用 use_logging，其他函数有需要也可以重用。
```python
import logging

def use_logging(func):
    logging.warning("%s is running" % func.__name__) # logging 的默认设置 warn 会输出，info 不会
    func()

def foo1():
    print('i am foo1')

use_logging(foo1)
# WARNING:root:foo1 is running
# i am foo1
```
缺点：调用的时候不再是调用真正的业务逻辑 foo 函数，而是换成了 use_logging 函数，这就破坏了原有的代码结构， 每次都要把原来的那个 foo 函数作为参数传递给 use_logging 函数

* 简单的装饰器：
```python
import logging


def use_logging(func):
    def wrapper():
        logging.warning("%s is running" % func.__name__) # logging 的默认设置 warn 会输出，info 不会
        func()
    return wrapper


def foo1():
    print('i am foo1')

foo1 = use_logging(foo1)  # use_logging 是一个装饰器，返回的是 wrapper函数的地址，这里相当于是 f=wrapper
foo1()  # 加括号表示执行 wrapper()
```
use_logging 就是一个装饰器，它一个普通的函数，它把执行真正业务逻辑的函数 func 包裹在其中，看起来像 foo 被 use_logging 装饰了一样，use_logging 返回的也是一个函数，这个函数的名字叫 wrapper。在这个例子中，函数进入和退出时 ，被称为一个横切面，这种编程方式被称为面向切面的编程。

* @语法糖：@ 符号就是装饰器的语法糖，它放在函数开始定义的地方，这样就可以省略最后一步再次赋值的操作。
```python
import logging


def use_logging(func):
    def wrapper():
        logging.warning("%s is running" % func.__name__) # logging 的默认设置 warn 会输出，info 不会
        func()
    return wrapper

@use_logging   # 等同于：foo1 = use_logging(foo1)
def foo1():
    print('i am foo1')

foo1()
```
如上所示，有了 @ ，我们就可以省去foo1 = use_logging(foo1)这一句了，直接调用 foo1() 即可得到想要的结果。你们看到了没有，foo() 函数不需要做任何修改，只需在定义的地方加上装饰器，调用的时候还是和以前一样，如果我们有其他的类似函数，我们可以继续调用装饰器来修饰函数，而不用重复修改函数或者增加新的封装。这样，我们就提高了程序的可重复利用性，并增加了程序的可读性。

装饰器在 Python 使用如此方便都要归因于 Python 的函数能像普通的对象一样能作为参数传递给其他函数，可以被赋值给其他变量，可以作为返回值，可以被定义在另外一个函数内。

* *args、**kwargs 
适用于业务逻辑 foo1 函数需要参数，*args表示是非key-value形式的参数； **kwargs表示key-value形式的参数

```python
import logging


def use_logging(func):
    def wrapper(name):  # 接收一个确定的参数，如果不知道几个的时候，可以用*args, 如果接收的是key-value对，用**kwargs
        logging.warning("%s is running" %func.__name__) # logging 的默认设置 warn 会输出，info 不会
        func(name)
    return wrapper


@use_logging
def foo1(name):              # def foo1(name, age)
    print('i am %s' % name)  #     print('i am %s, my age is %d' % (name,age))

foo1('foo1')
```
* 带参数的装饰器
在上面的装饰器调用中，该装饰器接收唯一的参数就是执行业务的函数 foo1 。装饰器的语法允许我们在调用时，提供其它参数，比如@decorator(a)。这样，就为装饰器的编写和使用提供了更大的灵活性。比如，我们可以在装饰器中指定日志的等级，因为不同业务函数可能需要的日志级别是不一样的。
```python
import logging


def use_logging(level):
    def decorate(func):
        def wrapper(*args, **kwargs):
            if level == 'warn':
                logging.warning("%s is running" % func.__name__)
            func(*args, **kwargs)
        return wrapper
    return decorate




@use_logging(level='warn')
def foo1(name, age=18):
    print('i am %s, my age is %d' % (name, age))


foo1('foo1')
```
上面的 use_logging 是允许带参数的装饰器。它实际上是对原有装饰器的一个函数封装，并返回一个装饰器。我们可以将它理解为一个含有参数的闭包。当我们使用@use_logging(level="warn")调用的时候，Python 能够发现这一层的封装，并把参数传递到装饰器的环境中。

* 类装饰器
```python
class Foo(object):
    def __init__(self, func):
        self._func = func

    def __call__(self):
        print ('class decorator runing')
        self._func()
        print ('class decorator ending')

@Foo
def bar():
    print ('bar')

bar()
```
没错，装饰器不仅可以是函数，还可以是类，相比函数装饰器，类装饰器具有灵活度大、高内聚、封装性等优点。使用类装饰器主要依靠类的__call__方法，当使用 @ 形式将装饰器附加到函数上时，就会调用此方法。

* functools.wraps
使用装饰器极大地复用了代码，但是他有一个缺点就是原函数的元信息不见了，比如之前的装饰器在最后打印 print(foo1.__name__)，得到的是 wrapper，而不是 foo1.
```python

```

* 装饰器的执行顺序
1. 
2. 一个函数还可以同时定义多个装饰器，它的执行顺序是从里到外，最先调用最里层的装饰器，最后调用最外层的装饰器，如：
```python
@a
@b
@c
def f ():
    pass
# 等同于 a(b(c(f)))
```


## 迭代器


## 生成器

## 内存管理
内存管理机制是：引用计数、垃圾回收、内存池机制
* 引用计数
1. 变量：变量指针指向具体对象的内存空间，取对象的值
2. 对象：每个对象包含一个头部信息：类型标识符 + 引用计数器
3. 注意：变量名没有类型，类型属于对象（因为变量引用对象，所以类型随对象），变量引用什么类型的对象，变量就是什么类型的。可以通过 id(object) 去查看对象的内存地址
```python
a = '123'
b = a
print(id(a))  # 37991008
print(id(b))  # 37991008  a 和 b 都是指向 '123' 这个对象

```
4. 引用所指判断 is ：is 是判断两个引用所指的对象是否相同 
  1. Python缓存了整数和短字符串，每个对象在内存中只存有一份，引用所指对象就是相同的，即使使用赋值语句，也只是创造新的引用；
    ```python
    a = 2 
    b = 2
    a is b # True

    c = 'abc'
    d = 'abc
    c is d # True
    ```
  2. Python没有缓存长字符串、列表及其他对象，可以由多个相同的对象，可以使用赋值语句创建出新的对象。
    ```python
    c = []
    d = []
    c is d # False
    ```
​ 3. python中对大于256的整数，会重新分配对象空间地址保存对象；对于字符串来说，如果不包含空格的字符串，则不会重新分配对象空间，对于包含空格的字符串则会重新分配

​ （a = 256, b = 256 , a is b == true, a = 300, b= 300, a is b == false;

​ x = "abc ef" , y="abc ef" x is y == false）

* 垃圾回收

* 内存池机制

## 进程、线程、协程

## 下划线变量
* 单前导下划线 _var：以单个下划线开头的变量或方法仅供内部使用,它通常不由Python解释器强制执行，仅仅作为一种对程序员的提示
```python
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
t = Test()
t.foo # 11
t._bar # 23,仅是一个约定，其实还是可以访问，但是影响点在于模块的导入
```
模块导入：如果使用通配符从模块中导入所有名称如：from module import * ，则Python不会导入带有前导下划线的名称；如果是常规导入，如：import module ， 则不受影响

* 单末尾下划线 var_: 一个变量的最合适的名称已经被一个关键字所占用，如：像class或def这样的名称不能用作Python中的变量名称，在这种情况下，可以附加一个下划线来解决命名冲突：
```python
def make_object(name, class):  # 报错：语法错误
def make_object(name, class_): # 可以
```
* 双前导下划线 __var： 双下划线前缀会导致Python解释器重写属性名称，以避免子类中的命名冲突，即：名称修饰，解释器更改变量的名称，以便在类被扩展的时候不容易产生冲突
```python
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 22
        self.__baz = 23
t = Test()
t.foo # 11
t._bar # 22
t.__baz  # AttributeError: 'Test' object has no attribute '__baz'
t._Test__baz # 23
```
_Test__baz 是Python解释器所做的名称修饰， 它这样做是为了防止变量在子类中被重写。如：

```python
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 22
        self.__baz = 23

class ExtendedTest(Test):
    def __init__(self):
        super().__init__()
        self.foo ='ovetwriteen'
        self._bar = 'overwriteen'
        self.__baz = 'overwriten'
t2 = ExtendedTest()
t2.foo  # ovetwriteen
t2._bar  # ovetwriteen
t2._ExtendedTest__baz # ovetwriteen
t2._Test__baz  # 23
```

双下划线名称修饰对程序员是完全透明:
```python
class Test:
    def __init__(self):
        self.__mangled = 'hello'

    def get(self):
        return self.__mangled
print(Test().get()) # hello

------ 以下也是可以的 ------
_Test__mangled = 23

class Test:
    def get(self):
        return __mangled

print(Test().get()) # 23
```
* 双前导和双末尾下划线 __var__: 由双下划线前缀和后缀包围的变量不会被Python解释器修改, 用于系统定义，有特殊用途
```python
__init__ # 构造函数
__call__ # 使一个对象可以被调用
```


* 单下划线 _ :用作一个名字，表示某个变量是临时的或者无关紧要的； 或者表示最近一个使用的变量或者表达式的结果
```python
arrs = [0 for _ in range(5)]  # 临时变量

20 + 3 # 23
_ # 23

```
