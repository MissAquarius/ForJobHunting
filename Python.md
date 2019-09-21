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

## 进程、线程、协程