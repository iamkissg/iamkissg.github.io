---
layout:     post
title:      "Python之元类助解"
subtitle:   ""
date:       2016-04-24
author:     "kissg"
header-img: "img/python-metaclass/light.jpg"
tags:
    - 菜鸟成长日记
    - python
    - metaclass
---

> 凡事都是有讲究的.

## 引文

自上一次写博客到现在已经过去整整15天了.这期间,我看过许多材料,也有许多想付诸笔端与大家分享的.但苦于前人几乎已经把该讲的不该讲的都讲了,而且讲得非常透彻,鞭僻入里,有他们的珠玉在前,加上我又希望坚持原创,一时竟不知从何落笔了.思前想后,不如就写一篇"读书笔记"吧.于是就选定了关于`python3 metaclass`的[这篇文章](http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python/6581949#6581949)(中文版请看[这里](http://blog.jobbole.com/21351/)).（为叙述方便，后文用“原文”来指代这篇文章）

## 正文

按照惯例,先介绍一些预备知识,以便读者能更好地理解.

首先,我想简单讲下`关键字`(keyword)的概念(有时也叫保留字(reserved word))

> 关键字,是编程语言的一类*语法结构*.它们在语言设计之初就被定义了.

说白了,**关键字就是编程语言已经为我们准备好的工具**,我们可以直接拿过来用.举一个例子,比如,要定义一个类,我们会这样写:

```python
>>> class Myclass(object):
...     pass
```

此处`class`关键字就相当于告诉python解释器:"解释器大哥,用户自定义了一个类,类名就叫Myclass,参数是...麻烦您给构造一下".然后,Myclass类就自动被创建了.

另一个需要澄清的概念是`动态编程语言`(dynamic programming language)

> 动态编程语言,是一类在运行时可以改变**程序结构**的语言,例如新的函数,对象甚至代码可以被引进,已有的函数可以被删除或者其他结构上的变化.

下面是一个简单的例子:

```python
>>> class Myclass(object):
...     pass
...
>>> mc = Myclass()
>>> mc.name = "kissg" # Myclass类本身并没有name属性，在运行期间为其添加了name属性
>>> print(mc.name)
kissg
```

*(注:动态编程语言 != 动态类型语言.动态类型语言是指在运行时**确定变量的类型**)*

---

众所周知,python是面向对象的编程语言.对此,我们要清楚并始终牢记一点:

> 在python的世界里,一切皆为对象.

整型浮点型字符串型变量是对象,函数也是对象，类还是对象。只不过,**类作为对象有点特殊,它是自身具有创建对象（类实例）能力的对象**.

那么，类是一个对象有何意义呢？这意味着，我们完全可以像操纵普通对象一样操纵一个类：

- 可以将它赋给一个变量
- 可以对它进行拷贝
- 可以为它增加属性
- 可以将它作为参数传递给某个函数

正是由于python的这些特点，为动态编程提供一个可能，一种思路。

前文已经提到，使用`class`关键字，python解释器就为我们自动地创建了一个类。其实，我们还可以手动地创建一个类，即调用`type`函数。

实际上，当我们使用`class`关键字定义好一个类，python解释器就是通过调用`type`函数来构造类的，它以我们写好的类定义(包括类名，父类，属性）作为参数，并返回一个类。官方文档对此描述如下：

> **class type(name, bases, dict)**\\
With three arguments, return a new type object. This is essentially a dynamic form of the class statement. The name string is the class name and becomes the \_\_name\_\_ attribute; the bases tuple itemizes the base classes and becomes the\_\_bases\_\_ attribute; and the dict dictionary is the namespace containing definitions for class body and becomes the \_\_dict\_\_ attribute.

以下2种方法创建类的方法完全相同。

```python
>>> class X(object):
...     a = 1
...
>>> X = type('X', (object,), dict(a=1))
```

那么，如何为动态创建的类添加method（为了不混淆视听，此处用method来表示类的方法）呢？有两种方法，一种是在创建类的时候指定，一种是在后期动态地添加。方法与前文介绍的添加属性基本类似。实际上，完全可以将method看作是特殊一点的属性，这样，处理method的时候，想想属性是如何处理的，就会简单许多。

```python
# 动态创建类时，指定method
>>> def echo_bar(self):
...       print(self.bar)
...
>>> Foo = type('Foo', (object,), {'echo_bar': echo_bar})

#=======================================================

# 动态地添加method
>>> def echo_bar_more(self):
...       print('yet another method')
...
>>> Foo.echo_bar_more = echo_bar_more
```

值得注意的是，我们使用`class`关键字定义类的时候，method的第一个参数一般总是`self`，它指向调用method的对象（类实例）本身。因此，在我们动态地添加method之前，定义函数时，千万别忘了`self`关键字。否则，错误将超乎你的想象.(思考一下，这个method并没有绑定到类实例上，这是你想要的效果吗？如果是，当我没说)

细心的同学可能已经发现了，`type(name, bases, dict)`其实是一个构造函数（见上文官方文档的引用: `return a new type object`)。而我之前却说，它返回一个类。其实，一个类就是一个`type`的对象。这个结果不算惊世骇俗：我们已经知道类实例是由类创建的一个对象，那么既然类也是一个对象，理应有一个更加强大的存在能够创建类。我们称这个更加强大的存在为`元类(metaclass)`，即类的类。

*（注如果你在其他地方接触过“元xx”，应该很容易就能理解。比如“元数据”就是描述数据的数据）*

无疑，`type`就是一个元类，并且它还是所有类的元类:

```python
# 通过__class__属性可获得对象的类
>>> age = 35
>>> age.__class__
<type 'int'>
>>> name = 'bob'
>>> name.__class__
<type 'str'>
>>> def foo(): pass
>>>foo.__class__
<type 'function'>
>>> class Bar(object): pass
> b = Bar()
>>> b.__class__
<class '__main__.Bar'>
# 通过查看对象的__class__.__class__,可以看出所有类的类都是type
>>> age.__class__.__class__
<type 'type'>
>>> name.__class__.__class__
<type 'type'>
>>> foo.__class__.__class__
<type 'type'>
>>> b.__class__.__class__
<type 'type'>
```

现在你知道，为什么是`type`，而不是`Type`了吧。(提示，对比下`str`，`int`你就懂啦)

> 注意到，上面的代码段有一句`<type 'function'>`，这表示存在一个内建的`function`类\\
所以为什么说函数也是一个对象，因为它们都是function类的一个实例。\\
联系`关键字`的知识，当我们使用`def`关键字时，就是在告诉python解释器“请给我一个function实例”

除了使用`type(name, bases, dict)`来动态地创建类外，我们还可以自定义元类，并用自定义的元类来**控制类的创建行为**。

比如说，我希望类的所有属性都加上前缀`kissg_`。当然我可以在定义类的时候为每个属性手动地加上`kiss_`前缀。而另一种行之有效的方法就是自定义一个元类，由它在创建类的时候，自动地给每个属性加上`kissg_`前缀。这就是所谓的“控制类的创建行为”,并且它是自动进行的。


那么，如何来自定义元类呢？其实只要我们明白了`type`是如何创建类的，自定义元类就是非常简单的一件事，无非就是接收类定义，并修改类定义，再返回一个类。而且，我们完全可以调用`type`函数来返回这个类，从而简化操作。

```python
# 我们已经知道type是一个元类，因此自定义元类应继承自type或其子类
# 有一个约定俗成的习惯，自定义元类一般以Metaclass作为后缀，以明确表示这是一个元类
class AddPrefixMetaclass(type):
    # __new__方法在__init__方法之前被调用
    # 因此，当我们想要控制类的创建行为时，一般使用__new__方法
    # 定义普通类的方法时，我们用self作为第一个参数，来指向调用方法的类实例本身
    # 此处addprefix_metaclass的意义与self类似，用于指向使用该元类创建的类本身
    # 其他参数就是类的定义了，依次是类名，父类的元组，属性的字典
    def __new__(addprefix_metaclass, class_name, class_bases, class_dict):
        prefix = "kissg_"
        addprefix_dict = {} # 我们用一个新的字典来储存加了前缀的属性
        # 遍历类的属性，为所有非特殊属性与私有属性加上前缀
        for name, val in class_dict.items():
            if not name.startswith('_'):
                addprefix_dict[prefix + name] = val
            else:
                addprefix_dict[name] = val

        # 调用type函数来返回类，此时我们使用的是加了前缀的属性字典
        return type(class_name, class_bases, addprefix_dict)

# 指定metaclass为自定义的元类，将在创建类时使用该自定义元类
class Myclass(object, metaclass=AddPrefixMetaclass):
    name = "kissg"

kg = Myclass()
print(hasattr(Myclass, "name"))
# 输出: False
print(hasattr(Myclass, "kissg_name"))
# 输出: True
print(kg.kissg_name)
# 输出: kissg
```

如你所见，自定义元类就这么简单，而元类的使用同样简单，只需在类定义时像使用关键字参数一样，指定metaclass为自定义的元类即可。

按照原文的说法，以上自定义元类不是面向对象编程(Object-oriented programming,简称OOP)的正确写法。正确的写法应该是这样的：

```python
class AddPrefixMetaclass(type):
    # 此处__new__的参数也是约定俗成的写法，就像用**kw表示关键字参数一样
    # cls - 使用自定义元类要创建的类，你可以就简单地记成self
    # clsname - 类名
    # bases - 父类的元组的(tuple)
    # dct - 类属性的字典
    def __new__(cls, clsname, bases, dct):
        prefix = "kissg_"
        addprefix_dict = {}
        for name, val in dct.items():
            if not name.startswith('_'):
                addprefix_dict[prefix + name] = val
            else:
                addprefix_dict[name] = val
        # 元类也是可以被继承的。
        # 调用父类的__new__方法来创建类,简化继承
        return super(AddPrefixMetaclass, cls).__new__(cls, clsname, bases, addprefix_dict)
```

是不是比之前的优雅了许多?既避免了直接调用`type`函数，又使用`super`使继承显得更容易了，而且使用约定俗称的命名方法立显规范与高大上气息。

最后，创建类的元类是可以被继承的，有点拗口，但请区别于元类是可以被继承的。这句话的意思是：定义子类时没有指定元类（即没有`metaclass=XXXMetaclass`），将自动使用其父类的元类来创建该子类。

*注: python2还可以通过指定__metaclass__属性为元类或其他任何能返回类的东西（比如函数）来控制类的创建。python3虽然保留了__metaclass__属性，但其实并无用处，因此本文不展开讲。有兴趣的同学可以看看[原文](http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python/6581949#6581949)，但我觉得没太大必要*

---

## 小结

元类被誉为python的黑魔法(black magic)之一，一方面强调了元类使用的困难，另一方面也强调了元类的强大。如果你仔细看过正文的内容，会发现元类的使用似乎也不太难。如果觉得仍有难度，再看一遍，或者可以看下[原文](http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python/6581949#6581949)。当然本文所举的例子都比较简单，你完全可以用元类实现一些更加强大的功能，比如自定义一个ORM(Object Relational Mapping，对象关系映射),我也是基于此，才找了一些材料学习元类的。

总的来说,元类就做了以下3件事:

1. 拦截类的创建
2. 修改类定义
3. 返回修改后的类

也许这样将步骤拆分了，你能更好得理解记忆。而元类真的就这么简单。

引用原文的一句话作结：

> Everything is an object in python, and they are all either instances of classes or instances of metaclasses.\\
(在python的世界里，一切皆为对象，它们要么是类的实例，要么是元类的实例。)

