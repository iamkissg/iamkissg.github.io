---
layout:	   post
title:     "Welcome to kissg Blog"
subtitle:   " \"Hello wolrd, Hello Blog\""
date:      2016-04-01
author:    "kissg"
header-img: "img/hello-blog/post-bg-hello-blog.jpg"
comments:    true
tags:
    - 无始
    - 菜鸟成长日记
---

## 前言

> kissg Blog is now on!

---

特别感谢[Hux,黄玄](http://huangxuan.me),本博客就是使用他的模板.

## 正文

第一篇博客,按照惯例,先讲讲搭建博客的过程吧.

第一次想写博客是在去年10月份.那天我生日,好友Coffee送了我一个域名.没错,是域名!看着我一脸懵逼的表情,他说:如果你不知道干嘛的话,去Google一下“Github + Jekyll”写博客吧.

头几天,兴致很足,跟着[阮一峰老师的教程](http://www.ruanyifeng.com/blog/2012/08/blogging_with_jekyll.html)走了一遍,成功!能通过 "username.github.io/xxx" 的方式访问博客了.但是绑定域名就一直失败.当时不知道原因所在,现在想想,原因应该是

  1. publishing branch不正确;
  2. _config.yml的baseurl没有设置好.

对于第一点,Github上有明确的说明:
![Github Pages Publishing branch 说明](/img/hello-blog/github-pages-repos.png)
而阮一峰老师的教程里使用的是`gh-pages`分支,因此是无法通过"username.github.io"直接访问博客的,而域名又直接绑定到"username.github.io"上,因此一直都会是404错误.可惜当时没明白,兴致缺缺然,也就没再玩了.

时隔半年,就在我差点就要忘了那个域名的存在的时候,邂逅了Hux的博客(好吧,这里我真的很想用"惊为天人"来描述我当时的感受).再加上一些其他因素,就再次萌生了写博客的冲动.

以下记录了本人搭建“Github Pages+ Jekyll”写作环境的全过程.希望能对其他有写博客想法的同学有所帮助.(如果有想学习如何写博客的同学,请移步[阮一峰老师的教程](http://www.ruanyifeng.com/blog/2012/08/blogging_with_jekyll.html))

以下的指令全部基于Ubuntu14.04 LTS,其他操作系统可能略有不同,请自行更正.

首先,安装`git`,这个似乎不需多描述.使用Ubuntu的包管理工具下载的git是1.9.1版的,并不影响使用.

```bash
sudo apt-get install git
```

其次就是安装`Jekyll`了.

根据`Jekyll`官网的推荐,我安装的是`node.js`,是下载的源码,自行编译安装的.

```bash
cd Downloads/ && wget https://nodejs.org/dist/v4.4.0/node-v4.4.0.tar.gz #强迫症使然,一定要下载到/Downloas下
sudo apt-get gcc make g++                                               #编译需要用到的工具
tar -xvf node-v4.4.0.tar.gz && cd node-v4.4.0/
./configure
make install                                                            #可通过node -v验证安装
```

之后就是安装`ruby`了.这里有一些波折.
我一开始使用Ubuntu的包管理工具安装`ruby`,版本比较旧,是1.9.3的,而2.0以下的`ruby`并不支持`Jekyll3.0`及以上.当我询问过Hux之后,被告知`Github Pages`官方进行了升级,要升级至`Jekyll3`以上(原话).因此,我也其实并没有尝试使用`Jekyll2`来搭配Hux的模板.

然后卸载ruby,重新安装.这次通过`ruby`管理工具来安装最新版本的`ruby`.我选择的是`rvm`,以下是安装过程:

```bash
gpg --keyserver hkp://keys.gnupg.net --recv-keys 409B6B1796C275462A1703113804BB82D39DC0E3
\curl -sSL https://get.rvm.io | bash -s stable
source ~/.rvm/scripts/rvm
```

再安装`ruby2.3.0`,以及`Jekyll`:

```bash
rvm install 2.3.0
gem source -r https://rubygems.org/ -a https://ruby.taobao.org/ #更新源
gem install jekyll
```

至此,环境算是基本搭建完成了~

终于可以开始我的博客之旅了!我直接使用的Hux的模板,因此比较省事.但问题依旧存在.

如前所属,如果想通过 "username.github.io" 直接进行博客访问,要设置_config.yml的baseurl的值为空或"/",并且选择`master`分支.本人在此吃了无数的亏,这是血与泪的教训啊!

希望绑定自己的域名的话,需要先在博客的根目录下创建CNAME文件,在其中添加想要绑定的域名.然后去域名提供商那里添加解析即可.Coffee送我的域名是万网上买的,现在被阿里收购了.因此我就是去万网添加域名解析.添加2条A型记录,主机记录都设置为`www`,解析线路我选择的默认,记录值分别是192.30.252.154/153.再添加一条C型记录,主机记录设为`@`,记录值设为`username.github.io`即可.

到此,就算是大功告成了~


## 后记

呼~从3.29到4.1,总算是将第一篇博文了结了!

看得出来,我是小小菜鸟一枚,姑且就将博客命名为"菜鸟成长日记"吧.

无始方能无终.
