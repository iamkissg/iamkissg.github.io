---
layout:	    post
title:      "Mendeley, 文献管理小助手"
subtitle:   "Mendeley简易使用指南"
date:       2017-12-01
author:     "kissg"
header-img: "img/2017-12-01-introduction_to_mendeley/cover.jpg"
comments:    true
mathjax:     true
tags:
    - 工具
---

> 本周邂逅了一款文献管理软件——Mendeley，玩了一天，甚得我心，推荐给大家。以下是我写的使用指南，基本涵盖了Mendeley文献管理的所有功能。

我使用的操作系统是Ubuntu，一款Linux发行版。其实我最初听说的是Endnote，但它并没有提供Linux版本。为了能方便地进行文献管理，不必开一个虚拟机或者重启Windows系统（双系统），我选择了提供全平台支持（Windows、Mac、Linux、iOS、Android、Browser Plugin）的Mendeley。除了全平台支持之外，Mendeley的另一个优势是，它是开源免费的（程序员总是偏爱开源的东西）。最重要的是，它还具有强大的文献管理能力，因此是Endnote很好的替代品。

以下是Mendeley的特点：

*自动从导入到库中的PDF 文件中提取文献信息；
*通过Web Importer（浏览器插件）从各种数据库中直接导入文献；
*强大的文件管理功能；
*具有高亮、注释功能的PDF阅读器；
*方便在Word中插入参考文献；
*通过群组功能与其他研究者共享文献；
*公开个人Profile。

安装完毕Mendeley，首次运行时，软件会提示安装Office插件，如图1所示。Libre Office是Linux桌面版下默认的Office，因此它提示安装Libre Office插件而不是Microsoft Office插件。若安装Windows版，程序应提示安装Microsoft Office插件。安装插件Office插件能极大方便参考文献的编写，因此选择安装。

![首次运行Mendeley，提示安装Office插件](/img/2017-12-01-introduction_to_mendeley/1.png)

Mendeley的界面比较分明，最上方是菜单栏、工具栏与搜索框；左侧上方是导航，下方是作者信息，此时未导入文献，显示为空；中央就是文献目录区，此时同样为空；右侧是选中文献的基本资料。如图2所示。

![Mendeley的用户界面，未导入文献](/img/2017-12-01-introduction_to_mendeley/2.png)

在使用文献管理工具之前，我已经搜集了不少资料。得益于Mendeley自动从导入的PDF中提取文献信息的功能，避免了手动录入的麻烦。Mendeley提取文献信息偶尔会出现错误。对文献标题与作者的简单核对，结果如图3所示。

Mendeley提供了“Citations”和“Table”两种视图，以下是Table形式的试图。可以看到，作者信息都被提取到了左下角的“Filter by Authors”区，可以方便地根据作者查找文献。文献区提供了文献的基本信息，包括作者、标题、发表年、期刊等。而最左侧的三列提供了加星，已读/未读，是否本地存储的有用小功能。由于我的都是本地导入的，因此第三列用一个Adobe PDF的标记表示本地存储。右侧显示的是选中的文献的更细节的信息，包括摘要、关键字、文献的可获得URL等，都是可编辑的。此外，如果阅读文献时做了笔记，点击“Notes”即可查看；“Contents”是选中文献的目录。

![Mendeley用户界面，导入文献](/img/2017-12-01-introduction_to_mendeley/3.png)

这些文献在不同时期被下载，因此存在重复的情况。Mendeley提供了文献去重功能，在菜单栏“Tools”下有一项“Check for Duplicates”，选择点击，Mendeley将根据文献的细节信息对不同文献进行对比。如图4所示，它从我的文献库中找到7组很可能是重复的文献，“Confidence”表示程序判断对应组是重复的置信度/可能性。原理很简单，就是从文献细节中提取相应字段进行对比，图4中程序选择了摘要、URL、ArXiv ID、ISSN等等，匹配上的字段阅读，置信度越高，就是这样。但是，用户是可以修改这些字段的，所以如果这些字段填写错误，去重的错误率也会增大。经比对，这几组确实都是重复文献，但文件名上稍有不同，全部选择合并。

![使用Mendeley对文献进行去重](/img/2017-12-01-introduction_to_mendeley/4.png)

Mendeley提供了基本的文献检索功能，左上角“Mendeley”下的“Literature Search”检索外部文献，右上角的搜索框则提供了在用户的文献库内进行检索的功能。图5展示了使用外部检索的结果。可以看到，很多结果实际上都指向同一篇文章。我怀疑，这是因为不同用户为同一篇文章填写的细节字段不同，而Mendeley的服务器并没有对所有文献进行去重，因此存在同一篇文章的多个版本。这也是导入PDF自动提取文献信息出错的一个原因。Mendeley根据它的数据库为导入的文章赋信息，而不是提取信息。我好几次遇到一篇文章被“信息提取”为另一篇文章的情况，就是这个原因。另外，可以看到，搜索结果中，如果用户文献库中已存在某篇文章，在结果前会有一个小勾；如果是用户文献库中不存在的文献，除了对应记录前没有打勾的标记外，选中时，右侧“Details”下会提示保存引用。

![使用Mendeley进行外部文献检索](/img/2017-12-01-introduction_to_mendeley/5.png)

切换成“Citations”视图时，文献会以指定的参考文献格式展示，图6所示为默认的参考文献格式（American Psychological Association 6th Edition）展示的结果。一种浓郁的参考文献气息扑面而来。这也是文献管理软件的管理文献之外的另一大用途——方便用户编写参考文献。右击文献，有一个“Copy As”的选项，可复制当前的参考文献格式。当然，更方便的还是直接使用Office插件。

![以Citation视图展示文献](/img/2017-12-01-introduction_to_mendeley/6.png)

点击菜单栏“View”=>“Citation Style”可更改参考文献格式，极大方便了不同参考文献格式的论文写作。图7所示是上述操作的对话框。在安装软件时，默认已经安装了基本的参考文献格式，用户也可以自行下载安装其他参考文献格式。

![Mendeley默认提供的引用风格](/img/2017-12-01-introduction_to_mendeley/7.png)

Mendeley还有一个比较好的地方是，右击文献，选择“Related Documents”可以查找与当前文献相关的文献。这对于研究有所帮助。图8展示了“Wasserstein GAN”的相关文献。10个结果中，有4个是这篇文章本身，可见Mendeley服务器上没有对文献进行去重的危害有多大。“Related Documents”的功能虽然有帮助，但帮助也没有想象中的大，总之能帮助做一点扩展阅读。

![使用Mendeley查找相关文献](/img/2017-12-01-introduction_to_mendeley/8.png)

Mendeley内置了PDF阅读器，对于本地存储的文献，双击即可打开。图9展示了Mendeley作为PDF阅读器的界面。从工具栏可以看出，Mendeley提供了做笔记、高亮等功能，还算挺实用的。

![使用Mendeley作为PDF阅读器](/img/2017-12-01-introduction_to_mendeley/9.png)

在Office中使用Mendeley来辅助参考文献的写作真的很方便。即使在Windows下，我也是使用WPS Office的，Mendeley并不为WPS Office提供插件（我在网上看到网友要求支持WPS Office的呼声很高）。在Libre Office中使用Mendeley插入参考文献很简单粗暴。图10是安装了Mendeley插件后，Libre Office Writer（对应Microsoft Office 的Word）的工具栏。

![安装Mendeley插件后，Libre Office Writer的工具栏](/img/2017-12-01-introduction_to_mendeley/10.png)

我查了下“Bibliography”是“文献目录”的意思。因此写作的时候，先在文档底部插入一个Bibliography，然后在行文过程中点击“Insert Citation”即可随时插入参考文献的引用，参考文献将自动加入Bibliography中。由于截图的困难性，图11是我写的一个例子。比如输入“Wasserstein GAN”，再点击“Insert Citation”将弹出Mendeley搜索框，输入文献关键字就能插入文献引用。我想，Microsoft Office的插件使用应大同小异，也是极方便的。

![使用Mendeley插入文献引用的示例](/img/2017-12-01-introduction_to_mendeley/11.png)

Mendeley提供的浏览器插件Web Importer对于即时保存检索到的文献挺有帮助的。图12、13展示了在Web of Science上使用Web Importer的结果。可以看到，作为Endnote的竞品，在Web of Science上使用Web Importer的效果并没有达到预期。比如在图12中，点击插件图标，Web Importer搜索的结果与Web of Science搜索的结果并不完全一致。Web of Science的搜索结果是文献列表，而此时Web Importer给出的却是文献所属期刊列表，这明显不是用户期望的结果。进入文献详情页，再点击插件图标，结果则有些画蛇添足，第一条记录是文献，第二条记录是期刊。再点击文献细节，可以看到基本信息都补全了。

![在Web of Science上使用Mendeley提供的浏览器插件Web Importer，1](/img/2017-12-01-introduction_to_mendeley/12.png)

![在Web of Science上使用Mendeley提供的浏览器插件Web Importer，2](/img/2017-12-01-introduction_to_mendeley/13.png)

在Scopus上使用Web Importer的效果比Web of Science好，直接给出了文献列表，而不是期刊，如图14所示。不得不感慨，毕竟都是ELSEVIER出品。

![在Scopus上使用Web Importer](/img/2017-12-01-introduction_to_mendeley/14.png)

至此，基本完成了对Mendeley文献管理功能的调研。此外，Mendeley还具有社交属性，可进行论文共读。但超出了文献管理的范畴，不再展开。

尽管未接触过Endnote，但是我发现Mendeley基本满足绝大多数用户对文献管理的需求，是一款很好的工具！
