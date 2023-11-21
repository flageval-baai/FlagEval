# C-Sem

[English](./README.md)

## 研究背景

在自然语言处理领域的研究和应用中，语义理解被视为关键基石。然而，当前在中文大语言模型评测领域，仍然比较缺乏从语言学角度出发的公开评测基准。
 
**北京大学与闽江学院作为FlagEval旗舰项目的共建单位，合作构建了 C-SEM（Chinese SEMantic  evaluation dataset）语义评测基准数据集。** 

C-SEM 针对当前大模型的可能存在的缺陷和不足，创新地构造多种层次、多种难度的评测数据， 并参考人类的语言思维习惯，考察模型在理解语义时的“思考”过程。当前开源的  C-SEM v1.0版本共包括四个子评测项，分别从词汇、句子两个级别评测模型的语义理解能力，通用性较强，适用于研究对照。
 
当前 C-SEM 的子评测项分别为词汇级的语义关系判断（LLSRC）、句子级别的语义关系判断（SLSRC）、词汇的多义理解问题（SLPWC），以及基础修饰知识检测（SLRFC）。后续  C-SEM 评测基准将会持续迭代，覆盖更多语义理解相关知识，形成多层次的语义理解评测体系。同时，FlagEval 大模型评测平台将在第一时间集成最新版本，加强对大语言模型的中文能力评测的全面性。
 
注：为了确保评测结果公平公正、防范评测集泄露的风险，**FlagEval 官网（[flageval.baai.ac.cn](https://flageval.baai.ac.cn/#/trending))** 采用的  C-SEM 评测集将保持与开源版本的异步更新。当前FlagEval 采用最新版本相较于开源版本而言，题目数量更多，题目形式更为丰富，并参考HELM工作采用5-shot的形成进行评测。


## 1、词汇级的语义关系判断（LLSRC）
Lexical Level Semantic Relationship Classification
 
本类数据用于测试模型对两个词之间的语义理解程度，要求模型在没有上下文的情况下，对两个独立的单词（或短语）之间可能的语义关系进行判断，例如“上-下位”，“整体-部分”，“近义”，“反义”等语义关系。
样例如下:

> 问题：“呆板”与“灵活” 这两个词语具有以下哪种语义关系？从下面4项中选择
> 
> A. 近义
> 
> B. 反义
> 
> C. 上下位
> 
> D. 整体部分 

## 2、句子级别的语义关系判断（SLSRC）

Sentence Level Semantic Relationship Classification
 
本类数据用于测试模型是否理解单词在特定的上下文中的语义，要求模型根据给定的上下文，回答特定词的语义关系。
样例如下：

> 问题：“笔尖的力量在我的手中化作了思想的火花，点燃了梦想的火炬。”这句话中“笔尖”与下列哪个词具有整体部分关系？
> 
> A. 笔画
> 
> B. 笔墨
> 
> C. 尖利
> 
> D. 钢笔
 
## 3、词汇的多义理解判断（SLPWC）

Sentence Level Polysemous Words Classification
 
本类数据用于测试模型是否理解对于“一词多义”。测试形式是给出同一个词，以及所处的不同上下文，期望模型能够区分语义差异。
样例如下：

> 问题：以下哪句话中“中学”的意思(或用法)与其他句子不同。
> 
> A. 中学教育在塑造青少年的品德、知识和技能方面起着重要的作用。
> 
> B. 曾纪泽、张自牧、郑观应、陈炽、薛福成等大抵讲“中学为体，西学为用”的人，无不持“西学中源”说。
> 
> C. 中学是为了培养青少年的综合素质而设立的教育机构。
> 
> D. 我们的学校是一所提供中学教育的优秀学校，致力于为学生提供高质量的教育和培养。
 
## 4、基础修饰知识判断（SLRFC）

Sentence Level Rhetoric Figure Classification,
 
本类数据用于测试模型是否能够判断句子的修饰用法。比喻、排比、拟人、反问等这些基础修饰手法在人们日常表达中经常使用，优秀的大语言模型也应该具备相应的能力和知识。
 
样例如下：

> 问题：以下哪个句子使用了拟人修辞手法？
> 
> A. 因为有了你，在生命的悬崖前，我不曾退缩过，因为有了你，在坠入深渊时，我始终都有挣扎向上的勇气与力量，因为有了你，珠穆琅玛峰上才会出现我的足迹，因为有了你，在阴暗的道路上行走我都不会感到丝毫的害怕，心头总暖暖的……
> 
> B. 春天是个害羞的小姑娘，遮遮掩掩，躲躲藏藏，春天是出生的婴儿，娇小可爱。
> 
> C. 他的思维如同一条蜿蜒的小溪，总是能找到通往解决问题的路径。
> 
> D. 月亮悬挂在夜空中，犹如一颗璀璨的珍珠镶嵌在黑天幕上。
