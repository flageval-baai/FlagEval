# C-SEM

[简体中文](./README-zh.md)

## Research Background

Semantic understanding is seen as a key cornerstone in the research and application of natural language processing. However, there is still a lack of publicly available benchmarks that approach from a linguistic perspective in the field of evaluating large Chinese language models.

**Peking University and Minjiang College, as co-builders of the FlagEval flagship project, have collaborated to create the C-SEM (Chinese SEMantic evaluation dataset) semantic evaluation benchmark dataset.**

C-SEM innovatively constructs various levels and difficulties of evaluation data to address the potential flaws and inadequacies of current large models. It examines the models' "thinking" process in understanding semantics, referencing human language cognition habits. The currently open-source version, C-SEM v1.0, includes four sub-evaluation items, assessing models' semantic understanding abilities at both the lexical and sentence levels, offering broad applicability for research comparison.

The sub-evaluation items of C-SEM are: Lexical Level Semantic Relationship Classification (LLSRC), Sentence Level Semantic Relationship Classification (SLSRC), Sentence Level Polysemous Words Classification (SLPWC), and Sentence Level Rhetoric Figure Classification (SLRFC). Future iterations of the C-SEM benchmark will continue to evolve, covering more semantic understanding-related knowledge and forming a multi-level semantic understanding evaluation system. Meanwhile, the FlagEval large model evaluation platform will integrate the latest versions promptly to enhance the comprehensiveness of evaluating Chinese capabilities of large language models.

Note: To ensure fair and impartial evaluation results and prevent the risk of evaluation set leakage, the C-SEM evaluation set used on the **FlagEval official website ([flageval.baai.ac.cn](https://flageval.baai.ac.cn/#/trending))** will be asynchronously updated with the open-source version. The current FlagEval version, compared to the open-source one, has more questions, richer formats, and uses a 5-shot formation for evaluation, referencing the HELM approach.


## 1. Lexical Level Semantic Relationship Classification (LLSRC)

This category tests the model's understanding of the semantic relationship between two words without context. It requires the model to judge the possible semantic relationships between two independent words (or phrases), such as "hypernym-hyponym," "whole-part," "synonymy," "antonymy," etc.

Example:

> 问题：“呆板”与“灵活” 这两个词语具有以下哪种语义关系？从下面4项中选择
> 
> A. 近义
> 
> B. 反义
> 
> C. 上下位
> 
> D. 整体部分 

## 2. Sentence Level Semantic Relationship Classification (SLSRC)

This category tests whether the model understands the semantics of words in a specific context. The model is required to answer the semantic relationship of specific words based on the given context. 

Example:

> 问题：“笔尖的力量在我的手中化作了思想的火花，点燃了梦想的火炬。”这句话中“笔尖”与下列哪个词具有整体部分关系？
> 
> A. 笔画
> 
> B. 笔墨
> 
> C. 尖利
> 
> D. 钢笔
 
## 3. Sentence Level Polysemous Words Classification (SLPWC)

This category tests the model's understanding of "one word, multiple meanings." The test format presents the same word in different contexts, expecting the model to distinguish semantic differences.


Example:


> 问题：以下哪句话中“中学”的意思(或用法)与其他句子不同。
> 
> A. 中学教育在塑造青少年的品德、知识和技能方面起着重要的作用。
> 
> B. 曾纪泽、张自牧、郑观应、陈炽、薛福成等大抵讲“中学为体，西学为用”的人，无不持“西学中源”说。
> 
> C. 中学是为了培养青少年的综合素质而设立的教育机构。
> 
> D. 我们的学校是一所提供中学教育的优秀学校，致力于为学生提供高质量的教育和培养。



## 4. Sentence Level Rhetoric Figure Classification (SLRFC)

This category tests whether the model can judge the rhetorical use of sentences. Basic rhetorical techniques like metaphor, parallelism, personification, rhetorical question, etc., are commonly used in everyday expression, and excellent large language models should also possess corresponding abilities and knowledge.

Example:

> 问题：以下哪个句子使用了拟人修辞手法？
> 
> A. 因为有了你，在生命的悬崖前，我不曾退缩过，因为有了你，在坠入深渊时，我始终都有挣扎向上的勇气与力量，因为有了你，珠穆琅玛峰上才会出现我的足迹，因为有了你，在阴暗的道路上行走我都不会感到丝毫的害怕，心头总暖暖的……
> 
> B. 春天是个害羞的小姑娘，遮遮掩掩，躲躲藏藏，春天是出生的婴儿，娇小可爱。
> 
> C. 他的思维如同一条蜿蜒的小溪，总是能找到通往解决问题的路径。
> 
> D. 月亮悬挂在夜空中，犹如一颗璀璨的珍珠镶嵌在黑天幕上。
