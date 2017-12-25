---
layout:	    post
title:      "[译记]Word Embeddings"
subtitle:   "从 0 到 1"
date:       2017-12-25
author:     "kissg"
header-img: "img/2017-12-25-word_embeddings_0to1/cover.jpg"
comments:    true
mathjax:     true
tags:
    - DL
    - NLP
    - note
---

> 在做一个课程作业, 看了一些 Word Embeddings 相关的论文和博客. 以下内容基本上是看博客的时候摘译的, 写得很好, 恨不得全翻了. 论文笔记反正已经欠了一屁股了:(

## Brief history

* Bengio 2003 年的文章发明了 word embeddings 一词, 此时 word embedding 和模型参数一起训练
* Collobert 2008 年的文章 (A unified architecture for natural language processing) 第一次将 word embeddings 作为后续任务的工具.
* Mikolov 2013 年的两篇文章创造性地提出了 word2vec.
* Pennington 2014 年提出了 GloVe 模型, 预示着 word embeddings 成为 NLP 领域的主流.

## Models

* 在之前的神经网络学习任务中, word embeddings 只是`副产品, by-product`; 后来的 word2vec 等模型则以生成 word embeddings 为直接目标. 两者的主要区别在于:
    1. `计算复杂度 computational complexity`: 用深度神经网络来生成 word embeddings 开销太大; 2013 出现的 word2vec 提出了训练 word embeddings 的简单模型, 计算开销大大减小, 燃. (计算复杂度是 word embeddings 模型的关键之一)
    2. word2vec 和 GloVe 能将语义关系编码进最终的 word embeddings, 这对于需要这一层关系的后续任务是很有帮助的; 常规的神经网络生成 task-specific embeddings, 不适用于其他任务.
* (word2vec, GloVe 之于 NLP, 就向 VGG 之于 CV)
* word embeddings 模型通常使用 [perplexity](https://en.wikipedia.org/wiki/Perplexity) 来评估, 这是一种基于 `交叉熵, cross entropy` 的度量方法.
* `语言模型, language models, LM`通常在给定前 n-1 个词的情况下预测 w_t 的概率 ![LM_probability](/img/2017-12-25-word_embeddings_0to1/LM_probability.png). 基于`链式法则, chain rule` 和`马尔可夫假设, Markov assumption`, 通过计算每个词在给定先行词下的概率的积, 能估计整个句子或文档的积: ![LM_sentence_product.png](/img/2017-12-25-word_embeddings_0to1/LM_sentence_product.png)
* 在基于 n-gram 的 LM 中, 通过单词所在 n-grams 的频率来计算其概率: ![probability_in_ngrams.png](/img/2017-12-25-word_embeddings_0to1/probability_in_ngrams.png)
* 5-gram + Kneser-Ney smoothing 能得到 smoothed 5-gram models, 这是一个 LM 的 strong baseline.
* 在神经网络中, 一般使用 softmax 来计算单词概率: ![softmax_in_NLP.png](/img/2017-12-25-word_embeddings_0to1/softmax_in_NLP.png). h 是输出层前一层的输出向量, v' 是单词对应的 embedding. 此公式表示的神经语言模型结构如下 (Bengio 2006).

![nn_language_model-1.jpg](/img/2017-12-25-word_embeddings_0to1/nn_language_model-1.jpg)

* 推理(测试)时, 每次选择预测概率最大的词(贪婪), 或使用 beam search.

### Bengio's classic neural language model

* Bengio 在 2003 年提出的经典神经语言模型如下所示: 使用了单隐层的前馈神经网络(实线很好理解; 紫色虚线表示通过单词索引乘以共享参数的矩阵 C 得到 word embeddings; 绿色虚线, 还没看过这篇论文的我猜不出). 其目标函数是: ![bengio_language_model_objective.png](/img/2017-12-25-word_embeddings_0to1/bengio_language_model_objective.png)

![bengio_language_model.png](/img/2017-12-25-word_embeddings_0to1/bengio_language_model.png)

* Bengio et al 2003 论文指出, softmax layer 是主要瓶颈, softmax 的计算开销和 vocab 大小成比例.

### C&W model

* Collobert and Weston (C&W) 在 2008 的时候指出, 在充分大的数据集上训练得到的 word embeddings 习得了`句法, syntactic`和`语义, semantic`, 能提成后续任务的性能.
* C&W 使用不同的 objective function 避免了 softmax 的计算开销. 其目标是(相比与错误的单词序列, )对于正确的单词序列输出更高的分数. 为实现此目标, 他们提出来`pairwise ranking criterion`, 像这样 ![c&w_pairwise_ranking_criterion.png](/img/2017-12-25-word_embeddings_0to1/c&w_pairwise_ranking_criterion.png)
* C&W 模型从所有可能的 windows X 中采样正确的 windows x, 再用错误的单词替换掉 x 的中心单词得到 x^(w). 现在来看上面的 objective function, 意在最大化正确单词序列的得分与错误单词序列得分之间的距离, with a margin of 1. (模型如下)

![nlp_almost_from_scratch_window_approach.png](/img/2017-12-25-word_embeddings_0to1/nlp_almost_from_scratch_window_approach.png)

### Word2Vec

* Technically, word2vec 不属于 Deep learning 的范畴, 它的设计理念是简单, 避免高昂的计算开销, 因此它的架构简单不深, 甚至没有使用 non-linear function.
* Mikolov et al 用两篇文章介绍了 word2vec. 在第一篇文章中, 他们提出了两种新的模型: `Continuous bag-of-words, CBOW` 和 `Skip-gram`; 第二篇文章使用了其他策略提高了这两个模型的训练速度和精度.
* CBOW 和 Skip-gram 相对于之前模型的优点在于:
    1. 没有隐层, 避免了隐层带来的开销;
    2. 使得 LM 能学习更多的内容 (得益于 1).

#### CBOW

* LM 只根据 previous words 来预测下一个单词, 对于只是要生成好的 word embeddings 的模型, 不必受限于 previous words. 类似于 C&W 的做法, CBOW 在 target word 的前后各取 n 个单词, 来预测它. (Bag-of-Word 方法不考虑 bags 的顺序, CBOW 也是如此. 此处的 Continuous 表示模型使用 continuous representations)

![cbow.png](/img/2017-12-25-word_embeddings_0to1/cbow.png)

* CBOW 的 objective function 和 LM objective 略有不同, 它预测的是中心单词: ![cbow_objective.png](/img/2017-12-25-word_embeddings_0to1/cbow_objective.png)

#### Skip-gram

* Skip-gram 的做法和 CBOW 正好相反, 它先确定一个 target word, 然后预测邻近的单词. 其结构如下:

![skip-gram.png](/img/2017-12-25-word_embeddings_0to1/skip-gram.png)

* Skip-gram 的 objective function 很自然地表示为所有邻近单词的概率和: ![skip-gram_objective.png](/img/2017-12-25-word_embeddings_0to1/skip-gram_objective.png)
* 给定 target word w_t, w_{t+j} 的条件概率计算与上类似: ![skip-gram_conditional_probability.png](/img/2017-12-25-word_embeddings_0to1/skip-gram_conditional_probability.png). 与上面公式使用隐层状态 h 不同的是, skip-gram 模型没有隐层, 因此直接用输入的 word embedding. 因此, 此处计算的是 input word embedding 与 output word embedding 的内积

#### GloVe

* 简单地说, GloVe 通过显式地将`意义, meaning` encode 成 embeddings space 中 vector 的`偏移量, offset`. (`Skip-gram with negative sampling`, SGNS 隐式地做了这项工作).
* GloVe 作者表明`两个单词同时出现概率的比例, the ratio of the co-occurrence probabilities of 2 words`包含了某种信息, glove 旨在将这种信息 encode 为 `向量的差, vector differences`.
* 为此, 他们提出了一个`加权最小二乘, weighted least squares` objective function J, 直接最小化 A. **两个单词的 vectors 的点积**与 B. **它们同时出现的次数的对数** 的差:

![glove_objective.png](/img/2017-12-25-word_embeddings_0to1/glove_objective.png)

* 式中, w_i, b_i 是单词 i 的词向量和 bias, wtilde_j, b_j 是单词 j 的 convtext word vector 和 bias. X_ij 是 i 出现在 j 的context 中的次数. f 是一个 weighting function, 它为很少同时出现和频繁同时出现的情况赋一个较低的权值.
* `co-occurrence` 次数可直接 encode 进 word-context co-occurence matrix. GloVe 使用该矩阵, 而不是整个 corpus 作为输入.

## Word embeddings vs. distributional semantics models

* `Distributional Semantics Models, DSM` 通过操作 co-occurrence matrixs 来统计单词间的 co-occurrence. 而 neural word embedding models 尝试着预测单词
* Levy et al (2015) 将 GloVe 被视作预测模型, 同时它又利用了 word-context co-occurrence  matrix, 有点像传统方法, 如 PCA, LSA. Levy et al还演示了 word2vec 隐式地 `分解, factorize`了 word-contxt PMI matrix.
* 所以, DSMs 和 word embedding models 表面上看上去使用了不同的算法来学习 word representations, 本质上却都是基于数据统计量的, 即单词间的 co-occurrence counts.

### Models

* `Pointwise Mutual Information, PMI`是一种度量两个单词间`关联强度, strength of association` 的常用方法, 定义为两个单词的`联合概率, jointly probability`与`边缘概率, marginal probabilities`点积的`对数比, log ratio`: ![PMI_equation.png](/img/2017-12-25-word_embeddings_0to1/PMI_equation.png). 当两个单词从来没有同时出现过, P(w, c)=0, PMI(w, c)=log0=-∞. 实际应用时, PMI 常被替换为 `Positive PMI, PPMI`, PPMI(w, c)=max(PMI(w, c), 0)
* `奇异值分解, Singular Value Decomposition, SVD`是降维的最常用方法之一, 通过`Latent Semantic Analysis, LSA`被引入 NLP. SVD 将 word-context co-occurrence matrix 写成三个矩阵的积: ![SVD_word_context_co-occurrence_matrix.png](/img/2017-12-25-word_embeddings_0to1/SVD_word_context_co-occurrence_matrix.png). U 和 V 是`正交矩阵, orthonormal matrix`, \Sigma 是特征值按降序排列的`对角矩阵, diagonal matrix`. 实际应用中, SVD 常被用于因子化 PPMI 得到的矩阵. 一般而言, \Sigma 中只有 top d 个元素会被保留, 从而得到了 ![SVD_W_topd.png](/img/2017-12-25-word_embeddings_0to1/SVD_W_topd.png) 和 ![SVD_C_topd.png](/img/2017-12-25-word_embeddings_0to1/SVD_C_topd.png), 分别用作 word representation 和 context representation.
* Word2vec 介绍了 3 种`预处理, pre-processing` corous 的方法, 同样可用于 DSMs
    1. `Dynamic context window`. 在传统 DSMs 中, context window 是不带权的, 固定大小的. 考虑到更接近的单词一般具有更重要的意义, SGNS 和 GloVe 将更多的权值赋给了更近的单词. SGNS 的 windows size 不固定, 训练时, 在 1 和预设最大 window size 中`均匀采样, sample uniformly`.
    2. `Subsampling frequent words`. SGNS 以概率 ![sgns_subsampling_frequent_words.png](/img/2017-12-25-word_embeddings_0to1/sgns_subsampling_frequent_words.png)(t 为阈值, f 为词频) 剔除单词来 dilutes (稀疏的意思) 高频单词. subsampling 在创建 windows 之前执行, 因此 SGNS 实际使用的 context windows 要大于 context window size 指示的大小 (这就是因为前面说的高频词汇被挖掉了, 比如 The dog is chasing a ball, context window size=2时, 在挖掉高频词汇 the, is, a 后, ball 就在 dog 的 context 中了).
    3. `Deleting rare words`. 在 SGNS 的预处理中, 罕用单词在创建 context windows 前被删除, 这更加增大了 context windows 的实际大小. (Levy et al (2015) 发现这并不能带来显著的性能提升)(context window 增大的例子与上类似)
* PMI 被证明是度量单词间`关联性, association`的有效`度量, metric`. Levy & Goldberg (2014) 证明了 SGNS 隐式分解了 PMI matrix, 由此, 以下源于此的变种可以引入 PMI 中:
    1. `Shifted PMI`. 在 SGNS 中, negative samples 数 k 会影响 PMI matrix 的`漂移, shift`, 即参数 k 会将 PMI 值漂移 log k. 将这一特性应用于 PMI, 就得到了 `Shifted PPMI, SPPMI`: ![SPPMI_equation.png](/img/2017-12-25-word_embeddings_0to1/SPPMI_equation.png)
    2. `Context distribution smoothing`. 在 SGNS 中, negative samples 从 `smoothed unigram distribution` 中采样. 所谓 smoothed unigram distribution 就是 unigram distribution 的 \alpha 次幂, 通常取 \alpha=3/4. 这导致高频单词被采样的次数比少于它们的词频. 将这一特性应用于 PMI, 可以通过将 context words 的频率修改为原先的 \alpha 次幂倍: ![context_distribution_smoothing_to_PMI_1.png](/img/2017-12-25-word_embeddings_0to1/context_distribution_smoothing_to_PMI_1.png), 其中 ![context_distribution_smoothing_to_PMI_2.png](/img/2017-12-25-word_embeddings_0to1/context_distribution_smoothing_to_PMI_2.png), f 是词频函数.
* 3 种修改 word vectors 的`后处理, post-processing`方法:
    1. `Adding context vectors`. GloVe 的作者提出了将 word vectors 和 context vectors 相加作为 output vectors 的方法, 即 ![adding_context_vectors.png](/img/2017-12-25-word_embeddings_0to1/adding_context_vectors.png). 这种做法将`一阶相似项, first-order similarity terms`相加起来. 不能应用于 PMI, 因为 PMI 生成的向量是`稀疏的, sparse`.
    2. `Eigenvalue weighting`. 前文提到, SVD 能生成 ![SVD_W_topd.png](/img/2017-12-25-word_embeddings_0to1/SVD_W_topd.png) 和 ![SVD_C_topd.png](/img/2017-12-25-word_embeddings_0to1/SVD_C_topd.png). 其中 C 阵是正交的, W 阵不是. 可以用额外的参数 p 来为特征值矩阵加权, 调整 W 阵: ![weighted_SVD_W_topd.png](/img/2017-12-25-word_embeddings_0to1/weighted_SVD_W_topd.png)
    3. `Vector normalisation`.
* Levy et al. 发现 SVD 而不是任何 word embeddings 算法在相似性任务上取得最好的效果, SGNS 在类比任务上效果最好. 他们同时指出:
    1. hyperparameters 的设置通常比算法的选择更重要. 没有任何算法总是表现地比其他方法更优.
    2. 在更大的 corpus 上训练, 对于一些任务有帮助. 但超过 1/2 的情况下, 调整超参数的效果带来的提升更大.
* 只要 hyperparameters 调得好, 没有哪种方法总是优于其他方法; SGNS 在各种任务中都优于 GloVe; CBOW 在任何任务中都不如 SGNS

### Recommendations

* 不要对 SPPMI 使用 SVD;
* 不要使用"自以为是"地使用 SVD, 比如不使用 eigenvector weighting (不使用 eigenvalue weighting (p=0.5), 性能下降了 15 %)
* 对 PPMI 和 SVD 使用 short context (比如 window size of 2)
* 对 SGNS 使用多 negative samples
* 对所有方法都使用 context distribution smoothing (用 \alpha=0.75 对 unigram distribution 求幂)
* 使用 SGNS 作为 baseline (SGNS, robust, fast and cheap to train)
* 试着向 SGNS 和 GloVe 加入 context vectors


## Reference

* [On word embeddings - Part 1](/img/2017-12-25-word_embeddings_0to1/http://ruder.io/word-embeddings-1/index.html)
* [On word embeddings - Part 3: The secret ingredients of word2vec](/img/2017-12-25-word_embeddings_0to1/http://ruder.io/secret-word2vec/index.html)
