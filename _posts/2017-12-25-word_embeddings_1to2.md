---
layout:	    post
title:      "[译记]Word Embeddings"
subtitle:   "从 1 到 2"
date:       2017-12-25
author:     "kissg"
header-img: "img/2017-12-25-word_embeddings_0to1/cover2.jpg"
comments:    true
mathjax:     true
tags:
    - DL
    - NLP
    - note
---

## Softmax in Word Embeddings

* softmax 的高计算开销在于需要计算 hidden state ***h*** 和所有单词的 output word embeddings 的内积, 求和作为分母.

### Hierarchical Softmax

* `Hierarchical softmax, 分层 Softmax, H-Softmax` 受`二叉树, binary tree`的启发而提出. 它用一个 hierarchical layer 替换 flat softmax layer, 每个单词都是一个`叶子节点, leaves`.

![hierarchical_softmax_example.png](/img/2017-12-25-word_embeddings_0to1/hierarchical_softmax_example.png)

* H-softmax 将一个单词概率的计算 decompose 成概率计算的序列, 按照到叶子所在叶节点的路径计算其概率即可, 避免了对所有单词概率做归一化的开销. H-Softmax 能将 word prediction 任务加速至少 50 倍!
* `平衡二叉树, balanced binary tree` 的深度为 ![log2_V.png](/img/2017-12-25-word_embeddings_0to1/log2_V.png). 因此最多只需要估计 ![log2_V.png](/img/2017-12-25-word_embeddings_0to1/log2_V.png) 步计算, 就能得到单词的最终概率. 并且得到的概率是归一化后的, 因为二叉树所有叶节点的概率和为 1.
* 每一对子节点都将父节点的概率一分为二. 因此, 遍历树时, 需要计算`左枝, left branch`和`右枝 right branch`的概率. 比如在节点 n 上选择右枝的概率为: ![h-softmax_probability_turn_right.png](/img/2017-12-25-word_embeddings_0to1/h-softmax_probability_turn_right.png). (下图是 h-softmax 的一个例子, sigm 即 sigmoid 函数)

![hierarchical_softmax.png](/img/2017-12-25-word_embeddings_0to1/hierarchical_softmax.png)
* 在 H-softmax 中, 不再需要 output embeddings, 取而代之的是, 每个节点都有 embeddings, 各不相同. 因此实际上 H-softmax 和 softmax 拥有差不多的参数量.
* 树结构对于 h-softmax 很重要, 比如为相似的概率分配相似的路径. 基于此, Morin & Bengio 在 WordNet 中使用了`同义词集, synsets` 作为树的`聚类 cluster`, 但效果不如 softmax; Minh & Hinton 使用聚类算法来学习树结构, 通过递归地将单词划分仅两个聚类, 实现了和 softmax 同样的性能
* Notably, 只能在训练阶段获得加速, 因为此时事先知道了 target word, 也就知道了其在树中的路径. 测试时, 要计算最有可能的单词, 仍需要计算所有单词的概率.

#### information content of word

* 一个单词的`信息量, information content`是它的概率的`负对数, negative logarithm`: ![word_information_content.png](/img/2017-12-25-word_embeddings_0to1/word_information_content.png)
* 而 Corpus 中所有单词的熵 H, 就是所有单词的 information content 的期望, 即: ![entropy_of_corpus.png](/img/2017-12-25-word_embeddings_0to1/entropy_of_corpus.png)
* 另外, 也可以用数据的平均的比特长度作为熵. 在 balanced bianry tree 中, 对于每个单词都一视同仁, 每个单词的概率都相同, 因此单词的熵等于其信息量. 套用以上公式, vocab_size=10000 的 balanced binary tree的平均单词熵为: ![entropy_10000_bbt.png](/img/2017-12-25-word_embeddings_0to1/entropy_10000_bbt.png)
* 利用好树结构, 可以使用`哈夫曼树 Huffman tree`代替平衡二叉树, 为信息量更少的单词赋更短的路径, 能取得更好的效果. 一些单词会以更大的概率出现, 可以认为它们携带的信息量更少. (作为常识: 太阳东升西落, 携带的信息量是很少的) 此时, 同样的 10000 个单词的平均熵是 9.16, 能带来 31 % 的加速. (Morin & Bengio 论文中的实例, 其他训练集的值会不同, 加速效果不知道怎么算的)
* 上文提到 LM 通常用 `perplexity` 来度量, 它可以表示为 2^H. Jozefowicz et al 2016 提出的 LM 取得了 state-of-the-art perplexity 24.2, 这意味着一个英语单词平均只需要 4.6 bits. 和香农爸爸实验得出的结论, 英语字母的信息率的下界在 0.6 到 1.3 bits之间, 单词的平均长度在 4.6 bits 很接近!

### Differentiated Softmax

* Chen et al 提出了 `Differentiated softmax, D-softmax`. 它的动机是, 不是所有单词都需要相同数量的参数, 频繁使用的单词需要更多的参数, 而极少使用的单词可能只需要少量参数去 fit.
* softmax 的矩阵大小为 d x vocab_size (d 为 embeddings 的维度). 鉴于以上思想, D-softmax 使用了`稀疏矩阵, sparse matrix`, 根据词频, 将 embeddings 划分进不同的 blocks, 每个 blocks 中的 embeddings 都具有相同的维度. blocks 数和 embddings size 是可调节的超参数.

![differentiated_softmax_1.png](/img/2017-12-25-word_embeddings_0to1/differentiated_softmax_1.png)

* 上图是 D-softmax 的稀疏矩阵的一个示例 (高度为 vocab size, 宽度为embedding size). block A 是常用词的 embedding block, 因为它分配到的参数更多. 图中非阴影部分全部置 0, 不学习这些位置的参数.
* 应用 D-softmax 时, 输出层的前一层 ***h*** 用于`特征级联, feature concatenation`, 上图右, ***h*** 的高度是 dA+dB+dC, 分别对应各 blocks 的维度. 因此计算 output embeddings 时, 每个 partition 仅与其对应的 h 的部分求内积.
* D-softmax 在测试时同样适用, 也能起到加速效果. 论文指出, D-softmax 在测试阶段是最快的方法, 并且精度也属于第一梯队.
* D-softmax 的缺点是, 词频最低的那部分单词的参数太少, 对这些单词的建模效果很差.

### CNN-softmax

* Kim et al. 提出了通过 character-level CNN 生成 input word embeddings 的方法. 受此启发, Jozefowicz et al 2016 将 character-level CNN 应用于 output word embeddings, 称为 `CNN-softmax`

![cnn-softmax_1.png](/img/2017-12-25-word_embeddings_0to1/cnn-softmax_1.png)

* cnn-softmax 也要做 normalization, 但通过 CNN, 模型的参数数量大大减少了. softmax 需要保存 d x vocab_size 大小的 embedding matrix, 现在只需要 CNN 的参数量. (对比 FC 与 CONV)
* 测试时, output word embeddings 可以预先计算出来, 因此不存在性能损失
* 由于在`连续空间, continuous space`表示字符, 导致模型趋向于学习更平滑的 character to word 的 mapping, 后果是 character-based models 很难区分拼写相近但具有不同意义的单词. 为解决该问题, 作者增加了一个 correction factor, learned per word, 显著减小了 softmax 和 cnn-softmax 的性能差. correction factor 的维度是一个超参数, 在模型大小和性能间有一个 `trade-off, 平衡`.
* Jozefowicz et al 还指出可以用 character-level LSTM 来代替 character-level CNN. 他们没有成功地应用 character-level LSTM, Ling et al 在 `机器翻译, Machine Translation` 任务中使用了类似层, 取得了很好的结果.

> 以上 H-softmax, D-sofmax, CNN-softmax 都属于`Softmax-based approaches`. 以下将介绍`Sampling-based approaches`

* Sampling-based approaches 完全摈弃掉了 softmax layer, 它们通过近似 softmax 的分母来实现 normalization. 这只对于训练阶段有用, 推理时仍需要 softmax 来计算 normalized probability.
* 训练时, 其目标是 minimize the cross entropy, 亦即输出的 softmax 的负对数. [Karpathy 大佬关于 softmax 与 cross entropy 间的联系的解释](https://cs231n.github.io/linear-classify/#softmax-classifier)
* 如上所述, 模型的 objective function 是这样的: ![j_theta_1.png](/img/2017-12-25-word_embeddings_0to1/j_theta_1.png). 对其进行一系列的变形:
    1. 将负对数改写成和的形式: ![j_theta_2.png](/img/2017-12-25-word_embeddings_0to1/j_theta_2.png);
    2. 将点积替换为 -\epsilon(w): ![j_theta_3.png](/img/2017-12-25-word_embeddings_0to1/j_theta_3.png)
    3. 为了 BP, 计算梯度: ![gradient_of_j_theta_1.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_1.png)
    4. 根据 log(x) 的导数是 1/x (如无特殊说明, log 默认为自然对数), 使用链式法则再变形: ![gradient_of_j_theta_2.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_2.png)
    5. 将第二项的梯度移到求和内部: ![gradient_of_j_theta_3.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_3.png)
    6. 根据 exp(x) 的导数还是 exp(x), 再次使用链式法则, 变形得到: ![gradient_of_j_theta_4.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_4.png)
    7. 改写上式, 得到: ![gradient_of_j_theta_5.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_5.png)
    8. 观察到第 7 步得到的式子中有一个 softmax 形式的项, 用 P(wi) 替换之, 得到: ![gradient_of_j_theta_6.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_6.png)
    9. 将第二项中的负号提到求和之前, 就得到了: ![gradient_of_j_theta_7.png](/img/2017-12-25-word_embeddings_0to1/gradient_of_j_theta_7.png)
* Bengio & Senécal (2003) 表明, 该梯度由两部分组成: 第一项是`positive reinforcement` for the target word, 第二项是其他所有单词的 `negative reinforcement`, P(wi) 表示依据他们各自的概率加权. negative reinforcement 刚好也是 wi 服从 P(wi) 分布时, \epsilon(wi) 梯度的期望: ![expection_of_the_gradient_of_epsilon_for_all_words.png](/img/2017-12-25-word_embeddings_0to1/expection_of_the_gradient_of_epsilon_for_all_words.png)
* 大多数 samping-based 方法都通过近似 negative reinforcement 来简化计算.

### Importance Sampling

* 通过`蒙特卡罗, Monte Carlo, MC`方法可近似任意概率分布的期望值, 它就是从概率分布中随机采样, 求均值作为期望值. (依据是大数定理)
* 使用上述方法, 需要知道概率分布 P 才能从中进行采样, 而计算 P 正是我们所不愿的. 替代的方法是, 使用找到其他分布 Q (称为`proposal distribution`), it's cheap to sample from Q, 就以 Q 作为 MC 采样的基础. 在 LM 中, 可以简单地使用训练集的 `unigram` 分布作为 Q.
* `古典, classical``重要性采样, Importance Sampling, IS`的核心内容就是: 通过从 proposal distribution Q 中进行 MC 采样, 来近似 target distribution P. 这仍然需要计算每个单词的概率 P(w). 为避免这样的计算, Bengio & Senécal (2003) 使用了 `biased estimator`. 当 P(w) 由乘积得时, 可以使用 biased estimator.
* 使用 biased estimator, 将原来 negative reinforcement 的中权重 P(wi) 替换为 ![biased_IS_factor.png](/img/2017-12-25-word_embeddings_0to1/biased_IS_factor.png), 这样就利用了 Q. 其中 ![biased_IS_factor_item_1.png](/img/2017-12-25-word_embeddings_0to1/biased_IS_factor_item_1.png), R 则等于 r(wj) 的和.
* 使用 biased_IS, 仍然需要计算 softmax 的分子, 但分母的计算使用了更易计算的 Q.
* 基于采样估计概率分布的一个问题是, 近似的准确度与样本数量相关. 样本太少时, P 会偏离 Q, 进而导致模型发散. Bengio & Senécal 提出了一个估计有效样本数量的方法.
* 基于 IS 的方法相比 softmax, 加速了 19 倍.

### Adaptive Importance Sampling

* Bengio & Senécal (2008) 表示, 为了防止训练后期, unigram distribution Q 偏离 P, 可将 Q 替换为更复杂的分布. 并提出了 n-gram distribution, 在训练过程中能适应性地做出改变, 从而紧紧拟合 P. 他们使用一个`混合函数, mixture function`来在 bigram distribution 和 unigram distribution 中做 interpolation, 最小化 Q 和 P 间的 KL 散度. (不知道论文中是怎么处理这里的 P, 感觉是病态的方法) 实验表明, AIS 加速了 100 倍.

### Target Sampling

* Jean et al (2015) 在显存受限的情况下, 提出了限制 target words 采样数量的方法 (还是采用 AIS): 对训练集进行划分, 每份包含固定数量的样本单词. 这也意味着对 Q 的划分, 对应子集的所有单词具有相等的非等概率, 其他所有单词概率为 0.

### Noise Contrastive Estimation

* Mnih & Teh 将`Noise Contrastive Estimation, NCE`作为更稳定的采样方法 (IS 可能会发散). NCE 不直接估计单词概率, 它使用一个同样旨在最大化正确单词概率的`辅助损失, auxiliary loss`.
* NCE 使用与 C&W (2008) 的 pairwise-ranking 类似的方法: 训练模型区分 target word 与 noise. 任务退化为二分类问题: 对于每个单词 wi, 给定其 context, 即 n previous words, 同时为其生成 k 个 noise samples. context 中的所有单词赋标签 1, noise samples 标记为 0.

![negative_sampling.png](/img/2017-12-25-word_embeddings_0to1/negative_sampling.png)

* 此时的 ojbective 为: ![nce_objective_1.png](/img/2017-12-25-word_embeddings_0to1/nce_objective_1.png). 为避免对 noise samples 期望的计算, 使用 MC 采样的均值作为期望的近似, 于是 objective 变形为: ![nce_objective_2.png](/img/2017-12-25-word_embeddings_0to1/nce_objective_2.png)
* 为每一个单词都采样 k 个 noise samples, 因此采样概率可以表示为 P 和 Q 的混合形式: ![mixture_P_and_Q.png](/img/2017-12-25-word_embeddings_0to1/mixture_P_and_Q.png)
* 用模型概率 P 来表示真实概率 P_train, 那么从混合模型中采样, 样本是正确单词的概率为: ![nce_true_conditional_probability.png](/img/2017-12-25-word_embeddings_0to1/nce_true_conditional_probability.png)
* 上式中, 条件概率 P(w\|c) 正是 softmax 的结果. 为了避免计算, 将分母改写为关于 c 的函数 Z(c) (因为分母只取决于 h). 在 NCE 中, 可以将 Z(c) 看作模型可以学习的参数. Mnih & Teh (2012) 和 Vaswani et al 将其固定为 1. 根据他们的说法, 这不影响性能. 该方法不仅避免了计算, 还减少了模型参数. Zoph et al 证实了以上做法的可取性, 实验习得的 Z(c) 接近于 1, 并且方差很小.
* 在 Z(c)=1 的情况下, ![probability_of_word_given_context.png](/img/2017-12-25-word_embeddings_0to1/probability_of_word_given_context.png). 代入 P(y=1\|w, c), 得: ![nce_true_conditional_probability_2.png](/img/2017-12-25-word_embeddings_0to1/nce_true_conditional_probability_2.png). 最后再代入 objective function, 得: ![nce_objective_3.png](/img/2017-12-25-word_embeddings_0to1/nce_objective_3.png)
* NCE 有很好的理论保证: 增大 noise samples 数时, NCE 的导数趋向于 softmax 的梯度. Mnih & Teh 表示 25 个 noise samples 足以达到 softmax 的性能, 但是有 45 倍的加速.
* NCE 的一个缺陷是, 每次训练都需要采样不同的 noise samples, noise samples 及其梯度无法保存在 dense matrixs 中, 就无法利用 dense matrix multiplication 的来加速计算.
* Jozefowicz et al (2016) 和 Zoph et al (2016) 各自独立提出了在一个 mini-batch 中共享 noise samples 的方法, 部分解决了以上问题.
* Jozefowicz et al (2016) 表明 NCE 与 IS 间有强联系. NCE 使用二分类问题, IS 可以描述成类似的形式: IS 使用 softmax 和 cross entropy function 来优化`多分类问题, multi-class classification`. 当 loss 会导致训练数据与 noise samples 间的`tied updates`(不知道是什么, 望解惑), 由于 IS 执行多分类任务, 它比 NCE 更适合 LM.
* Jozefowicz et al (2016) 使用 IS 在 1B Word benchmard 上取得了 state-of-the-art 的性能.

### Negative Sampling

* `Negative sampling, NEG`可以看作 NCE 的近似, 它更简化了 NCE. NCE 随着 noise samples 数的增多, 其目标会接近 softmax. 但 NEG 没有该理论保证, 因为它的目标是学习更好的 word representation 而不是 low perplexity.
* NCE 与 NEG 的关键区别在于, NEG 以尽可能简化计算为目标近似 P(y=1\|w, c), 因此它将 kQ(w) 设为 1, 因此 ![neg_true_conditional_probability.png](/img/2017-12-25-word_embeddings_0to1/neg_true_conditional_probability.png)
* 令 kQ(w)=1 的依据是, P(y=1\|w, c) 可以转换成 sigmoid function 的形式. 代入 objective function, 经过一系列变换, 可以得到: ![neg_objective.png](/img/2017-12-25-word_embeddings_0to1/neg_objective.png)
* 当 noise samples 数等于 vocab size, Q 是 uniform distribution 时, NEG 等价于 NEC. 在其他情况下, NEG 仅仅近似 NCE, 它不直接优化正确单词的 likelihood.
* NEG 对于学习 word embeddings 可能更有帮助, 但它缺少`渐近一致性保证, asymptotic consistency guarantes`, 使得它不适合 LM.

### Self-Normalization

* Self-normalization 并不是 sampling-based 方法. 如前所述, 将 NCE 的Z(c) 设为 1, 模型能 self-normalization.
* 对于 loss function: ![negative_log-likelihood.png](/img/2017-12-25-word_embeddings_0to1/negative_log-likelihood.png), 当限制 Z(c)=1, 或 log(Z(c))=0, 可以避免计算 Z(c) 中的 normalization.
* 基于以上观察, Devlin et al (2014) 提出向 loss function 中加入一个`均方误差惩罚项, squared error penalty term`, 将鼓励模型保持 log(Z(c)) 尽可能接近 0. ![negative_log-likelihood_with_squared_error_penalty](/img/2017-12-25-word_embeddings_0to1/negative_log-likelihood_with_squared_error_penalty.png)
* Devlin et al (2014) 实现了一个 NM symtem, 在 decode 阶段, 他们就设置 softmax 的分母为 1, 只使用 softmax 的分子和惩罚项计算 P(w\|c), 此时损失函数为: ![negative_log-likelihood_with_penalty_term.png](/img/2017-12-25-word_embeddings_0to1/negative_log-likelihood_with_penalty_term.png)
* 根据 Devlin 的论文, 相比 non-self-normalization LM, self-normalizaton 在 BLEU 上 实现了 15 倍的加速, 只有很小的降分.


### Infrequent Normalization

* Andreas & Klein 建议仅仅 normalize 训练样本的一部分就足够了, 仍然能获得近似 self-normalization 的效果, 于是提出了 `Infrequent Normalization, IN`, 它 down-samples the penalty term (这里的 down-sample 不同于 ConvNet 的 subsample), 使之变成一个 sampling-based 方法
* IN 的 loss 被改写成如下形式: ![IN_loss.png](/img/2017-12-25-word_embeddings_0to1/IN_loss.png). 只为子集 C 计算 normalization. gamma 控制子集 C 的大小.
* Andreas & Klein 表示 IN 集合了 NCE 和 self-normalization 的优势, 它不必为所有训练样本计算 normalizatoin, 又可控模型准确度和 normalization 近似度间的平衡. 在他们的实验中, 仅对 1/10 的训练样本进行 normalization, 取得了 10 倍的加速, 但没有明显的性能下降.

以上各方法的比较, 如下表所示. (性能比较是基于 LM 的)

![comparison_of_approaches_of_approximate_softmax_for_LM.png](/img/2017-12-25-word_embeddings_0to1/comparison_of_approaches_of_approximate_softmax_for_LM.png)

## Reference

* [On word embeddings - Part 2: Approximating the Softmax](/img/2017-12-25-word_embeddings_0to1/http://ruder.io/word-embeddings-softmax/index.html)
