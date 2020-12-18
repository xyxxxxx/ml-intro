[toc]

# 认知神经学中的注意力

注意力是一种人类不可或缺的复杂认知功能，指人可以在关注一些信息的同时忽略另一些信息的选择能力。在日常生活中，我们通过视觉、听觉、触觉等方式接收大量的感觉输入。但是人脑还能在这些外界的信息轰炸中有条不紊地工作，是因为人脑可以有意或无意地从这些大量输入信息中选择小部分的有用信息来重点处理，并忽略其他信息。这种能力就叫作**注意力(attention)**。注意力可以作用在外部的刺激（听觉、 视觉、 味觉等），也可以作用在内部的意识（思考、回忆等）。

注意力一般分为两种：

1. 自上而下的有意识的注意力，称为**聚焦式注意力(focus attention)**。聚焦式注意力是指有预定目的、依赖任务的，主动有意识地聚焦于某一对象的注意力。
2. 自下而上的无意识的注意力，称为**基于显著性的注意力(saliency based attention)**。基于显著性的注意力是由外界刺激驱动的注意，不需要主动干预，也和任务无关。如果一个对象的刺激信息不同于其周围信息，一种无意识的 “赢者通吃”(Winner-Take-All)或者门控(gating)机制就可以把注意力转向这个对象。不管这些注意力是有意还是无意，大部分的人脑活动都需要依赖注意力，比如记忆信息、阅读或思考等。

一个和注意力有关的例子是<u>鸡尾酒会效应</u>. 当一个人在吵闹的鸡尾酒会上和朋友聊天时，尽管周围噪音干扰很多，他还是可以听到朋友的谈话内容，而忽略其他人的声音（即<u>聚焦式注意力</u>）。同时，如果背景声中有重要的词（比如他的名字），他会马上注意到（即<u>显著性注意力</u>）。

聚焦式注意力一般会随着环境、情景或任务的不同而选择不同的信息。比如当要从人群中寻找某个人时，我们会专注于每个人的脸部；而当要统计人群的人数时，我们只需要专注于每个人的轮廓。

提出多头注意力机制



# 注意力机制

在计算能力有限的情况下，**注意力机制(attention mechanism)**作为一种资源分配方案，将有限的计算资源用来处理更重要的信息，是解决信息超载问题的主要手段。

当用神经网络来处理大量的输入信息时，也可以借鉴人脑的注意力机制，只选择一些关键的信息输入进行处理，来提高神经网络的效率。在目前的神经网络模型中，我们可以将最大汇聚(max pooling)、门控(gating)机制近似地看作自下而上的基于显著性的注意力机制。除此之外，自上而下的聚焦式注意力也是一种有效的信息选择方式。以阅读理解任务为例，给定一篇很长的文章，然后就此文章的内容进行提问，提出的问题只和段落中的一两个句子相关，其余部分都是无关的。为了减小神经网络的计算负担，只需要把相关的片段挑选出来让后续的神经网络来处理，而不需要把所有文章内容都输入给神经网络。

用$$X = [\pmb x_1 , ⋯ , x_N ] ∈\mathbb{R}^{D×N}$$表示$$N$$组输入信息，其中$$D$$维向量$$\pmb x_n ∈
\mathbb{R}^D, n ∈ [1, N]$$  表示一组输入信息。为了节省计算资源，不需要将所有信息都输入神经网络，只需要从$$X$$中选择一些和任务相关的信息。注意力机制的计算可以分为两步：一是在所有输入信息上计算注意力分布，二是根据注意力分布来计算输入信息的加权平均。

**注意力分布**

为了从$$N$$个输入向量$$[\pmb x_1 , ⋯ ,\pmb x_N ]$$中选择出和某个特定任务相关的信息，我们需要引入一个和任务相关的表示，称为**查询向量(query vector)**，并通过一个打分函数来计算每个输入向量和查询向量之间的相关性。

给定一个和任务相关的查询向量$$\pmb q$$，我们用注意力变量$$z ∈ [1, N]$$来表示被选择信息的索引位置，即$$z = n$$表示选择了第$$n$$个输入向量。为了方便计算，我们采用一种“软性”的信息选择机制。首先计算在给定$$\pmb q$$和$$X$$下，选择第$$i$$个输入向量的概率$$ α_n $$，
$$
\alpha_n=p(z=n|X,\pmb q)\\
={\rm softmax}(s(\pmb x_n,\pmb q))\\
=\frac{\exp(s(\pmb x_n,\pmb q))}{\sum_{j=1}^N \exp(s(\pmb x_j,\pmb q))}
$$
其中$$α_n$$称为**注意力分布(attention distribution)**，$$s(\pmb x,\pmb q)$$为**注意力打分函数**，可以使用以下几种方式来计算：
$$
\begin{align}
加性模型/{\rm additive}/{\rm perceptron}&\quad s(\pmb x,\pmb q)=\pmb v^{\rm T}\tanh(W \pmb x+U\pmb q)\\
点积模型/{\rm dot\ product}&\quad s(\pmb x,\pmb q)=\pmb x^{\rm T}\pmb q\\
缩放点积模型/{\rm scaled\ dot\ product}&\quad s(\pmb x,\pmb q)=\frac{\pmb x^{\rm T}\pmb q}{\sqrt{D}} \\
双线性模型/{\rm general}&\quad s(\pmb x,\pmb q)=\pmb x^{\rm T}W\pmb q
\end{align}
$$
其中$$W,U,\pmb v$$为可学习的参数，$$D$$为输入向量的维度。

理论上，加性模型和点积模型的复杂度差不多，但是点积模型在实现上可以更好地利用矩阵乘积，从而计算效率更高。当输入向量的维度$$D$$比较高时，点积模型的值通常有比较大的方差，从而导致Softmax函数处于梯度非常小的位置。因此，缩放点积模型可以较好地解决这个问题[Vaswani et al., 2017]。

双线性模型是一种泛化的点积模型，假设上式中$$W = U^{\rm T} V$$，双线性模型可以写为$$s(\pmb x,\pmb q) =\pmb x^{\rm T}U^{\rm T}V\pmb q = (U\pmb x)^{\rm T} (V\pmb q)$$, 即分别对$$\pmb x$$和$$\pmb q$$进行线性变换后计算点积。相比点积模型，双线性模型在计算相似度时引入了非对称性。

**加权平均**

注意力分布$$α_n$$可以解释为在给定任务相关的查询$$\pmb q$$时，第$$n$$个输入向量受关注的程度。我们采用一种“软性”的信息选择机制对输入信息进行汇总，即
$$
{\rm att}(X,\pmb q)=\sum_{n=1}^N\alpha_n\pmb x_n=E_{z\sim p(z|X,\pmb q)}(\pmb x_z)
$$
上式称为**软性注意力机制(soft attention mechanism**)。下图给出软性注意力机制的示例。

![](https://i.loli.net/2020/11/09/LsnliCVB4IYPUdv.png)

注意力机制通常用作神经网络中的一个组件。



## 注意力机制的变体

### 硬性注意力

软性注意力选择的信息是所有输入向量在注意力分布下的期望。与之相对的，**硬性注意力(hard attention)**只关注某一个输入向量。硬性注意力有两种实现方式：

1. 选取概率最高的一个输入向量，即
   $$
   {\rm att}(X,\pmb q)=\pmb x_{\hat n}
   $$
   其中$$\hat n = \arg\max_{n=1}^N \alpha_n$$。

2. 在注意力分布式上做随机采样。

硬性注意力的一个缺点是基于最大采样或随机采样的方式来选择信息，使得最终的损失函数与注意力分布之间的函数关系不可导，无法使用反向传播算法进行训练。因此，硬性注意力通常需要使用强化学习来进行训练。为了使用反向传播算法，一般使用软性注意力来代替硬性注意力。



### 键值对注意力

更一般地，我们可以用**键值对(key-value pair)**格式来表示输入信息，其中“键”用来计算注意力分布$$α_n$$，“值”用来计算聚合信息。

用$$(K, V) = [(\pmb k_1 ,\pmb v_1 ), ⋯ , (\pmb k_N ,\pmb v_N)]$$表示$$N$$组输入信息，给定任务相关的查询向量$$q$$时，注意力函数为
$$
{\rm att}((K,V),\pmb q)=\sum_{n=1}^N\alpha_n\pmb v_n\\
=\sum_{n=1}^N {\rm softmax}(s(\pmb k_n,\pmb q))\pmb v_n \\
=\sum_{n=1}^N \frac{\exp(s(\pmb k_n,\pmb q))}{\sum_{j=1}^N \exp(s(\pmb k_j,\pmb q))}\pmb v_n
$$
   其中$$s(\pmb k_n,\pmb q)$$为打分函数。

   

### 多头注意力

**多头注意力(multi-head attention)**是利用多个查询$$Q = [\pmb q_1 , ⋯ ,\pmb q_M ]$$， 来并行地从输入信息中选取多组信息。每个注意力关注输入信息的不同部分。
$$
{\rm att}((K,V),Q)={\rm att}((K,V),\pmb q_1)\oplus\cdots\oplus {\rm att}((K,V),\pmb q_M)
$$
其中$$⊕$$表示向量拼接。



### *指针网络

注意力机制可以分为两步：一是计算注意力分布$$α$$，二是根据$$α$$来计算输入信息的加权平均。我们可以只利用注意力机制中的第一步，将注意力分布作为一个软性的**指针(pointer)**来指出相关信息的位置。

**指针网络(pointer network)** [Vinyals et al., 2015] 是一种序列到序列模型，输入是长度为$$N$$的向量序列$$X =\pmb x_1 , ⋯ ,\pmb x_N$$，输出是长度为$$M$$的下标序列$$\pmb c_{1∶M} = c_1 , c_2 , ⋯ , c_M , c_m ∈ [1, N], ∀m$$。

和一般的序列到序列任务不同, 这里的输出序列是输入序列的下标（索引）。



# 自注意力模型

当使用神经网络来处理一个变长的向量序列时, 我们通常可以使用卷积网络或循环网络进行编码来得到一个相同长度的输出向量序列，如下图所示。

![](https://i.loli.net/2020/11/09/wI8tRnVz6GoKuT7.png)

基于卷积或循环网络的序列编码都是一种局部的编码方式，只建模了输入信息的局部依赖关系。虽然循环网络理论上可以建立长距离依赖关系，但是由于信息传递的容量以及梯度消失问题，实际上也只能建立短距离依赖关系。

如果要建立输入序列之间的长距离依赖关系，可以使用以下两种方法：一种方法是增加网络的层数，通过一个深层网络来获取远距离的信息交互；另一种方法是使用全连接网络，全连接网络是一种非常直接的建模远距离依赖的模型，但是无法处理变长的输入序列。不同的输入长度，其连接权重的大小也是不同的，这时我们就可以利用注意力机制来 “动态” 地生成不同连接的权重，这就是**自注意力模型(self-attention model)**。?

为了提高模型能力，自注意力模型经常采用**查询-键-值(Query-Key-Value, QKV)**模式，其计算过程如下图所示，其中红色字母表示矩阵的维度。

![](https://i.loli.net/2020/11/09/godBrpMX85tTAu9.png)

假设输入序列为$$X = [\pmb x_1 , ⋯ ,\pmb x_N] ∈\mathbb{R}^{D_x×N}$$，输出序列为$$H = [\pmb h_1 , ⋯ ,\pmb h_N ] ∈\mathbb{R}^{D_v ×N}$$，自注意力模型的具体计算过程如下：

1. 对于每个输入$$\pmb x_i$$，我们首先将其线性映射到三个不同的空间，得到查询向量$$\pmb q_i\in\mathbb{R}^{D_k}$$，键向量$$\pmb k_i\in\mathbb{R}^{D_k}$$和值向量$$\pmb v_i\in\mathbb{R}^{D_v}$$，映射过程即为
   $$
   Q=W_qX\in \mathbb{R}^{D_k\times N}\\
   K=W_kX\in \mathbb{R}^{D_k\times N}\\
   V=W_vX\in \mathbb{R}^{D_v\times N}\\
   $$
   其中$$W_q,W_k,W_v$$为参数矩阵，$$Q=[\pmb q_1 , ⋯ ,\pmb q_N ],K=[\pmb k_1 , ⋯ ,\pmb k_N ],V=[\pmb v_1 , ⋯ ,\pmb v_N ]$$分别是由<u>查询向量</u>、<u>键向量</u>和<u>值向量</u>构成的矩阵。

2. 对于每一个查询向量$$\pmb q_n\in Q$$，利用键值对注意力机制公式，可以得到输出向量$$\pmb h_n$$，
   $$
   \pmb h_n={\rm att}((K,V),\pmb q_n)\\
   =\sum_{i=1}^N \alpha_{ni}\pmb v_i \\
   =\sum_{i=1}^N {\rm softmax}(s(\pmb k_i,\pmb q_n))\pmb v_i\\
   =\sum_{i=1}^N \frac{\exp(s(\pmb k_i,\pmb q_n))}{\sum_{j=1}^N \exp(s(\pmb k_j,\pmb q_n))}\pmb v_i
   $$
   其中$$n,i\in[1,N]$$为输出和输入向量序列的位置，

如果使用缩放点积作为注意力打分函数，输出向量序列可以简写为
$$
H=V {\rm softmax}(\frac{K^{\rm T}Q}{\sqrt{D_k}})
$$
其中$${\rm softmax}()$$函数按列进行归一化。

下图给出全连接模型和自注意力模型的对比，其中实线表示可学习的权重，虚线表示动态生成的权重。由于自注意力模型的权重是动态生成的，因此可以处理变长的信息序列。

![](https://i.loli.net/2020/11/09/ep5A1CQMu9GHvIV.png)

自注意力模型可以作为神经网络中的一层来使用，既可以用来替换卷积层和循环层 [Vaswani et al., 2017]，也可以和它们一起交替使用(比如$$X$$可以是卷积层或循环层的输出)。自注意力模型计算的权重$$α_{ij}$$只依赖于$$\pmb q_i$$和$$\pmb k_j$$的相关性，而忽略了输入信息的位置信息，因此在单独使用时，自注意力模型一般需要加入位置编码信息来进行修正 [Vaswani et al., 2017]。自注意力模型可以扩展为多头自注意力(multi-head self-attention)模型，在多个不同的投影空间中捕捉不同的交互信息。





# 序列到序列模型

在序列生成任务中，有一类任务是序列到序列生成任务，即输入一个序列，生成另一个序列，比如机器翻译、语音识别、文本摘要、对话系统、图像标题生成等。

**序列到序列(Sequence-to-Sequence , Seq2Seq)**是一种条件的序列生成问题，给定一个序列$$\pmb x_{1∶S}$$，生成另一个序列$$\pmb y_{1∶T}$$。输入序列的长度$$S$$和输出序列的长度$$T$$可以不同，比如在机器翻译中，输入为源语言，输出为目标语言。下图给出了基于循环神经网络的序列到序列机器翻译示例，其中$$⟨EOS⟩$$表示输入序列的结束，虚线表示用上一步的输出作为下一步的输入。

![](https://i.loli.net/2020/11/10/jHZJGxo6Wmc7aSn.png)

序列到序列模型的目标是估计条件概率
$$
p_\theta(\pmb y_{1∶T}|\pmb x_{1∶S})=\prod_{t=1}^T p_{\theta}(\pmb y_t|\pmb y_{1∶(t-1)},\pmb x_{1∶S})
$$
其中$$\pmb y_t ∈\mathcal{V}$$为词表$$\mathcal{V}$$中的某个词。

给定一组训练数据$$\{(\pmb x_{S_n}, \pmb y_{T_n})\}^N_{n=1}$$，我们可以使用最大似然估计来训练模型参数
$$
\hat{\theta}=\arg\max_\theta\sum_{n=1}^N\log p_\theta(\pmb y_{1:T_n}|\pmb x_{1:S_n})
$$
一旦训练完成，模型就可以根据一个输入序列$$x$$来生成最可能的目标序列
$$
\hat{\pmb y} =\arg\max_y p_{\hat{\theta}}(\pmb y|\pmb x)
$$
具体的生成过程可以通过贪婪方法或束搜索来完成。

和一般的序列生成模型类似，条件概率$$p_θ (y_t|\pmb y_{1∶(t−1)},\pmb x_{1∶S})$$可以使用各种不同的神经网络来实现。这里我们介绍三种主要的序列到序列模型：基于循环神经网络的序列到序列模型、基于注意力的序列到序列模型、基于自注意力的序列到序列模型。



## 基于循环神经网络的序列到序列模型

实现序列到序列的最直接方法是使用两个循环神经网络来分别进行编码和解码，也称为**编码器 - 解码器(encoder-decoder)**模型。

**编码器**

首先使用一个循环神经网络$$f_{\rm enc}$$来编码输入序列$$\pmb x_{1∶S}$$得到一个固定维数的向量$$\pmb u$$，$$\pmb u$$一般为<u>编码循环神经网络最后时刻的隐状态</u>。
$$
\overline{\pmb h}_s=f_{\rm enc}(\overline{\pmb h}_s-1,\pmb x_{s-1},\theta_{\rm enc}),\ s=1,\cdots,S\\
\pmb c=\overline{\pmb h}_S
$$
其中$$f_{\rm enc}(⋅)$$为<u>编码循环神经网络</u>，可以是 LSTM 或 GRU ，其参数为$$\theta_{\rm enc}$$，$$\pmb x_{s-1}$$为词$$x$$的词向量；$$\pmb c$$称为上下文向量(context vector)。

**解码器**

在生成目标序列时，使用另外一个循环神经网络$$f_{\rm dec}$$来进行解码。在解码过程的第$$t$$步时，已生成前缀序列为$$\pmb y_{1:T_n}$$ . 令$$\overline{\pmb h}_t$$表示在网络$$f_{\rm dec}$$的隐状态，$$\pmb o_t ∈ (0, 1)^{|\mathcal{V}|}$$为词表中所有词的后验概率，则
$$
\pmb h_0=\pmb c=\overline{\pmb h}_S\\
\pmb h_t=f_{\rm dec}(\pmb h_{t-1},\pmb y_{t-1},\theta_{\rm dec})\\
\pmb o_t=g(\overline{\pmb h}_t,\theta_o),\ t=1,\cdots,T
$$
其中$$f_{\rm dec}(\cdot)$$为<u>解码循环神经网络</u>，$$g(⋅)$$为最后一层为 Softmax 函数的前馈神经网络，$$\theta_{\rm dec}$$和$$θ_o$$为网络参数，$$\pmb y_{t-1}$$为词$$y$$的词向量，$$\pmb y_0$$为一个特殊符号，比如$$⟨EOS⟩$$。

基于循环神经网络的序列到序列模型的缺点是：(1) 编码向量$$\pmb u$$的容量问题，输入序列的信息很难全部保存在一个固定维度的向量中； (2) 当序列很长时， 由于循环神经网络的长程依赖问题，容易丢失输入序列的信息。



## 基于注意力的序列到序列模型

为了获取更丰富的输入序列信息，我们可以在每一步中通过注意力机制来从输入序列中选取有用的信息。

在解码过程的第$$t$$步中，先用上一步的隐状态$$\pmb h_{t-1}$$作为查询向量，利用注意力机制从所有输入序列的隐状态$$H_{\rm enc}=[\overline{\pmb h}_1,\cdots,\overline{\pmb h}_S]$$中选择相关信息
$$
\pmb c_t={\rm att}(H_{\rm enc},\pmb h_{t-1})\\
=\sum_{i=1}^S \alpha_{i}\overline{\pmb h}_i \\
=\sum_{i=1}^S {\rm softmax}(s(\overline{\pmb h}_i,\pmb h_{t-1}))\overline{\pmb h}_i\\
$$
其中$$\pmb c_t$$称为上下文向量，$$s(\cdot)$$为注意力打分函数。

然后将从输入序列中选择的信息$$\pmb c_t$$也作为解码器$$f_{\rm dec}(\cdot)$$在第$$t$$步时的输入，得到第$$t$$步的隐状态
$$
\pmb h_t=f_{\rm dec}(\pmb h_{t-1},[\pmb y_{t-1}, \pmb c_t],\theta_{\rm dec})\\
$$
最后将$$\overline{\pmb h}_t$$输入到分类器$$g(\cdot)$$中来预测词表中每个词出现的概率。



### Bahdanau Attention

> 参考论文[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

Bahdanau attention是由Bahdanau等人在2015年提出来的一种注意力机制（也是attention首次应用于NLP），其结构仍然采用encoder-decoder形式，如下图所示。

![](https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111503843-584496480.png)

论文中编码器采用的是双向循环神经网络结构；解码器的attention机制设计如下：

![](https://i.loli.net/2020/11/11/CWIYdh8l5ZH3Awi.png)

首先定义条件概率
$$
p(\pmb y_i|\pmb y_{1∶(i-1)},\pmb x_{1∶S})=g(\pmb y_{i-1},\pmb s_i,\pmb c_i)
$$
其中$$\pmb s_i$$是解码循环神经网络的隐状态，由下式计算
$$
\pmb s_i=f(\pmb s_{i-1},\pmb y_{i-1},\pmb c_i)
$$
上下文向量$$c_i$$由编码循环神经网络的隐状态加权求和得到
$$
\pmb c_i=\sum_{j=1}^T\alpha_{ij}\pmb h_j\\
=\sum_{j=1}^T{\rm softmax}({\rm score}(\pmb s_{i-1},\pmb h_j))\pmb h_j
$$
其中$$h_j$$由双向循环神经网络分别计算的隐状态拼接得到，即$$\pmb h_j=\vec{\pmb h_j}\oplus\overleftarrow{\pmb h_j}$$。

实验结果显示，相比传统的encoder-decoder模型，使用注意力机制的模型的表现更好，并且受输入句子长度增加的影响更小甚至几乎没有影响，表明在长句的处理上更有优势。

![](https://i.loli.net/2020/12/17/wMyD5C3ZobQU6cN.png)

此外，它还可以可视化注意力分布，使模型的翻译过程更具有解释力。

![](https://i.loli.net/2020/12/17/KZqekzTJwfhI2B3.png)



### Luong Attention

> 参考论文[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025v5.pdf)

Luong attention也是在2015年由Luong提出来的一种注意力机制。Luong在论文中提出了两种类型的注意力机制：一种是全局注意力模型，即每次计算上下文向量时都考虑输入序列的所有隐状态；另一种是局部注意力模型，即每次计算上下文向量时只考虑输入序列隐状态中的一个子集。

Luong attention的模型结构也是采用encoder-decoder的形式，encoder和decoder均采用多层LSTM，如下图所示。对于全局注意力模型和局部注意力模型，在计算上下文向量时，均使用encoder和decoder最顶层的LSTM的隐状态。

![](https://i.loli.net/2020/11/11/WBnJPoQX8Tmdy9V.png)



**全局注意力模型**

全局注意力模型在计算decoder的每个时间步的上下文向量$$\pmb c_t$$时，均考虑encoder的所有隐状态，记每个时间步对应的权重向量为$$\pmb a_t$$，其计算公式如下：
$$
\pmb a_t(s)={\rm align}(\pmb h_t,\overline{\pmb h}_s)\\
={\rm softmax}({\rm score}(\pmb h_t,\overline{\pmb h}_s))
$$
其中，$$\pmb h_t$$表示当前decoder第$$t$$个时间步的隐状态，$$\overline{\pmb h}_s$$表示encoder第$$s$$个时间步的隐状态。Luong attention在计算权重时提供了三种计算方式，并且发现<u>对于全局注意力模型，采用dot的权重计算方式效果要更好</u>：
$$
{\rm score}(\pmb h_t,\overline{\pmb h}_s)=

\left\{ 
    \begin{array}{l}
        \pmb h_t^{\rm T} \overline{\pmb h}_s,\quad\quad {\rm dot} \\ 
        \pmb h_t^{\rm T}W_a \overline{\pmb h}_s,\ \ {\rm general} \\ 
        \pmb v_a^{\rm T}\tanh(W_a[\pmb h_t;\overline{\pmb h}_s]),\ \ {\rm concat}\\
    \end{array}
\right.
$$
其中，concat模式跟Bahdanau attention的计算方式一致，而dot和general则直接采用矩阵乘积的形式。在计算完权重向量$$\pmb a_t$$后，将其对encoder的隐状态进行加权平均得到此刻的上下文向量$$\pmb c_t$$，
$$
\pmb c_t=\pmb a_t(s)^{\rm T}\overline{\pmb h}_s
$$
然后Luong attention将其与decoder此刻的隐状态$$\pmb h_t$$进行拼接，并通过一个带有tanh的全连接层得到$$\tilde{\pmb h}_t$$：
$$
\tilde{\pmb h}_t=\tanh(W_c[\pmb c_t;\pmb h_t])
$$
最后，将$$\tilde{\pmb h}_t$$传入带有softmax的输出层即可得到此刻目标词汇的概率分布：
$$
p(\pmb y_t|\pmb y_{1∶(t-1)},\pmb x)={\rm softmax}(W_s\tilde{\pmb h}_t)
$$



**局部注意力模型**

然而，全局注意力模型由于在每次decoder时，均考虑encoder所有的隐状态，因此其计算成本是非常昂贵的，特别是对于一些长句子或长篇文档，其计算就变得不切实际。因此作者又提出了另一种注意力模式，即局部注意力模型，即每次decoder时不再考虑encoder的全部隐状态了，只考虑局部的隐状态。

在局部注意力模型中，在decoder的每个时间步$$t$$，需要先确定输入序列中与该时刻对齐的一个位置$$p_t$$，然后以该位置为中心设定一个窗口大小，即$$[p_t-D,p_t+D]$$，其中$$D$$是表示窗口大小的整数，具体的取值需要凭经验设定，作者在论文中设定的是10。接着在计算权重向量时，只考虑encoder中在该窗口内的隐状态，当窗口的范围超过输入序列的范围时，则对超出的部分直接舍弃。局部注意力模型的计算逻辑如下图所示。

![](https://i.loli.net/2020/11/11/CUfWaAhH13qvc6m.png)

 在确定位置$$p_t$$时，作者也提出了两种对齐方式，一种是单调对齐，一种是预测对齐，分别定义如下：

+ 单调对齐(local-m)：即直接设定$$p_t=t$$，该对齐方式假设输入序列与输出序列的按时间顺序对齐的，接着计算$$\pmb a_t$$的方式与全局注意力模型相同。

+ 预测对齐(local-p)：预测对齐在每个时间步时会对$$p_t$$进行预测，其计算公式如下：
  $$
  p_t=S\cdot {\rm sigmoid}(\pmb v_p^{\rm T}\tanh(W_p \pmb h_t))
  $$
  其中，$$W_p,\pmb v_p$$为参数，$$S$$为输入序列的长度，这样一来，$$p_t\in [0,S]$$。另外在计算$$\pmb a_t$$时，作者还采用了高斯分布进行修正，其计算公式如下：
  $$
  \pmb a_t(s)={\rm align}(\pmb h_t,\overline{\pmb h}_s)\exp(-\frac{(s-p_t)^2}{2\sigma^2})
  $$
  其中，$${\rm align}(\pmb h_t,\overline{\pmb h}_s)$$与全局注意力模型的计算公式相同，$$s$$表示输入序列的位置，$$\sigma=D/2$$。

计算完权重向量后，后面$$\pmb c_t,\ \tilde{\pmb h}_t$$以及概率分布的计算都与全局注意力模型的计算方式相同，这里不再赘述。作者在实验中发现局部注意力模型采用local-p的对齐方式往往效果更好，因为在真实场景中输入序列和输出序列往往不会严格单调对齐，比如在翻译任务中，往往两种语言在表述同样一种意思时，其语序是不一致的。另外，计算权重向量的方式采用general的方式效果比较好。



**input-feeding**

在包含注意力机制的模型中，前面的词汇的对齐过程往往对后续词汇的对齐和预测是有帮助的。为此作者提出了input-feeding方法，即把上一个时刻的$$\tilde{\pmb h}_t$$与下一个时刻的输入进行拼接，共同作为下一个时刻的输入，如下图所示。实验结果显示，此方法可以显著提高decoder的效果。

![](https://i.loli.net/2020/11/11/JZr9MzaD5lcVmhS.png)



### 比较

1. Luong attention使用encoder和decoder的stacked LSTM最上层的隐状态，而Bahdanau attention使用encoder的BiRNN的隐状态拼接和decoder的non-stacking RNN的隐状态
2. Luong attention的计算路径是$$\pmb h_t\to \pmb a_t\to \pmb c_t\to \tilde{\pmb h}_t$$，而Bahdanau attention的计算路径是$$\pmb h_{t-1}\to \pmb a_t\to \pmb c_t\to \pmb h_t$$，并且之后进行预测的处理也有所不同

<img src="https://i.stack.imgur.com/yqJpG.png" style="zoom:150%;" />







## 基于自注意力的序列到序列模型

> 参考论文[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

除长程依赖问题外，基于循环神经网络的序列到序列模型的另一个缺点是无法并行计算。为了提高并行计算效率以及捕捉长距离的依赖关系，我们可以使用**自注意力模型(self-attention model)**来建立一个全连接的网络结构。本节介绍一个目前非常成功的基于自注意力的序列到序列模型：Transformer [Vaswani et al., 2017]。



### 自注意力

对于一个向量序列$$H=[\pmb h_1,\cdots,\pmb h_{T}]\in \mathbb{R}^{D_h\times T}$$，首先用自注意力模型对其进行编码，即
$$
{\rm selfatt}(Q,K,V)=V{\rm softmax}(\frac{K^{\rm T}Q}{\sqrt{D_k}})\in\mathbb{R}^{D_v\times T}\\
Q=W_qH,K=W_kH,V=W_vH
$$
其中$$D_k$$是矩阵$$Q,K$$的列向量的维度，$$W_q\in\mathbb{R}^{D_k\times D_h},W_k\in\mathbb{R}^{D_k\times D_h},W_v\in\mathbb{R}^{D_v\times D_h}$$是三个投影矩阵。



### 多头自注意力

自注意力模型可以看作在一个线性投影空间中建立$$H$$中不同向量之间的关系。为了提取更多的关系信息，我们可以使用多头自注意力(multi-head self-attention)，在多个不同的投影空间中捕捉不同的关系信息。假设在$$M$$个投影空间中分别应用自注意力模型，有
$$
{\rm MultiHead}(H)=W_o[{\rm head}_1;{\rm head}_M]\in \mathbb{R}^{D_h\times T}\\
{\rm head}_m={\rm selfatt}(Q_m,K_m,V_m)\in\mathbb{R}^{D_v\times T}\\
Q_m=W_{q,m}H,K_m=W_{k,m}H,V_m=W_{v,m}H,\ m=1,2,\cdots,M
$$
其中$$W_o\in\mathbb{R}^{D_h\times MD_v}$$是输出投影矩阵，$$W_{q,m}\in\mathbb{R}^{D_k\times D_h},W_{k,m}\in\mathbb{R}^{D_k\times D_h},W_{v,m}\in\mathbb{R}^{D_v\times D_h}$$是投影矩阵。



### 基于自注意力模型的序列编码

对于序列$$\pmb x_{1∶T}$$，我们可以构建一个含有多层多头自注意力模块的模型来对其进行编码。由于自注意力模型没有循环(recurrence)或者卷积，为了利用序列的顺序信息，我们需要在序列中注入位置信息，方法是在第1层的输入序列加上**位置编码(positional encoding)**。

对于一个输入序列$$\pmb x_{1∶T}\in\mathbb{R}^{D\times T}$$，令
$$
H^{(0)}=[\pmb x_1+\pmb p_1,\cdots,\pmb x_T+\pmb p_T]
$$
其中$$\pmb p_t\in\mathbb{R}^D$$为位置$$t$$的向量表示，即位置编码。$$\pmb p_t$$可以作为学习的参数，也可以预定义值，这里使用：
$$
\pmb p_{t,2i}=\sin(t/10000^{2i/D})\\
\pmb p_{t,2i+1}=\cos(t/10000^{2i/D})
$$
其中$$\pmb p_{t,2i}$$表示第$$t$$个位置的编码向量的第$$2i$$维，D 是编码向量的维度。

给定第$$l − 1$$层的隐状态$$H^{(l−1)}$$，第$$l$$层的隐状态$$H^{(l)}$$可以通过一个多头自注意力模块和一个非线性的前馈网络得到。每次计算都需要残差连接以及层归一化操作，具体计算为
$$
Z^{(l)}={\rm norm}(H^{(l-1)}+{\rm MultiHead}(H^{(l-1)}))\\
H^{(l)}={\rm norm}(Z^{(l)}+{\rm FFN}(Z^{(l)}))
$$
其中$${\rm norm}(\cdot)$$表示层归一化，$${\rm FFN}(\cdot)$$表示**逐位置的前馈神经网络(position-wise feed-forward network)**，是一个简单的两层网络。对于输入序列中每个位置上向量$$\pmb z\in Z^{(l)}$$，
$$
{\rm FFN}(\pmb z)=W_2+{\rm ReLu}(W_1\pmb z+\pmb b_1)+\pmb b_2
$$
其中$$W_1,W_2,\pmb b_1,\pmb b_2$$为网络参数。

基于自注意力模型的序列编码可以看作一个全连接的前馈神经网络，第$$l$$层的每个位置都接受第$$l − 1$$层的所有位置的输出。不同的是，其连接权重是通过注意力机制动态计算得到。



### *Transformer模型

Transformer 模型 [Vaswani et al., 2017] 是一个基于多头自注意力的序列到序列模型……



