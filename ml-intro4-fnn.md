# 神经元

**人工神经元(artificial neuron)**，简称**神经元(neuron)**，是构成神经网络的基本单元，它模拟生物神经元的结构和特性，接收一组输入信号并产生输出。

假设一个神经元接收$$D$$个输入$$x_1,x_2,\cdots,x_D$$，令向量$$\pmb x=[x_1,x_2,\cdots,x_D]$$来表示这组输入，并用**净输入(net input)**$$z\in \mathbb{R}$$表示一个神经元获得的输入信号$$\pmb x$$的加权和，
$$
z=\sum_{d=1}^Dw_dx_d+b\\
=\pmb w^{\rm T}\pmb x+b
$$
其中$$\pmb w=[w_1,w_2,\cdots,w_D]\in \mathbb{R}^D$$是$$D$$维的权重向量，$$b\in\mathbb{R}$$是偏置。

净输入$$z$$经过一个非线性函数$$f(\cdot)$$后，得到神经元的**活性值(activation)** $$a$$，
$$
a=f(z)
$$
其中非线性函数$$f(\cdot)$$称为**激活函数(activation function)**。下图给出了典型的神经元结构示例。

![](https://i.loli.net/2020/09/15/YMHfE6OiSJTqrZy.png)

激活函数是神经元中非常重要的部分。为了增强网络的表示能力和学习能力，激活函数需要具备以下性质：

1. 连续并可导(允许少数点上不可导)的非线性函数。可导的激活函数可以直接利用数值优化的方法来学习网络参数
2. 激活函数及其导数要尽量简单，以提高网络计算效率
3. 激活函数的导数的值域要在一个合适的区间内，不能过大或过小，否则会影响训练的效率和稳定性。

下面介绍集中常用的激活函数。



## Sigmoid型函数

Sigmoid 型函数是一类S型曲线函数，为两端饱和函数。常用的 Sigmoid 型函数有 Logistic 函数和 Tanh 函数。

> 对于函数$$f(x)$$，若$$x\to -\infty$$时$$f'(x)\to 0$$，称其为左饱和；若$$x\to +\infty$$时$$f'(x)\to 0$$，则称其为右饱和；同时满足左饱和和右饱和的函数称为两端饱和。

> Sigmoid 型函数模拟了生物神经元的特点，即对于一些输入产生兴奋，而对于另一些输入产生抑制。



**Logistic 函数**定义为
$$
\sigma(x)=\frac{1}{1+\exp(-x)}
$$


Logistic 函数将一个实数域的输入映射到$$(0,1)$$区间。当输入值在0附近时，Logistic 函数近似为线性函数。与感知器使用的阶跃激活函数相比，Logistic 函数是连续可导的，数学性质更好。

由于 Logistic 函数的性质，使用 Logistic 激活函数的神经元具有以下性质：

+ 输出可以视作概率分布，使得神经网络可以更好地和统计学习模型结合
+ 可以看作一个**软性门(soft gate)**，用于控制其它神经元输出信息的数量



**Tanh 函数**定义为
$$
\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
$$
Tanh 函数可以看作 Logistic 函数的放大平移，值域为$$(-1,1)$$
$$
\tanh(x)=2\sigma(2x)-1
$$
下图给出了 Logistic 函数和 Tanh 函数的形状。

![](https://i.loli.net/2020/09/15/A9GWxr7Tov1lyqQ.png)



### Hard-Logistic函数和Hard-Tanh函数

Logistic函数和Tanh函数需要计算$$\exp()$$函数值，使用分段函数近似可以减少计算开销。

Logistic函数在0附近的一阶泰勒展开为
$$
g_l(x)\approx \sigma(0)+\sigma'(0)x=0.25x+0.5
$$
因此可以用分段函数近似
$$
{\rm hard-logistic}(x)=\max\{\min\{0.25x+0.5,1\},0\}
$$
同样的，Tanh函数在0附近的一阶泰勒展开为
$$
g_t(x)\approx \tanh(0)+\tanh'(0)x=x
$$
因此可以用分段函数近似
$$
{\rm hard-logistic}(x)=\max\{\min\{x,1\},-1\}
$$
下图给出了Hard-Logistic函数和Hard-Tanh函数的形状

![](https://i.loli.net/2020/09/15/JHh1LkZD67wxXrV.png)



## ReLU函数

**ReLU(Rectified Linear Unit，修正线性单元)**是目前深度神经网络中经常使用的激活函数。ReLU实际上是一个**斜坡(ramp)**函数，定义为
$$
{\rm ReLU}(x)=\max\{0,x\}
$$
**优点**

+ ReLU只需要进行一次比较，计算上更加高效。

  > ReLU函数也被认为具有生物学上的合理性，比如单侧抑制，宽兴奋边界。
  >
  > 在生物神经网络中，同时处于兴奋状态的神经元非常稀疏，人脑中在同一时刻大概只有 1%∼4% 的神经元处于活跃状态。 Sigmoid 型激活函数会导致一个非稀疏的神经网络，而 ReLU 却具有很好的稀疏性，大约 50% 的神经元会处于激活状态。

+ ReLU函数为左饱和函数，且在$$x>0$$时导数为1，在一定程度上缓解了神经网络的梯度消失问题，加速梯度下降的收敛速度。

**缺点**

+ ReLU 神经元在训练时比较容易“死亡”。在训练时，如果参数在一次不恰当的更新后，第一个隐藏层中的某个 ReLU 神经元在所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是 0，在以后的训练过程中永远不能被激活。这种现象称为死亡 ReLU 问题(Dying ReLU Problem)。



### 带泄露的ReLU

**带泄露的 ReLU (Leaky ReLU)**在输入$$x < 0$$时，保持一个很小的梯度$$γ$$，这样当神经元非激活时也能有一个非零的梯度可以更新参数，避免永远不能被激活 [Maas et al., 2013] 。带泄露的 ReLU 的定义如下：
$$
{\rm LeakyReLU}(x)=\max\{x,\gamma x\}
$$
其中$$\gamma$$是一个很小的正数，如0.01。



### ELU函数

**ELU(Exponential Linear Unit , 指数线性单元)** [Clevert et al., 2015] 是一个近似的零中心化的非线性函数，其定义为
$$
{\rm ELU}(x)=\max(0,x)+\min(0,\gamma(\exp(x)-1))
$$
其中其中$$\gamma$$是一个很小的正数，决定$$x\le 0$$时的饱和曲线。



### Softplus函数

Softplus 函数 [Dugas et al., 2001] 可以看作 Rectifier 函数的平滑版本，其定义为
$$
{\rm Softplus}(x)=\log(1+\exp(x))
$$
Softplus 函数其导数刚好是 Logistic 函数。Softplus 函数虽然也具有单侧抑制、宽兴奋边界的特性，但没有稀疏激活性。



下图给出了ReLU、Leaky ReLU、ELU、Softplus函数的示例。

![](https://i.loli.net/2020/09/15/GvKc9BmLChizP7T.png)





## Swish函数

Swish 函数 [Ramachandran et al., 2017] 是一种**自门控(Self-Gated)**激活函数，定义为
$$
{\rm swish}(x) = xσ(βx)
$$
其中 $$σ(⋅)$$ 为 Logistic 函数，$$β$$ 为可学习的参数或一个固定超参数。 $$σ(⋅) ∈ (0, 1)$$可以看作一种软性的门控机制：当$$σ(βx)$$接近于 1 时，门处于“开”状态，激活函数的输出近似于$$x$$本身；当$$σ(βx)$$接近于0时，门的状态为“关”，激活函数的输出近似于0。下图给出了 Swish 函数的示例。

![](https://i.loli.net/2020/09/15/sEro4ayYJ5m6fAQ.png)

当$$β = 0$$时，Swish 函数变成线性函数$$x/2$$。当$$β = 1$$时，Swish 函数在$$x > 0$$时近似线性，在$$x < 0$$时近似饱和，同时具有一定的非单调性。当$$β → +∞$$时，$$σ(βx)$$趋向于单位阶跃函数，Swish 函数近似为 ReLU 函数。因此 Swish 函数可以看作线性函数和 ReLU 函数之间的非线性插值函数，其程度由参数$$β$$控制。



## GELU函数

**GELU(Gaussian Error Linear Unit , 高斯误差线性单元)** [Hendrycks et al.,2016] 也是一种通过门控机制来调整其输出值的激活函数，和 Swish 函数比较类似
$$
{\rm GELU}(x)=xP(X\le x)
$$
其中$$X$$服从高斯分布$$\mathcal{N}(\mu,\sigma^2)$$，一般设$$\mu=0,\sigma=1$$。由于高斯分布的累积分布函数为S型函数，因此 GELU 函数可以用 Tanh 函数或 Logistic 函数来近似
$$
{\rm GELU}(x)\approx 0.5x(1+\tanh(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)))\\
{\rm GELU}(x)\approx x\sigma(1.702x)
$$



## Maxout单元

**Maxout 单元** [Goodfellow et al., 2013] 也是一种分段线性函数。Sigmoid 型函数、ReLU 等激活函数的输入是神经元的净输入$$z$$，是一个标量，而 Maxout 单元的输入是上一层神经元的全部原始输出，是一个向量$$\pmb x = [x_1, x_2, ⋯ , x_D ]$$。

每个 Maxout 单元有$$K$$个权重向量$$\pmb w_k\in \mathbb{R}^D$$和偏置$$b_k(1\le k\le K)$$。对于输入$$\pmb x$$，可以得到$$K$$个净输入$$z_k=\pmb w_k^{\rm T}\pmb x+b_k,1\le k\le K$$。Maxout 单元的非线性函数定义为
$$
{\rm maxout}(\pmb x)=\max_{k\in \{1,2,\cdots,k\}}(z_k)
$$
Maxout 激活函数可以看作任意凸函数的分段线性近似，并且在有限的点上是不可微的。





# 网络结构

一个生物神经细胞的功能比较简单，而人工神经元只是生物神经细胞的理想化和简单实现，功能更加简单。要想模拟人脑的能力，单一的神经元是远远不够的，需要通过很多神经元一起协作来完成复杂的功能。这样通过一定的连接方式或信息传递方式进行协作的神经元可以看作一个网络，就是神经网络。

> 虽然这里将神经网络结构大体上分为三种类型，但是大多数网络都是复合型结构，即一个神经网络中包括多种网络结构。

## 前馈网络

前馈网络中各个神经元按接收信息的先后分为不同的组，每一组可以看作一个神经层，每一层中的神经元接收前一层神经元的输出，并输出到下一层神经元。整个网络中的信息是朝一个方向传播，没有反向的信息传播，可以用一个有向无环路图表示。

前馈网络可以看作一个<u>函数</u>，通过简单非线性函数的多次复合，实现输入空间到输出空间的复杂映射。这种网络结构简单，易于实现。

前馈网络包括全连接前馈网络和卷积神经网络等。



## 记忆网络

记忆网络，也称为反馈网络，网络中的神经元不但可以接收其他神经元的信息，也可以接收自己的历史信息。和前馈网络相比，记忆网络中的神经元具有记忆功能，在不同的时刻具有不同的状态。记忆神经网络中的信息传播可以是单向或双向传递，因此可用一个有向循环图或无向图来表示。

记忆网络可以看作一个<u>程序</u>，具有更强的计算和记忆能力。

记忆网络包括循环神经网络、Hopfield 网络、玻尔兹曼机、受限玻尔兹曼机等。

为了增强记忆网络的记忆容量，可以引入外部记忆单元和读写机制，用来保存一些网络的中间状态，称为记忆增强神经网络(Memory Augmented NeuralNetwork , MANN)，比如神经图灵机 [Graves et al., 2014] 和记忆网络 [Sukhbaatar et al., 2015] 等。



## 图网络

前馈网络和记忆网络的输入都可以表示为向量或向量序列，但实际应用中很多数据是图结构的数据，比如知识图谱、社交网络、分子(molecular)网络等。前馈网络和记忆网络很难处理图结构的数据。

图网络是定义在图结构数据上的神经网络，图中每个节点都由一个或一组神经元构成，节点之间的连接可以是有向的，也可以是无向的，每个节点可以收到来自相邻节点或自身的信息。

图网络是前馈网络和记忆网络的泛化，包含很多不同的实现方式，比如图卷积网络(Graph Convolutional Network, GCN) [Kipf et al., 2016] 、图注意力网络(Graph Attention Network , GAT) [Veličković et al., 2017] 、消息传递神经网络(Message Passing Neural Network ,MPNN) [Gilmer et al., 2017] 等。



下图给出了前馈网络、记忆网络和图网络的网络结构示例，其中圆形节点表示一个神经元，方形节点表示一组神经元。

![](https://i.loli.net/2020/11/16/CsRP5ruqUcEh4lI.png)





# 前馈神经网络

**前馈神经网络(Feedforward Neural Network , FNN)**是最早发明的简单人工神经网络。前馈神经网络也经常称为多层感知器(Multi-Layer Perceptron, MLP)，但多层感知器的叫法并不是十分合理，因为前馈神经网络其实是由多层的 Logistic 回归模型（连续的非线性函数）组成，而不是由多层的感知器（不连续的非线性函数）组成 [Bishop, 2007]。

在前馈神经网络中，各神经元分别属于不同的层，每一层的神经元可以接收前一层神经元的信号，并产生信号输出到下一层。第0层称为输入层，最后一层称为输出层，其他中间层称为隐藏层。整个网络中无反馈，信号从输入层向输出层单向传播，可用一个有向无环图表示。前馈神经网络如下图所示。

![](https://i.loli.net/2020/09/15/Yi1LzNP5mfGgCaH.png)

下表给出了描述前馈神经网络的记号。

| 记号                                               | 含义                                |
| -------------------------------------------------- | ----------------------------------- |
| $$L$$                                              | 神经网络的层数                      |
| $$M_i$$                                            | 第$$l$$层神经元的个数               |
| $$f_l(\cdot)$$                                     | 第$$l$$层神经元的激活函数           |
| $$\pmb W^{(l)}\in \mathbb{R}^{M_l\times M_{l-1}}$$ | 第$$l-1$$层到第$$l$$层的权重矩阵    |
| $$\pmb b^{(l)}\in \mathbb{R}^{M_l}$$               | 第$$l-1$$层到第$$l$$层的偏置        |
| $$\pmb z^{(l)}\in \mathbb{R}^{M_l}$$               | 第$$l$$层神经元的净输入（净活性值） |
| $$\pmb a^{(l)}\in \mathbb{R}^{M_l}$$               | 第$$l$$层神经元的输出（活性值）     |

令$$\pmb a^{(0)}=\pmb x$$，前馈神经网络通过不断迭代以下公式进行信息传播：
$$
\pmb z^{(l)}=\pmb W^{(l)}\pmb a^{(l-1)}+\pmb b^{(l)}\\
\pmb a^{(l)}=f_l(\pmb z^{(l)})
$$
首先根据第$$l-1$$层神经元的**活性值(activation)** $$\pmb a^{(l−1)}$$计算出第$$l$$层神经元的**净活性值(net activation)** $$z^{(l)}$$，然后经过一个激活函数得到第$$l$$层神经元的活性值。因此，我们也可以把每个神经层看作一个仿射变换(affine transformation)和一个非线性变换。

前馈神经网络通过逐层的信息传递得到网络最后的输出$$\pmb a^{(L)}$$。整个网络可以看做一个复合函数$$\phi(\pmb x;\pmb W,\pmb b)$$：
$$
\pmb x= \pmb a^{(0)}\to \pmb z^{(1)}\to \pmb a^{(1)}\to \pmb z^{(2)}\to \cdots \to \pmb z^{(L)}\to \pmb a^{(L)}=\phi(\pmb x;\pmb W,\pmb b)
$$
其中$$\pmb W,\pmb b$$表示所有层的连接权重和偏置。



## 通用近似定理

前馈神经网络具有很强的拟合能力，常见的连续非线性函数都可以用前馈神经网络来近似。

**通用近似定理(Universal Approximation Theorem)** [Cybenko, 1989; Hornik et al., 1989] ：$$\mathcal{I}_D$$是一个$$D$$维的单位超立方体$$[0, 1]^D$$，$$C(\mathcal{I}_D)$$ 是定义在$$\mathcal{I}_D$$上的连续函数集合。对于任意给定的一个函数$$f ∈ C(\mathcal{I}_D)$$ , 存在整数$$M$$，实数$$v_m,b_m ∈ \mathbb{R}$$，实数向量$$\pmb w_m ∈ \mathbb{R}^D ,m = 1, ⋯ , M$$和非常数、有界、单调递增的连续函数$$\phi(⋅)$$，使得对于$$\forall \varepsilon>0$$，可以定义函数
$$
F(\pmb x) =\sum_{m=1}^{M} v_m \phi(\pmb w^{\rm T}_m\pmb x + b_m )
$$
作为函数$$f$$的近似实现，即
$$
|F(\pmb x)-f(\pmb x)|<\varepsilon, \forall \pmb x\in \mathcal{I}_D
$$

> 通用近似定理在实数空间$$\mathbb{R}^D$$的有界闭集上依然成立。

根据通用近似定理，对于具有线性输出层和至少一个使用 “挤压” 性质的激活函数的隐藏层组成的前馈神经网络，只要其隐藏层神经元的数量足够，它可以<u>以任意的精度来近似任何一个定义在实数空间$$\mathbb{R}^D$$中的有界闭集函数</u> [Funa-hashi et al., 1993; Hornik et al., 1989] 。所谓 “挤压” 性质的函数是指像 Sigmoid 函数的有界函数，但神经网络的通用近似性质也被证明对于其他类型的激活函数，比如 ReLU ，也都是适用的。



## 应用到机器学习

根据通用近似定理，神经网络在某种程度上可以作为一个 “万能” 函数来使用，可以用来进行复杂的特征转换或逼近一个复杂的条件分布。

多层前馈神经网络也可以看成是一种特征转换方法，将输入$$\pmb x\in\mathbb{R}^D$$映射到输出$$\phi(\pmb x)\in \mathbb{R}^{D'}$$，再将输出$$\phi(\pmb x)$$作为分类器的输入进行分类。

> 根据通用近似定理，只需要一层隐藏层就可以逼近任何函数，那么多层的神经网络的前几层就可以视作特征转换过程。

特别地，如果分类器$$g(\cdot)$$是 Logistic 回归分类器或 Softmax 回归分类器，那么$$g(⋅)$$也可以看成是网络的最后一层，即神经网络直接输出不同类别的条件概率。对于二分类问题$$y\in \{0,1\}$$，Logistic 回归分类器可以看成神经网络的最后一层，只有一个神经元，并且其激活函数为 Logistic 函数. 网络的输出可以直接作为类别$$y = 1$$的条件概率，即
$$
p(y = 1|\pmb x) = a (L)
$$
其中$$a^{(L)} ∈ \mathbb{R}$$为第$$L$$层神经元的活性值。

对于多分类问题$$y ∈ {1, ⋯ , C}$$，如果使用 Softmax 回归分类器，相当于网络最后一层设置$$C$$个神经元，其激活函数为 Softmax 函数。网络最后一层(第$$L$$层)的输出可以作为每个类的条件概率。



## 参数学习

如果采用交叉熵损失函数，对于样本$$(\pmb x,\pmb y)$$，其损失函数为
$$
\mathcal{L}(\pmb y,\hat{\pmb y})=-\pmb y^{\rm T}\log \hat{\pmb y}
$$
其中$$\pmb y\in \{0,1\}^C$$是标签$$y$$对应的 one-hot 向量表示。

给定训练集$$\mathcal{D}=\{(\pmb x^{(n)},\pmb y^{(n)})\}_{n=1}^N$$，将每个样本$$\pmb x^{(n)}$$输入给前馈神经网络，得到输出$$\hat{\pmb y}^{(n)}$$，其在数据集$$\mathcal{D}$$上的结构化风险函数为
$$
\mathcal{R}(\pmb W,\pmb b)=\frac{1}{N}\sum_{n=1}^N\mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})+\frac{1}{2}\lambda||\pmb W||^2_F
$$
其中$$\pmb W$$和$$\pmb b$$分别表示网络中的权重矩阵和偏置向量，$$||\pmb W||^2_F$$是正则化项，一般使用Frobenius范数
$$
||\pmb W||^2_F=\sum_{l=1}^L\sum_{i=1}^{M_l}\sum_{j=1}^{M_{l-1}}(w_{ij}^{(l)})^2
$$

> 参考https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3%E7%AF%84%E6%95%B8
>
> 矩阵的Frobenius范数，也称为矩阵元-2范数，定义为
> $$
> ||A||_F=\sqrt{\sum^m_{i=1}\sum^n_{j=1}|a_{ij}|^2}
> $$

有了训练集和学习准则，网络参数可以通过梯度下降法进行学习。在梯度下降法的每次迭代中，第$$l$$层的参数$$\pmb W^{(l)}$$和$$\pmb b^{(l)}$$参数的更新方式为
$$
\pmb W^{(l)} \leftarrow \pmb W^{(l)}-\alpha\frac{\partial\mathcal{R_D}(\pmb W,\pmb b)}{\partial \pmb W^{(l)}}\\
=\pmb W^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^N(\frac{\partial \mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})}{\partial \pmb W^{(l)}})+\lambda \pmb W^{(l)})\\
\pmb b^{(l)} \leftarrow \pmb b^{(l)}-\alpha\frac{\partial\mathcal{R_D}(\pmb W,\pmb b)}{\partial \pmb b^{(l)}}\\
=\pmb b^{(l)}-\alpha(\frac{1}{N}\sum_{n=1}^N\frac{\partial \mathcal{L}(\pmb y^{(n)},\hat{\pmb y}^{(n)})}{\partial \pmb b^{(l)}})\\
$$
其中$$\alpha$$为学习率。

梯度下降法需要计算损失函数对参数的偏导数，然而使用链式法则求偏导比较低效，在神经网络的训练中一般使用反向传播算法来高效计算梯度。





# 反向传播算法

假设采用随机梯度下降进行神经网络参数学习，给定<u>一个</u>样本$$(\pmb x,\pmb y)$$，将其输入到神经网络模型中，得到网络输出为$$\hat{\pmb y}$$。假设损失函数为$$\mathcal{L}(\pmb y,\hat{\pmb y})$$，要进行参数学习就需要计算损失函数关于每个参数的导数。

不失一般性，对第$$l$$层中的参数$$\pmb W^{(l)}$$和$$\pmb b^{(l)}$$计算偏导数。根据链式法则
$$
\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial w_{ij}^{(l)}}=\frac{\partial \pmb z^{(l)}}{\partial w_{ij}^{(l)}}\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}  \\
\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb b^{(l)}}=\frac{\partial \pmb z^{(l)}}{\partial \pmb b^{(l)}}\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}
$$
只需要计算偏导数$$\frac{\partial \pmb z^{(l)}}{\partial w_{ij}^{(l)}}$$，$$\frac{\partial \pmb z^{(l)}}{\partial \pmb b^{(l)}}$$，$$\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}$$：

1. 计算偏导数$$\frac{\partial \pmb z^{(l)}}{\partial w_{ij}^{(l)}}$$，根据$$\pmb z^{(l)}=\pmb W^{(l)}\pmb a^{(l-1)}+\pmb b^{(l)}\\$$，

   > 回想$$\pmb z^{(l)}\in \mathbb{R}^{M_l}$$，$$\pmb W^{(l)}\in \mathbb{R}^{M_l\times M_{l-1}}$$，$$\pmb a^{(l)}\in \mathbb{R}^{M_l}$$

   $$
   \frac{\partial \pmb z^{(l)}}{\partial w_{ij}^{(l)}}=[\frac{\partial z^{(l)}_1}{\partial w_{ij}^{(l)}},\cdots,\frac{\partial z^{(l)}_i}{\partial w_{ij}^{(l)}},\cdots,\frac{\partial z^{(l)}_{M_l}}{\partial w_{ij}^{(l)}}]\\
   =[0,\cdots,\frac{\partial (\pmb w_{i*}^{(l)}\pmb a^{(l-1)}+b_i^{(l)})}{\partial w_{ij}^{(l)}},\cdots,0]\\
   =[0,\cdots,a_j^{l-1},\cdots,0]\triangleq \mathbb{I}_i(a_j^{(l-1)})\in \mathbb{R}^{1\times M_l}
   $$

   其中$$\pmb w_{i*}^{(l)}$$表示$$\pmb W^{(l)}$$的第$$i$$行，$$\mathbb{I}_i(a_j^{(l-1)})$$表示第$$i$$个元素为$$a_j^{(l-1)}$$，其余为0的行向量。

2. 计算偏导数$$\frac{\partial \pmb z^{(l)}}{\partial \pmb b^{(l)}}$$，根据$$\pmb z^{(l)}=\pmb W^{(l)}\pmb a^{(l-1)}+\pmb b^{(l)}$$，

   $$
   \frac{\partial \pmb z^{(l)}}{\partial \pmb b^{(l)}}=\pmb I_{M_l}\in \mathbb{R}^{M_l\times M_l}
   $$
   为单位矩阵。

3. 计算偏导数$$\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}$$，这一项表示第$$l$$层神经元对最终损失的影响，因此一般称为第$$l$$层神经元的**误差项**，用$$\pmb \delta^{(l)}$$表示，

   $$
   \pmb \delta^{(l)}\triangleq \frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}\in \mathbb{R}^{M_l}
   $$
   误差项也间接反映了不同神经元对网络能力的贡献程度，从而比较好地解决**贡献度分配问题(Credit Assignment Problem, CAP)**。

   根据$$\pmb z^{(l+1)}=\pmb W^{(l+1)}\pmb a^{(l)}+\pmb b^{(l+1)}$$，
   $$
   \frac{\partial \pmb z^{(l+1)}}{\partial \pmb a^{(l)}}=(\pmb W^{(l+1)})^{\rm T} \in \mathbb{R}^{M_l\times M_{l+1}}
   $$
   根据$$\pmb a^{(l)}=f_l(\pmb z^{(l)})$$，其中$$f_l(\cdot)$$是按位计算的函数（即$$a_1^{(l)}=f_{l1}(z_1^{(l)})$$，$$a_2^{(l)}=f_{l2}(z_2^{(l)})$$，…），
   $$
   \frac{\partial \pmb a^{(l)}}{\partial \pmb z^{(l)}}={\rm diag}(f_l'(\pmb z^{(l)}))\in \mathbb{R}^{M_l\times M_l}
   $$
   根据链式法则
   $$
   \pmb \delta^{(l)}\triangleq \frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l)}}\\
   =\frac{\partial \pmb a^{(l)}}{\partial \pmb z^{(l)}}\frac{\partial \pmb z^{(l+1)}}{\partial \pmb a^{(l)}}\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb z^{(l+1)}}\\
   ={\rm diag}(f_l'(\pmb z^{(l)}))\cdot(\pmb W^{(l+1)})^{\rm T}\cdot\pmb \delta^{(l+1)}\\
   =f'_l(\pmb z^{(l)})\odot((\pmb W^{(l+1)})^{\rm T}\cdot \pmb \delta^{(l+1)})\in \mathbb{R}^{M_l}
   $$
   其中$$\odot$$是向量的点积运算符。从上式可以看出，第$$l$$层的误差项可以通过第$$l + 1$$层的误差项计算得到，这就是误差的**反向传播(Back Propagation, BP)**。反向传播算法的含义是：第$$l$$层的一个神经元的误差项（或敏感性）是所有与该神经元相连的第$$l + 1$$层的神经元的误差项的权重和，再乘上该神经元激活函数的梯度。

现在就可以计算偏导数$$\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial w_{ij}^{(l)}},\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb b^{(l)}}$$，
$$
\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial w_{ij}^{(l)}}=\mathbb{I}_i(a_j^{(l-1)})\pmb \delta^{(l)}=\delta_i^{(l)}a_j^{l-1}\Rightarrow \frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb W^{(l)}}=\pmb \delta^{(l)}(\pmb a^{(l-1)})^{\rm T}\in \mathbb{R}^{M_l\times M_{l-1}} \\
\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb b^{(l)}}=\pmb I_{M_l}\pmb \delta^{(l)}=\pmb \delta^{(l)}\in \mathbb{R}^{M_l}
$$
下图展示了使用随机梯度下降法和误差反向传播算法的前馈神经网络训练算法：

![](https://i.loli.net/2020/09/16/H9poPa4cAuFJRlV.png)





# 自动梯度计算

梯度下降法需要计算风险函数的梯度，即风险函数对各个参数的偏导数。我们可以手动使用链式法则计算并用代码实现，但此过程效率较低且容易出错。目前主流的深度学习框架都包含了自动梯度计算的功能，因此我们只需要考虑网络结构，大大提高了开发效率。

下面介绍自动梯度计算的三种方法。

## 数值微分

**数值微分(numerical differentiation)**使用数值方法来计算函数$$f(x)$$的导数。函数$$f(x)$$在点$$x$$的导数定义为
$$
f'(x)=\lim _{\Delta x\to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}
$$
要计算函数$$f(x)$$在点$$x$$的导数，可以对$$x$$加上一个很小的扰动$$Δx$$，通过上述定义来直接计算函数$$f(x)$$的梯度。数值微分方法非常容易实现，但找到一个合适的扰动$$Δx$$却十分困难：如果$$Δx$$过小，会引起数值计算问题，比如舍入误差；如果$$Δx$$过大，会增加截断误差，使得导数计算不准确。

> 在数值计算中，**舍入误差(round-off error)**指由于数字的舍入造成的近似值和精确值之间的差异，如用浮点数表示实数；**截断误差(truncation error)**指由于将超越计算的极限或无穷过程截断为有限过程造成的近似值和精确值之间的差异，如计算$$x-\frac{x^3}{3!}+\frac{x^5}{5!}$$以替代计算$$\sin x$$。

数值微分的另外一个问题是计算复杂度。为了计算$$\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb W^{(l)}}$$或$$\frac{\partial \mathcal{L}(\pmb y,\hat{\pmb y})}{\partial \pmb b^{(l)}}$$，对任意一个参数$$w_{ij}^{(l)}$$或$$b_{ij}^{(l)}$$施加一个扰动，为了得到损失函数的变化，都需要前馈计算每一层的$$\pmb z^{(l)}$$和$$\pmb a^{(l)}$$直到得到扰动后的$$\hat{\pmb y}$$，导致总的时间复杂度较大。由于以上原因，数值微分的实用性比较差。



## 符号微分

**符号微分(symbolic differentiation)**是一种基于符号计算的自动求导方法。符号计算也叫代数计算，是指用计算机来处理带有变量的数学表达式。这里的变量被看作符号(symbols)，一般不需要代入具体的值。符号计算的输入和输出都是数学表达式，一般包括对数学表达式的化简、因式分解、微分、积分、解代数方程、 求解常微分方程等运算。

符号计算一般来讲是对输入的表达式，通过迭代或递归使用一些事先定义的规则进行转换。当转换结果不能再继续使用变换规则时，便停止计算。

符号微分可以在编译时就计算梯度的数学表示，并进一步利用符号计算方法进行优化。此外，符号计算的一个优点是符号计算和平台无关，可以在 CPU 或GPU 上运行。符号微分也有一些不足之处： (1) 编译时间较长，特别是对于循环，需要很长时间进行编译； (2)为了进行符号微分，一般需要设计一种专门的语言来表示数学表达式，并且要对变量（符号）进行预先声明； (3) 很难对程序进行调试。



## 自动微分

**自动微分(Automatic Differentiation , AD)**是一种可以对一个（程序）函数进行计算导数的方法。

自动微分的基本原理是所有的数值计算可以分解为一些基本操作，包含$$+, −, ×, /$$和一些初等函数$$\exp, \log, \sin, \cos$$等，然后利用链式法则来自动计算一个复合函数的梯度。



这里以神经网络中一个常见的复合函数的例子来说明自动微分的过程，令
$$
f(x;w,b)=\frac{1}{\exp(-(wx+b))+1}
$$
其中$$x$$为输入标量，$$w$$和$$b$$分别为权重和偏置参数。求$$f(x;w,b)$$在$$x=1,w=0,b=0$$时的梯度。

首先，我们将复合函数$$f(x; w, b)$$分解为一系列的基本操作，并构成一个**计算图(computational graph)**。计算图是数学运算的图形化表示，其中的每个非叶子节点表示一个基本操作，每个叶子节点为一个输入变量或常量。下图给
出了当$$ x = 1, w = 0, b = 0 $$时复合函数$$f(x; w, b)$$的计算图，其中连边上的红色数字表示前向计算时复合函数中每个变量的实际取值。

![](https://i.loli.net/2020/11/16/soP9f3lMQq2haYt.png)

复合函数$$f(x;w,b)$$被分解为一系列的基本函数$$h_i$$，如下表所示，每个基本函数的导数都十分简单，可以通过规则来实现。
$$
\begin{align}
\hline
&函数 && 导数 & \\
\hline
&h_1=wx && \frac{\partial h_1}{\partial w}=x & \\
\hline
&h_2=h_1+b && \frac{\partial h_2}{\partial h_1}=1 & \frac{\partial h_2}{\partial b}=1\\
\hline
&h_3=-h_2 && \frac{\partial h_3}{\partial h_2}=-1 &\\
\hline
&h_4=\exp(h_3) && \frac{\partial h_4}{\partial h_3}=\exp(h_3) &\\
\hline
&h_5=h_4+1 && \frac{\partial h_5}{\partial h_4}=1 &\\
\hline
&h_6=1/h_5 && \frac{\partial h_6}{\partial h_5}=-\frac{1}{h_5^2} &\\
\hline
\end{align}
$$
因此$$f(x;w,b)$$对参数$$w$$和$$b$$的偏导数可以通过链式法则求得
$$
\frac{\partial f(x;w,b)}{\partial w}=\frac{\partial f(x;w,b)}{\partial h_6}\frac{\partial h_6}{\partial h_5}\frac{\partial h_5}{\partial h_4}\frac{\partial h_4}{\partial h_3}\frac{\partial h_3}{\partial h_2}\frac{\partial h_2}{\partial h_1}\frac{\partial h_1}{\partial w}\\
\frac{\partial f(x;w,b)}{\partial b}=\frac{\partial f(x;w,b)}{\partial h_6}\frac{\partial h_6}{\partial h_5}\frac{\partial h_5}{\partial h_4}\frac{\partial h_4}{\partial h_3}\frac{\partial h_3}{\partial h_2}\frac{\partial h_2}{\partial b}\\
$$
当$$x=1,w=0,b=0$$时，
$$
\frac{\partial f(x;w,b)}{\partial w}|_{x=1,w=0,b=0}=\frac{\partial f(x;w,b)}{\partial h_6}\frac{\partial h_6}{\partial h_5}\frac{\partial h_5}{\partial h_4}\frac{\partial h_4}{\partial h_3}\frac{\partial h_3}{\partial h_2}\frac{\partial h_2}{\partial h_1}\frac{\partial h_1}{\partial w}\\
=1\times -0.25\times1\times1\times-1\times1\times1=0.25
$$

如果函数和参数之间有多条路径，可以将这多条路径上的导数再进行相加，得到最终的梯度。

按照计算导数的顺序，自动微分可以分为两种模式：前向模式和反向模式。**前向模式**是按计算图中计算方向的相同方向来递归地计算梯度；**反向模式**是按计算图中计算方向的相反方向来递归地计算梯度。前向模式和反向模式可以看作应用链式法则的两种梯度累积方式，从反向模式的计算顺序可以看出，<u>反向模式和反向传播的计算梯度的方式相同</u>。对于一般的神经网络，损失函数只有一个，但是参数非常多，因此反向模式的计算量远小于前向模式。



@下面的例子比较了前向模式和反向模式。

计算图

<img src="http://fancyerii.github.io/img/autodiff/tree-def.png" alt="" style="zoom: 33%;" />

前向模式

<img src="http://fancyerii.github.io/img/autodiff/tree-forwradmode.png" alt="" style="zoom:33%;" />

反向模式

<img src="http://fancyerii.github.io/img/autodiff/tree-backprop.png" alt="" style="zoom:33%;" />



计算图按构建方式可以分为**静态计算图(static computational graph)**和**动态计算图(dynamic computational graph)**。 静态计算图是在编译时构建计算图，计算图构建好之后在程序运行时不能改变，而动态计算图是在程序运行时动态构建。两种构建方式各有优缺点：静态计算图<u>在构建时可以进行优化，并行能力强</u>，但灵活性比较差。动态计算图则不容易优化，当不同输入的网络结构不一致时，难以并行计算，但是<u>灵活性比较高</u>。

> 在目前深度学习框架里，Theano 和 Tensorflow 采用的是静态计算图，而 DyNet 、Chainer 和 PyTorch 采是动态计算图。Tensorflow 2.0 也支持了动态计算图。






# 优化问题

神经网络的参数学习比线性模型更加困难，主要原因包括非凸优化问题和梯度消失问题。

## 非凸优化问题

以一个最简单的1-1-1结构的两层神经网络为例，
$$
y=\sigma(w_2\sigma(w_1x))
$$
其中$$w_1,w_2$$为参数，$$\sigma(\cdot)$$为 Logistic 函数。

给定一个输入样本$$(x,y)=(1,1)$$，分别使用两种损失函数：平方误差损失$$\mathcal{L}(w_1,w_2)=(1-y)^2$$，交叉熵损失$$\mathcal{L}(w_1,w_2)=\log y$$。下图展示了损失函数与参数$$w_1,w_2$$的关系，可以看到两种损失函数都是关于$$w_1,w_2$$的非凸函数。

![](https://i.loli.net/2020/09/16/K3R2AodilVFp6L9.png)

> 降低损失函数，即$$y\to 1$$，即$$w_2\to +\infty$$且$$w_2\sigma(w_1x)\to +\infty$$



## 梯度消失问题

误差反向传播的迭代公式为
$$
\pmb \delta^{(l)}=f'_l(\pmb z^{(l)})\odot((\pmb W^{(l+1)})^{\rm T}\cdot \pmb \delta^{(l+1)})\in \mathbb{R}^{M_l}
$$
计算每一层误差时都需要计算该层的激活函数的导数。当激活函数是 Logistic 函数 $$σ(x)$$ 或 Tanh 函数时，其导数为
$$
\sigma'(x)=\sigma(x)(1-\sigma(x))\in [0,0.25]\\
\tanh'(x)=1-\tanh^2(x)\in [0,1]
$$
![](https://i.loli.net/2020/09/16/H4AJYEDtMBKCeSa.png)

因此误差每反向传播一层都会衰减一次，当网络层数很深时梯度会衰减到近乎为0，使得整个网络很难训练。这就是所谓的**梯度消失问题(vanishing gradient problem)**。

在深度神经网络中，减轻梯度消失问题的方法有很多种。一种简单有效的方式是使用导数比较大的激活函数，比如 ReLU 等。

> 梯度消失问题在过去的三十年里一直没有得到有效解决，是阻碍神经网络发展的重要原因之一。



