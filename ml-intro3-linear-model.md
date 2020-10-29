# 3.线性模型

\[toc\]

**线性模型\(linear model\)**是机器学习中应用最广泛的模型，指通过样本特征的线性组合来进行预测的模型。给定一个$$D$$维样本$$\pmb x=[x_1,\cdots,x_D]^{\rm T}$$，其线性组合函数为

$$
f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b
$$

其中$$\pmb w=[w_1,\cdots,w_D]^{\rm T}$$为$$D$$维的权重向量，$$b$$为偏置。线性回归即是典型的线性模型，直接使用**判别函数\(discriminant function\)**$$f(\pmb x;\pmb w)$$来预测输出目标$$y=f(\pmb x;\pmb w)$$。

## 线性回归

**线性回归\(linear regression\)**是机器学习和统计学中最基础和最广泛应用的模型，是一种对自变量和因变量之间关系进行建模的回归分析。自变量数量为1时称为**简单回归**，自变量数量大于1时称为**多元回归**。

从机器学习的角度来看，自变量就是样本的特征向量$$\pmb x ∈ \mathbb{R}^D$$\(每一维对应一个自变量\)，因变量是标签$$y$$，这里$$y ∈ \mathbb{R}$$是连续值\(实数或连续整数\)。假设空间是一组参数化的线性函数：

$$
f(\pmb x; \pmb w,b)=\pmb w^{\rm T}\pmb x+b
$$

其中权重向量$$\pmb w \in \mathbb{R}^D$$和偏置$$b\in \mathbb{R}$$都是可学习的参数，函数$$f(\pmb x; \pmb w，b) \in \mathbb{R}$$也称为**线性模型**。

为简单起见，将上式写为

$$
f(\pmb x;\hat{\pmb w})=\hat{\pmb w}^{\rm T}\hat{\pmb x}
$$

其中$$\hat{\pmb w}$$和$$\hat{\pmb x}$$分别称为增广权重向量和增广特征向量:

$$
\hat{\pmb x}=\pmb x \oplus 1=\begin{bmatrix}x_1\\ \vdots \\x_D \\1
\end{bmatrix}\\
\hat{\pmb w}=\pmb w \oplus b=\begin{bmatrix}w_1\\ \vdots \\w_D \\b
\end{bmatrix}\\
$$

其中$$\oplus$$定义为两个向量的拼接操作。

之后将采用简化的表示方法，即直接用$$\pmb w$$和$$\pmb x$$表示増广权重向量和増广特征向量。

### 参数学习

给定一组包含$$N$$个训练样本的训练集$$\mathcal{D} = \{(\pmb x^{(n)},y^{(n)})\}_{n=1}^N$$，我们希望能够学习一个最优的线性回归的模型参数$$\pmb w$$。

这里介绍四种不同的参数估计方法：经验风险最小化、结构风险最小化、最大似然估计、最大后验估计。

#### 经验风险最小化（最小二乘法）

由于线性回归的标签 $$y$$ 和模型输出都为连续的实数值，因此平方损失函数非常合适衡量真实标签和预测标签之间的差异。

根据经验风险最小化准则，训练集 $$\mathcal{D}$$ 上的经验风险定义为

$$
\mathcal{R}(\pmb w)=\sum_{n=1}^N \mathcal{L}(y^{(n)},f(\pmb x^{(n)};\pmb w))\\
=\frac{1}{2}\sum_{n=1}^N(y^{(n)}-\pmb w^{\rm T}\pmb x^{(n)})^2\\
=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2
$$

其中

$$
\pmb y = [y^{(1)},\cdots,y^{(N)}]^{\rm T}\\
X=\begin{bmatrix} x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(N)}\\
\vdots & \vdots & \ddots & \vdots\\
x_D^{(1)} & x_D^{(2)} & \cdots & x_D^{(N)}\\
1 & 1 & \cdots & 1
\end{bmatrix}
$$

风险函数$$\mathcal{R}(\pmb w)$$是关于$$\pmb w$$的凸函数，其对$$\pmb w$$的偏导数为

$$
\frac{\partial \mathcal{R}(\pmb w)}{\partial \pmb w}=\frac{1}{2}\frac{\partial ||\pmb y-X^{\rm T} \pmb w||^2}{\partial \pmb w}\\
=-X(\pmb y-X^{\rm T}\pmb w)
$$

令该偏导数为$$\pmb 0$$，得到最优参数

$$
\pmb w^* = (X X^{\rm T})^{-1} X \pmb y\\
$$

建立这种求解线性回归参数的方法也叫**最小二乘法\(Least Square Method , LSM\)**。

@对平面直角坐标系上的点：$$(1,1),(3,2),(5,5),(8,6),(9,7),(11,8)$$进行线性回归，其中$$x$$为自变量，$$y$$为因变量。

$$
X=\begin{bmatrix}1&3&5&8&9&11\\
1&1&1&1&1&1
\end{bmatrix},
\pmb y=(1,2,5,6,7,8)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(0.7162,0.4165)^{\rm T}\\
$$

回归方程为$$y=0.7162x+0.4165$$，如图所示。

![Screenshot from 2020-10-28 18-07-23.png](https://i.loli.net/2020/10/28/zbmDUGLBrSJ21cw.png)

最小二乘法要求$$XX^{\rm T}\in \mathbb{R}^{(D+1)\times (D+1)}$$必须存在逆矩阵。一种常见的$$XX^{\rm T}$$不可逆的情况是样本数量$$N$$小于特征数量$$(D+1)$$，这时$$XX^{\rm T}$$的秩为$$N$$。

当$$XX^{\rm T}$$不可逆时, 可以通过下面两种方法来估计参数：

1. 先使用主成分分析等方法来预处理数据，消除不同特征之间的相关性，然后再使用最小二乘法来估计参数；
2. 使用梯度下降法来估计参数：先初始化$$\pmb w =\pmb 0$$，然后通过下面公式进行迭代：

   $$
   \pmb w ←\pmb w +αX(\pmb y −X^{\rm T}\pmb w),
   $$

   这种方法也称为最小均方\(least mean squares, LMS\)算法。

@对平面直角坐标系上的点：$$(1,1),(1,2)$$进行线性回归，其中$$x$$为自变量，$$y$$为因变量。

$$
X=\begin{bmatrix}1&1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
XX^{\rm T}不可逆
$$

使用梯度下降法，设定初始值$$\pmb w_0=(0,0)^{\rm T}$$，$$\alpha=0.1$$，

$$
\pmb w_1=\pmb w_0+αX(\pmb y −X^{\rm T}\pmb w_0)=(0.3,0.3)^{\rm T}\\
\pmb w_2=\pmb w_1+αX(\pmb y −X^{\rm T}\pmb w_1)=(0.48,0.48)^{\rm T}\\
\cdots\\
\pmb w_\infty = (0.75,0.75)^{\rm T}
$$

![Screenshot from 2020-10-28 20-06-19.png](https://i.loli.net/2020/10/28/KBC6g5aUwTSHqZV.png)

#### 结构风险最小化（岭回归和Lasso回归）

即使$$XX^{\rm T}$$可逆，如果特征之间有较大的**多重共线性\(multicollinearity\)**，也会使得$$XX^{\rm T}$$的逆在数值上无法准确计算。数据集$$X$$上一些小的扰动就会导致$$(XX^{\rm T})^{-1}$$发生大的改变，进而使得最小二乘法的计算变得很不稳定。

> 共线性\(collinearity\)指一个特征可以通过其他特征的线性组合来较准确地预测

为了解决这个问题，**岭回归\(ridge regression\)**给$$XX^{\rm T}$$的对角线元素都加上一个常数$$λ$$使得$$(XX^{\rm T}+ λI)$$满秩。最优的参数$$\pmb w^∗$$为

$$
\pmb w^*=(XX^{\rm T} +\lambda I)^{-1}X\pmb y
$$

其中$$\lambda >0$$。

@求线性回归

$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(20,-19)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(10,-9)^{\rm T}\\
$$

求岭回归，设$$\lambda =0.01$$

$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(1.857,-0.401)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(2.529,-1.149)^{\rm T}\\
$$

![Screenshot from 2020-10-28 21-28-53.png](https://i.loli.net/2020/10/28/f7knMNPehYCw1Om.png)

求岭回归，设$$\lambda =0.1$$

$$
X=\begin{bmatrix}1&1.05\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(0.852,0.597)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&1.1\\1&1
\end{bmatrix},\pmb y=(1,2)^{\rm T} \\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(0.952,0.476)^{\rm T}\\
$$

![Screenshot from 2020-10-28 21-29-39.png](https://i.loli.net/2020/10/28/Q6DzdBVx2MtjlrK.png)

@求线性回归

$$
X=\begin{bmatrix}1&2&3&4.05&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(17.143,-7.029,7.714)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&2&3&4.1&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T})^{-1} X \pmb y=(8.571,1.543,-0.857)^{\rm T}\\
$$

求岭回归，设$$\lambda =0.1$$

$$
X=\begin{bmatrix}1&2&3&4.05&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(6.399,3.632,-2.536)^{\rm T}\\\quad\\
X=\begin{bmatrix}1&2&3&4.1&5\\2&3&4&5&6 \\
1&1&1&1&1
\end{bmatrix},
\pmb y=(11,20,32,42,51)^{\rm T}\\
\pmb w^*=(X X^{\rm T}+\lambda I)^{-1} X \pmb y=(6.372,3.628,-2.504)^{\rm T}\\
$$

岭回归的解$$\pmb w^*$$可以看作结构风险最小化准则下的最小二乘法估计，其结构风险在经验风险的基础上增加了$$l_2$$范数的正则化项：

$$
\mathcal{R}(\pmb w)=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2+\frac{1}{2}\lambda||\pmb w||^2
$$

类似地，**LASSO回归**的结构风险在经验风险的基础上增加了$$l_1$$范数的正则化项：

$$
\mathcal{R}(\pmb w)=\frac{1}{2}||\pmb y-X^{\rm T} \pmb w||^2+\lambda\sum_{i=1}^{D+1}\sqrt{w_i^2+\varepsilon}
$$

#### 最大似然估计

除了直接建立$$\pmb x$$和标签$$y$$之间的函数关系外，线性回归还可以从建立条件概率$$p(y|\pmb x)$$的角度来进行参数估计。

在给定$$\pmb x$$的条件下，假设标签$$y$$为一个随机变量，由函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x$$加上一个随机噪声$$ε$$决定，即

$$
y=\pmb w^{\rm T}\pmb x+\varepsilon
$$

其中$$\varepsilon$$服从均值为0，方差为$$\sigma^2$$的高斯分布。这样，$$y$$服从均值为$$\pmb w^{\rm T}\pmb x$$，方差为$$\sigma^2$$的高斯分布

$$
p(y|\pmb x;\pmb w,\sigma)=\mathcal{N}(y;\pmb w^{\rm T}\pmb x,\sigma^2) =\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y-\pmb w^{\rm T}\pmb x)^2}{2\sigma^2})
$$

参数$$\pmb w$$在训练集$$\mathcal{D}$$上的似然函数为

$$
p(\pmb y|X;\pmb w,\sigma)=\prod_{n=1}^N p(y^{(n)}|\pmb x^{(n)};\pmb w,\sigma)
$$

> **最大似然估计\(MLE\)**方法是找到一组参数$$\pmb w$$使得似然函数取最大值。

建立似然方程组

$$
\frac{\partial \ln p(\pmb y|X;\pmb w,\sigma)}{\partial \pmb w}=\pmb 0
$$

解得

$$
\pmb w^{MLE} = (X X^{\rm T})^{-1} X \pmb y\\
$$

可以看到，最大似然估计的解和最小二乘法的解相同。

#### 最大后验估计

假设参数$$\pmb w$$为一个随机向量，并服从一个先验分布$$p(\pmb w;\nu)$$。为简单起见，一般令$$p(\pmb w;\nu)$$为各向同性的高斯分布

$$
p(\pmb w;\nu)=\mathcal{N}(\pmb w;\pmb 0,\nu^2I)
$$

其中$$\nu^2$$为每一维上的方差。根据贝叶斯公式，参数$$\pmb w$$的后验分布为

$$
p(\pmb w|X,\pmb y;\nu,\sigma)=\frac{p(\pmb w,\pmb y|X;\nu,\sigma)}{\sum_{\pmb w}p(\pmb w,\pmb y|X;\nu,\sigma)}\propto p(\pmb w;\nu)p(\pmb y|X,\pmb w;\nu,\sigma)
$$

其中$$p(\pmb y|X,\pmb w;\nu,\sigma)$$为$$\pmb w$$的似然函数。

> 这种估计参数$$\pmb w$$的后验概率分布的方法称为**贝叶斯估计**，采用贝叶斯估计的线性回归也称为**贝叶斯线性回归**。
>
> 为得到点估计，可以采用最大后验估计方法。**最大后验估计\(MAP\)**方法是找到参数使后验分布取最大值
>
> $$
> \pmb w^{MAP}=\arg\max_{\pmb w}p(\pmb y|X,\pmb w;\sigma)p(\pmb w;\nu)
> $$

令似然函数为前面定义的高斯密度函数，那么

$$
p(\pmb w|X,\pmb y;\nu,\sigma)\propto -\frac{1}{2\sigma^2}\|\pmb y-X^{\rm T}\pmb w \|^2-\frac{1}{2\nu^2}\pmb w^{\rm T}\pmb w
$$

可以看到，最大后验估计等价于$$\ell_2$$正则化的的结构风险最小化，其中正则化系数$$\lambda=\sigma^2/\lambda^2$$。

最大似然估计和贝叶斯估计可以分别看作频率学派和贝叶斯学派对需要估计的参数$$\pmb w$$的不同解释。

## 线性判别函数和决策边界

在分类问题中，由于输出目标$$y$$是一些离散的标签，而$$f(\pmb x;\pmb w)$$的值域为实数，因此无法直接用$$f(\pmb x;\pmb w)$$来进行预测，需要引入一个非线性的**决策函数\(decision function\)**$$g(⋅)$$来预测输出目标

$$
\hat y=g(f(\pmb x;\pmb w))
$$

例如对于二分类问题，$$g(\cdot)$$可以是**符号函数\(sign function\)**，定义为

$$
g(f(\pmb x;\pmb w))={\rm sgn}(f(\pmb x;\pmb w))=\begin{cases}1,&f(\pmb x;\pmb w)>0\\
-1,&f(\pmb x;\pmb w)<0
\end{cases}
$$

一个**线性分类模型\(linear classification model\)**或**线性分类器\(linear classifier\)**，是由一个（或多个）线性的判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$和非线性的决策函数$$g(⋅)$$组成。

### 二分类

**二分类\(binary classification\)**问题的类别标签$$y$$只有两种取值，通常可以设为$$\{+1, −1\}$$或$$\{0, 1\}$$。在二分类问题中，常用**正例\(positive sample\)**和**负例\(negative sample\)**来分别表示属于类别 +1 和 −1 的样本。

在二分类问题中，我们只需要一个线性判别函数$$f(\pmb x;\pmb w)=\pmb w^{\rm T}\pmb x+b$$。 **特征空间**$$\mathbb{R}^D$$中所有满足$$f(\pmb x;\pmb w)=0$$的点组成一个分割**超平面\(hyperplane\)**，称为**决策边界\(decision boundary\)**或**决策平面\(decision surface\)**。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。

> 超平面就是三维空间中的平面在$$D$$维空间的推广。$$D$$维空间中的超平面是$$D − 1$$维的。

给定$$N$$个样本的训练集$$\mathcal{D} = \{(\pmb x^{(n)}, y^{(n)})\}_{n=1}^N$$，其中$$y^{(n)} ∈ \{+1, −1\}$$，线性模型试图学习到参数 $$\pmb w^∗$$，使得对于每个样本$$(\pmb x^{(n)},y^{(n)})$$尽量满足

$$
y^{(n)}f(\pmb x^{(n)};\pmb w^*)>0,\forall n\in [1,N]
$$

**定义** 如果存在权重向量$$\pmb w^*$$，使得上式对所有$$n$$满足，则称训练集$$\mathcal{D}$$是**线性可分**的。

为了学习参数$$\pmb w$$，我们需要定义合适的损失函数以及优化方法。对于二分类问题，最直接的损失函数为 0-1 损失函数，即

$$
\mathcal{L}(y,f(\pmb x;\pmb w))=I(yf(\pmb x;\pmb w)<0)
$$

其中$$I(\cdot)$$为指示函数。但 0-1 损失函数的数学性质不好，其关于$$\pmb w$$的导数为0，因而无法使用梯度下降法。

![Screenshot from 2020-10-27 21-52-01.png](https://i.loli.net/2020/10/27/Wo7N4GanOXms9BI.png)

### 多分类

**多分类\(multi-class classification\)**问题是指分类的类别数$$C$$大于 2 。多分类一般需要多个线性判别函数，但设计这些判别函数有很多种方式。

假设一个多分类问题的类别为$$\{1, 2, ⋯ , C\}$$，常用的方式有以下三种：

1. “一对其余”方式：把多分类问题转换为$$C$$个二分类问题，这种方式共需要$$C$$个判别函数，其中第$$c$$个判别函数$$f_c$$是将类别$$c$$的样本和不属于类别$$c$$的样本分开。
2. “一对一”方式：把多分类问题转换为$$C(C − 1)/2$$个 “一对一” 的二分类问题，这种方式共需要$$C(C − 1)/2$$个判别函数，其中第$$(i, j)$$个判别函数是把类别$$i$$和类别$$j$$的样本分开。
3. “argmax”方式：这是一种改进的“一对其余”方式，共需要$$C$$个判别函数

   $$
   f_c(\pmb x;\pmb w_c)=\pmb w_c^{\rm T} \pmb x+b_c,\quad c\in \{1,2,\cdots,C\}
   $$

   对于样本$$\pmb x$$，如果存在一个类别$$c$$，相对于所有的其他类别$$\tilde c(\tilde c ≠ c)$$有$$f_c(\pmb x;\pmb w_c ) > f_{\tilde c}(\pmb x;\pmb w_{\tilde c} )$$，那么$$\pmb x$$属于类别$$c$$。“argmax” 方式的预测函数定义为

   $$
   y=\arg \max_{c=1}^C f_c(\pmb x;\pmb w_c)
   $$

“一对其余”方式和“一对一”方式都存在一个缺陷：特征空间中会存在一些难以确定类别的区域，而“ argmax ”方式很好地解决了这个问题。下图给出了用这三种方式进行多分类的示例，其中红色直线表示判别函数$$f(⋅) = 0$$的直线，不同颜色的区域表示预测的三个类别的区域\($$ω_1 , ω_2$$和$$ω_3$$\)和难以确定类别的区域\(‘?’\)。在“argmax”方式中，相邻两类$$i$$和$$j$$的决策边界实际上是由$$f_i(\pmb x;\pmb w_i) − f_j(\pmb x;\pmb w_j) = 0$$决定, 其法向量为$$\pmb w_i −\pmb w_j$$。

![](https://i.loli.net/2020/09/17/lWAS7GHzIQEePBn.png)

> 图\(a\)实际是按照$${\rm sgn}(f_1),{\rm sgn}(f_2),{\rm sgn}(f_3)$$的组合划分（共$$\frac{(1+C)C}{2}+1$$个）区域。

**定义** 如果存在$$C$$个权重向量$$\pmb w_1^*,\pmb w_2^*,\cdots,\pmb w_C^*$$，使得第$$c$$类的所有样本都满足$$f_c(\pmb x;\pmb w_c^*) > f_{\tilde c}(\pmb x;\pmb w_{\tilde c}^*),\forall\tilde c\neq c$$，则称训练集$$\mathcal{D}$$是**多类线性可分**的。

由以上定义可知，如果训练集多类线性可分的，那么一定存在一个“argmax”方式的线性分类器可以将它们正确分开。

## Logistic回归

> Logistic函数是一种常用的S型函数，是比利时数学家Pierre François Verhulst在1844~1845年研究种群数量的增长模型时提出命名的，最初作为一种生态学模型。
>
> Logistic 函数定义为
>
> $$
> {\rm logistic}(x)=\frac{L}{1+\exp(-K(x-x_0))}
> $$
>
> 其中$$x_0$$是中心店，$$L$$是最大值，$$K$$是曲线的倾斜度。下图给出了几种不同参数的Logistic函数曲线
>
> ![Screenshot from 2020-10-29 10-35-36.png](https://i.loli.net/2020/10/29/HQDoZUIpzFS6Tn5.png)
>
> 当参数为$$x_0=0,L=1,k=1$$时，Logistic函数称为标准Logistic函数或Sigmoid函数，记作$$\sigma(x)$$
>
> $$
> \sigma(x)=\frac{1}{1+\exp(-x)}
> $$
>
> 标准Logistic函数在机器学习中使用得非常广泛，经常用来将一个实数空间的数映射到$$(0, 1)$$区间。

**Logistic 回归\(Logistic Regression , LR\)**是一种常用的处理二分类问题的线性模型。在本节中，我们采用$$y ∈ \{0, 1\}$$以符合 Logistic 回归的描述习惯。

这里引入非线性函数$$g:\mathbb{R}\to (0,1)$$来预测类别标签的后验概率$$\hat p(y=1|\pmb x)$$，即

$$
\hat p(y=1|\pmb x)=g(f(\pmb x;\pmb w))
$$

其中$$g(⋅)$$通常称为**激活函数\(activation function\)**，其作用是把线性函数的值域从实数区间 “挤压” 到了$$(0, 1)$$之间, 可以用来表示概率。

在 Logistic 回归中，我们使用 Logistic 函数来作为激活函数。标签$$y = 1$$的后验概率为

$$
\hat p(y=1|\pmb x)=\sigma(\pmb w^{\rm T}\pmb x)\\
\triangleq\frac{1}{1+\exp(-\pmb w^{\rm T}\pmb x)}
$$

为简单起见，这里$$\pmb x=[x_1,\cdots,x_D,1]^{\rm T},\pmb w=[w_1,\cdots,w_D,b]^{\rm T}$$分别为$$D+1$$维的**増广特征向量**和**増广权重向量**。

将上式进行变换得到

$$
\pmb w^{\rm T}\pmb x=\ln \frac{\hat p(y=1|\pmb x)}{\hat p(y=0|\pmb x)}
$$

其中$$\hat p(y=1|\pmb x)/\hat p(y=0|\pmb x)$$为样本$$\pmb x$$是正反例后验概率的比值，称为**几率\(odds\)**，几率的对数称为**对数几率\(log odds, 或logit\)**。上式等号左边是线性函数，因此Logistic回归可以看作预测值为标签的对数几率的线性回归模型。因此Logistic回归也称为**对数几率回归\(logit regression\)**。

![Linear Regression vs Logistic Regression - Javatpoint](https://static.javatpoint.com/tutorial/machine-learning/images/linear-regression-vs-logistic-regression.png)

![img](https://www.equiskill.com/wp-content/uploads/2018/07/WhatsApp-Image-2020-02-11-at-8.30.11-PM.jpeg)

### 参数学习

Logistic回归采用交叉熵作为损失函数，并使用梯度下降法对参数进行优化。

给定N个训练样本$$\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，用Logistic回归模型对每个样本$$\pmb x^{(n)}$$进行预测，输出其标签为1的后验概率，记作$$\hat y^{(n)}$$，

$$
\hat y^{(n)}=\sigma(\pmb w^{\rm T}\pmb x^{(n)}),\ 1\le n\le N
$$

由于$$y^{(n)}\in \{0,1\}$$，样本$$(\pmb x^{(n)},y^{(n)})$$的真实条件概率可以表示为

$$
p(y^{(n)}=1|\pmb x^{(n)})=y^{(n)}\\
p(y^{(n)}=0|\pmb x^{(n)})=1-y^{(n)}
$$

使用交叉熵损失函数，其风险函数为

$$
\mathcal{R}(\pmb w)=-\frac{1}{N}\sum_{n=1}^N(y^{(n)}\log \hat y^{(n)}+(1-y^{(n)})\log (1-\hat y^{(n)}))
$$

> 这里没有引入正则化项。

关于参数$$\pmb w$$的偏导数为

$$
\frac{\partial \mathcal{R}(\pmb w)}{\partial \pmb w}=-\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(y^{(n)}-\hat y^{(n)})
$$

采用梯度下降法，Logistic回归的训练过程为：初始化$$\pmb w_0←0$$，然后通过下式迭代更新参数：

$$
\pmb w_{t+1}←\pmb w_t+\alpha\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(y^{(n)}-\hat y_{\pmb w_t}^{(n)})
$$

其中$$\alpha$$是学习率，$$\hat y_{\pmb w_t}^{(n)}$$是当参数为$$\pmb w_t$$时，Logistic回归模型的输出。

此外，风险函数$$\mathcal{R}(\pmb w)$$是关于参数$$\pmb w$$的连续可导的凸函数，因此Logistic回归还可以使用凸优化中的高阶优化方法（如牛顿法）进行优化。

## Softmax回归

> Softmax函数将$$K$$维向量$$\pmb x=(x_1,\cdots,x_K)$$映射为一个概率分布
>
> $$
> \hat{\pmb z}={\rm softmax}(\pmb x)=\frac{\exp(\pmb x)}{\sum_{k=1}^K\exp(x_k)}=\frac{\exp(\pmb x)}{\pmb1_K^{\rm T} \exp(\pmb x)}
> $$

**Softmax回归\(Softmax regression\)**，也称为多项\(multinomial\)或多分类\(multi-class\)的Logistic回归，是Logistic回归在多分类问题上的推广。

对于多分类问题，类别标签$$y\in \{1,2,\cdots,C\}$$可以有$$C$$个取值。给定一个样本$$\pmb x$$，Softmax回归预测的属于类别c的后验概率为

$$
\hat p(y=c|\pmb x)={\rm softmax}(\pmb w_c^{\rm T}\pmb x)\\
=\frac{\exp(\pmb w_c^{\rm T}\pmb x)}{\sum_{i=1}^C \exp(\pmb w_i^{\rm T}\pmb x)}
$$

其中$$\pmb w_c$$是第c类的权重向量。

上式用向量形式可以写为

$$
\hat{\pmb y}={\rm softmax}(W^{\rm T}\pmb x)\\
=\frac{\exp(W^{\rm T}\pmb x)}{\pmb 1_C^{\rm T}\exp(W^{\rm T}\pmb x)}
$$

其中$$W=[\pmb w_1,\cdots,\pmb w_C]$$是由C个类的权重向量组成的矩阵，$$\pmb 1_C$$是C维的全1向量，$$\hat{\pmb y}\in\mathbb{R}^C$$是所有类别的预测后验概率组成的向量。

![Neural network softmax activation - Cross Validated](https://i.stack.imgur.com/0rewJ.png)

![Softmax Activation Function Explained \| by Dario Rade&#x10D;i&#x107; \| Towards Data Science](data:image/png;base64,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)

### 参数学习

给定N个训练样本$$\{\pmb x^{(n)},y^{(n)} \}^N_{n=1}$$，Softmax回归模型使用交叉熵损失函数来学习最优的参数矩阵$$W$$。

为了方便起见，我们用C维的one-hot向量$$\pmb y$$来表示类别标签。对于类别c，其向量表示为

$$
\pmb y=[I(c=1),I(c=2),\cdots,I(C=c)]^{\rm T}
$$

其中$$I(\cdot)$$是指示函数。

使用交叉熵损失函数，其风险函数为

$$
\mathcal{R}(W)=-\frac{1}{N}\sum_{n=1}^N(\pmb y^{(n)})^{\rm T}\log \hat{\pmb y}^{(n)}
$$

其中$$\hat{\pmb y}^{(n)}={\rm softmax}(W^{\rm T}\pmb x^{(n)})$$为样本$$\pmb x^{(n)}$$在每个类别的后验概率。

> 这里没有引入正则化项。

关于$$W$$的偏导数为

$$
\frac{\partial \mathcal{R}(W)}{\partial W}=-\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(\pmb y^{(n)}-\hat{\pmb y}^{(n)})^{\rm T}
$$

采用梯度下降法，Softmax回归的训练过程为：初始化$$W_0←0$$，然后通过下式迭代更新参数：

$$
W_{t+1}←W_t+\alpha\frac{1}{N}\sum_{n=1}^N\pmb x^{(n)}(\pmb y^{(n)}-\hat{\pmb y}_{W_t}^{(n)})^{\rm T}
$$

其中$$\alpha$$是学习率，$$\hat y_{W_t}^{(n)}$$是当参数为$$W_t$$时，Softmax回归模型的输出。

## 感知器

## 支持向量机

## 总结

