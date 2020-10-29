机器学习入门笔记——数学基础

[toc]

# 范数（线性代数）

**范数(norm)**是从向量空间到实数域的映射，并且满足正定性、绝对一次齐次性和三角不等式。对于一个$$N$$维向量$$\boldsymbol v$$，一个常见的范数函数为$$\ell_p$$范数
$$
\ell_p(\boldsymbol v)=||\boldsymbol v||_p=(\sum_{n=1}^N|v_n|^p)^{1/p}
$$
其中$$p\ge 0$$为一个标量的参数，常用的$$p$$的取值有1，2，∞等。
$$
\ell_1(\boldsymbol v)=||\boldsymbol v||_1=\sum_{n=1}^N|v_n|\\
\ell_2(\boldsymbol v)=||\boldsymbol v||_2=||\boldsymbol v||=\sqrt{\sum_{n=1}^Nv_n^2}
$$
$$\ell_2$$范数又称为Euclidean范数或者Frobenius范数。从几何角度，向量也可以表示为从原点出发的一个带箭头的有向线段，其$$\ell_2$$范数为线段的长度，也常称为向量的**模**。

下图给出了常见范数的示例，其中红线表示不同范数的$$\ell_p = 1 $$的点

![Screenshot from 2020-09-01 19-02-39.png](https://i.loli.net/2020/09/02/2VMxbcDkFCmasPn.png)





# 矩阵微积分

## 标量、向量与矩阵之间的导数

**标量关于向量的偏导数** 对于$$M$$维向量$$\pmb x\in \mathbb{R}^M$$和函数$$y=f(\pmb x)\in \mathbb{R}$$，$$y$$关于$$\pmb x$$的偏导数为
$$
\frac{\partial y}{\partial \pmb x}=[\frac{\partial y}{\partial  x_1},\cdots,\frac{\partial y}{\partial x_M}]^{\rm T}
$$

> 多元微积分中的梯度即是标量对向量求偏导数

$$y$$关于$$\pmb x$$的二阶偏导数为
$$
H=\frac{\partial^2 y}{\partial \pmb x^2}=\begin{bmatrix} \frac{\partial^2 y}{\partial x_1^2} & \cdots & \frac{\partial^2 y}{\partial  x_1 \partial x_M}\\
\vdots & \ddots & \vdots \\
\frac{\partial^2 y}{\partial x_M \partial x_1} & \cdots & \frac{\partial^2 y}{\partial  x_M^2}
\end{bmatrix}
\in \mathbb{R}^{M\times M}
$$
称为函数$$f(\pmb x)$$的**Hessian矩阵**，也写作$$\nabla^2 f(\pmb x)$$

**向量关于标量的偏导数** 对于标量$$x\in \mathbb{R}$$和函数$$\pmb y=f(x)\in \mathbb{R}^N$$，$$\pmb y$$关于$$x$$的偏导数为
$$
\frac{\partial \pmb y}{\partial x}=[\frac{\partial y_1}{\partial  x},\cdots,\frac{\partial y_N}{\partial x}]
$$
**向量关于向量的偏导数** 对于$$M$$维向量$$\pmb x\in \mathbb{R}^M$$和函数$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb y$$关于$$\pmb x$$的偏导数为
$$
\frac{\partial \pmb y}{\partial \pmb x}=\begin{bmatrix} \frac{\partial y_1}{\partial  x_1} & \cdots & \frac{\partial y_N}{\partial  x_1}\\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial  x_M} & \cdots & \frac{\partial y_N}{\partial  x_M}
\end{bmatrix}
\in \mathbb{R}^{M\times N}
$$
称为函数$$f(\pmb x)$$的**雅可比矩阵（Jacobian Matrix）**的转置。

**标量关于矩阵的偏导数** 对于$$M\times N$$维矩阵$$X\in \mathbb{R}^{M\times N}$$和函数$$y=f(X)\in \mathbb{R}$$，$$y$$关于$$X$$的偏导数为
$$
\frac{\partial y}{\partial X}=\begin{bmatrix}
\frac{\partial y}{\partial  x_{11}}&\cdots&\frac{\partial y}{\partial x_{1N}}\\
\vdots & \ddots & \vdots\\
\frac{\partial y}{\partial  x_{M1}}&\cdots&\frac{\partial y}{\partial x_{MN}}\\
\end{bmatrix}\in\mathbb{R}^{M\times N}
$$

**矩阵关于标量的偏导数**  对于标量$$x\in \mathbb{R}$$和函数$$Y=f(x)\in \mathbb{R}^{M\times N}$$，$$Y$$关于$$x$$的偏导数为
$$
\frac{\partial Y}{\partial x}=\begin{bmatrix}
\frac{\partial y_{11}}{\partial  x}&\cdots&\frac{\partial y_{M1}}{\partial x}\\
\vdots & \ddots & \vdots\\
\frac{\partial y_{1N}}{\partial  x}&\cdots&\frac{\partial y_{MN}}{\partial x}\\
\end{bmatrix}\in \mathbb{R}^{N\times M}
$$




@$$y=\pmb x^{\rm T}A\pmb x$$，其中$$\pmb x\in \mathbb{R}^n,A\in\mathbb{R}^{n\times n}$$，计算$$\frac{\partial y}{\partial \pmb x}$$。
$$
y=\pmb x^{\rm T}A\pmb x=\sum_{i=1}^n \sum_{j=1}^n a_{ij}x_ix_j\quad(二次型)\\
\frac{\partial y}{\partial x_1}=\sum_{i=1}^na_{i1}x_i+\sum_{j=1}^na_{1j}x_j=(A^{\rm T}\pmb x)_1+(A\pmb x)_1\\
\therefore \frac{\partial y}{\partial \pmb x}=A\pmb x+A^{\rm T}\pmb x
$$


@$$y={\rm tr}(A)$$其中$$A\in \mathbb{R}^{n\times n}$$，计算$$\frac{\partial y}{\partial A}$$。
$$
\frac{\partial y}{\partial a_{ij}}=\frac{\partial\sum_{k=1}^na_{kk}}{\partial a_{ij}}=\begin{cases}1,&i=j\\
0,&i\neq j
\end{cases}\\
\therefore \frac{\partial y}{\partial A}=I
$$



## 导数计算法则

### 加法法则

若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
$$
\frac{\partial (\pmb y+ \pmb z)}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}+\frac{\partial \pmb z}{\partial \pmb x} \in \mathbb{R}^{M\times N}
$$


### 乘法法则

1. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^N$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb y^{\rm T} \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\pmb z+\frac{\partial \pmb z}{\partial \pmb x}\pmb y \in \mathbb{R}^{M}
   $$

2. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^S$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^T$$，$$A \in \mathbb{R}^{S\times T}$$和$$\pmb x$$无关，则
   $$
   \frac{\partial \pmb y^{\rm T} A \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}A\pmb z+\frac{\partial \pmb z}{\partial \pmb x} A^{\rm T} \pmb y \in \mathbb{R}^{M}
   $$

3. 若$$\pmb x\in \mathbb{R}^M$$，$$y=f(\pmb x)\in \mathbb{R}$$，$$\pmb z=g(\pmb x)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial y \pmb z}{\partial \pmb x}=y\frac{\partial \pmb z}{\partial \pmb x}+\frac{\partial y}{\partial \pmb x}\pmb z^{\rm T} \in \mathbb{R}^{M\times N}
   $$

4. 若$$x\in \mathbb{R},Y\in \mathbb{R}^{M\times N},Z\in \mathbb{R}^{N\times P}$$，则
   $$
   \frac{\partial YZ}{\partial x}=Z^{\rm T}\frac{\partial Y}{\partial x}+\frac{\partial Z}{\partial x}Y^{\rm T}\in \mathbb{R}^{P\times M}
   $$



### 链式法则(Chain Rule)

1. 若$$x\in \mathbb{R}$$，$$\pmb y=f(x)\in \mathbb{R}^M$$，$$\pmb z=g(\pmb y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb z}{\partial x}=\frac{\partial \pmb y}{\partial x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{1\times N}
   $$

2. 若$$\pmb x\in \mathbb{R}^M$$，$$\pmb y=f(\pmb x)\in \mathbb{R}^K$$，$$\pmb z=g(\pmb y)\in \mathbb{R}^N$$，则
   $$
   \frac{\partial \pmb z}{\partial \pmb x}=\frac{\partial \pmb y}{\partial \pmb x}\frac{\partial \pmb z}{\partial \pmb y} \in \mathbb{R}^{M\times N}
   $$

3. 若$$X\in \mathbb{R}^{M\times N}$$，$$\pmb y=f(X)\in \mathbb{R}^K$$，$$z=g(\pmb y)\in \mathbb{R}$$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\frac{\partial \pmb y}{\partial x_{ij}}\frac{\partial z}{\partial \pmb y} \in \mathbb{R}
   $$

4. 若$$X\in \mathbb{R}^{M\times N}$$，$$Y=f(X)\in \mathbb{R}^{M\times N}$$，$$z=g(Y)\in \mathbb{R}$$，则
   $$
   \frac{\partial z}{\partial x_{ij}}=\sum_{p=1}^{M}\sum_{q=1}^{N}\frac{\partial y_{pq}}{\partial x_{ij}}\frac{\partial z}{\partial y_{pq}} \in \mathbb{R}
   $$



## 矩阵微分

回顾一元和多元微积分中的微分与导数的关系
$$
y=f(x):{\rm d}y=y'{\rm d}x\\
y=f(x_1,x_2,\cdots,x_n)=f(\pmb x):{\rm d}y=\sum_{i=1}^n\frac{\partial y}{\partial x_i}{\rm d}x_i=(\frac{\partial y}{\partial \pmb x})^{\rm T}{\rm d}\pmb x
$$
类似地，我们建立矩阵微分与导数的关系
$$
y=f(X):{\rm d}y=\sum_{i=1}^m\sum_{j=1}^n \frac{\partial y}{\partial x_{ij}}{\rm d}x_{ij}={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)
$$

> $${\rm tr}(A^{\rm T}B)=\sum_{i,j} a_{ij}b_{ij}$$称为矩阵$$A,B$$的内积

| 常用微分   |                                                              |
| ---------- | ------------------------------------------------------------ |
| 加减法     | $${\rm d}(X\pm Y)={\rm d}X\pm {\rm d}Y$$                     |
| 数乘       | $${\rm d}(\alpha X)=\alpha {\rm d} X$$                       |
| 乘法       | $${\rm d}(XY)={\rm d}X\ Y+X{\rm d}Y$$                        |
| 幂         | $${\rm d}X^n=\sum_{i=0}^{n-1} X^i {\rm d}X X^{n-1-i}$$       |
| 转置       | $${\rm d}(X^{\rm T})=({\rm d}X)^{T}$$                        |
| 迹         | $${\rm d}{\rm tr}(X)={\rm tr}({\rm d}X)$$                    |
| 逆         | $${\rm d}X^{-1}=-X^{-1}{\rm d}XX^{-1}$$                      |
| 行列式     | $${\rm d}\vert X\vert={\rm tr}(X^*{\rm d}X)$$, $$X^*$$为伴随矩阵 |
|            | $${\rm d}\vert X\vert=\vert X\vert{\rm tr}(X^{-1}{\rm d}X)$$, 如果$$X$$可逆 |
| 逐元素乘法 | $${\rm d}(X\odot Y)={\rm d}X\odot Y+X\odot {\rm d}Y $$       |
| 逐元素函数 | $${\rm d}\sigma(X)=\sigma'(X)\odot {\rm d}X$$, $$\sigma$$为逐元素函数运算 |




## 导数计算的微分方法

微分方法通过推导出微分与导数的关系式得到导数
$$
{\rm d}y=(\frac{\partial y}{\partial \pmb x})^{\rm T}{\rm d}\pmb x\\
{\rm d}y={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)
$$
计算对矩阵的偏导数时，一些迹技巧（trace trick）非常有用：
$$
a={\rm tr}(a)\\
{\rm tr}(A^{\rm T})={\rm tr}(A)\\
{\rm tr}(A\pm B)={\rm tr}(A)\pm {\rm tr}(B)\\
{\rm tr}(AB)={\rm tr}(BA)\\
{\rm tr}(A^{\rm T}(B\odot C))={\rm tr}((A\odot B)^{\rm T} C)
$$



@$$y=\pmb x^{\rm T}A\pmb x$$，其中$$\pmb x\in \mathbb{R}^n,A\in\mathbb{R}^{n\times n}$$，计算$$\frac{\partial y}{\partial \pmb x}$$。
$$
{\rm d}y={\rm d}(\pmb x^{\rm T})A\pmb x+\pmb x^{\rm T}{\rm d}A\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=({\rm d}\pmb x)^{\rm T}A\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=\pmb x^{\rm T}A^{\rm T}{\rm d}\pmb x+\pmb x^{\rm T}A{\rm d}\pmb x\\
=(A\pmb x+A^{\rm T}\pmb x)^{\rm T}{\rm d}\pmb x\\
\therefore \frac{\partial y}{\partial \pmb x}=A\pmb x+A^{\rm T}\pmb x
$$



@$$W\in\mathbb{R}^{R\times S}$$，$$X=g(W)=AWB\in\mathbb{R}^{M\times N}$$，$$y=f(X)\in \mathbb{R}$$，$$\frac{\partial y}{\partial X}$$已知，求$$\frac{\partial y}{\partial W}$$。
$$
\because {\rm d}y={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}{\rm d}X)={\rm tr}((\frac{\partial y}{\partial X})^{\rm T}A{\rm d}WB)={\rm tr}(B(\frac{\partial y}{\partial X})^{\rm T}A{\rm d}W)={\rm tr}((A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T})^{\rm T}{\rm d}W)\\
\therefore \frac{\partial y}{\partial W}=A^{\rm T}(\frac{\partial y}{\partial X})B^{\rm T}
$$





# 梯度下降法（数学优化）

**梯度下降法(gradient descent method)**是一个一阶最优化算法，其基本思想是，每次迭代用当前位置的负梯度方向作为搜索方向。因为对于可微函数$$f(\pmb x)$$，函数值下降最快的方向就是<u>负梯度方向</u>。

> 实际上，负梯度方向是欧氏度量意义下的最速下降方向，即条件为限制方向的$$\ell_2$$范数$$||\pmb p||$$不大于1。若改用其它度量，得到的最速下降方向与会有所不同。负梯度方向作为最速下降方向的最速下降法(steepest descent method)称为梯度下降法。

梯度下降法的计算步骤如下：

1. 给定初点$$\pmb x^{(1)}\in\mathbb{R}^n$$，允许误差$$\varepsilon >0$$，置$$k=1$$

2. 计算搜索方向$$\pmb p^{(k)}=-\nabla f(\pmb x^{(k)})$$

3. 若$$||\pmb p^{(k)}||\le \varepsilon$$，则停止计算；否则从$$\pmb x^{(k)}$$出发，沿$$\pmb p^{(k)}$$进行线搜索，求$$\alpha_k$$使得
   $$
   f(\pmb x^{(k)}+\alpha_k\pmb p^{(k)})=\min_{\alpha \ge 0}f(\pmb x^{(k)}+\alpha\pmb p^{(k)})
   $$

   > 线搜索可以使用解析法、试探法、插值法、回溯法等，这里使用解析法

4. 令$$\pmb x^{(k+1)}=\pmb x^{(k)}+\alpha_k\pmb p^{(k)}$$，置$$k=k+1$$，goto 2



@用梯度下降法解下列非线性规划问题：
$$
\min\quad f(\pmb x)=2x_1^2+x_2^2
$$
初点$$\pmb x^{(1)}=(1,1)^{\rm T},\varepsilon = 0.1$$。

计算梯度
$$
\nabla f(\pmb x)=\begin{bmatrix}4x_1\\2x_2 \end{bmatrix}
$$
第1次迭代，
$$
\pmb p^{(1)}=-\nabla f(\pmb x^{(1)})=\begin{bmatrix}-4\\-2 \end{bmatrix},\ ||\pmb p^{(1)}||=2\sqrt{5}>0.1\\
\min_{\alpha\ge 0}\quad\varphi (\alpha)\triangleq f(\pmb x^{(1)}+\alpha\pmb p^{(1)})=f(\begin{bmatrix}1-4\alpha\\1-2\alpha \end{bmatrix})=2(1-4\alpha)^2+(1-2\alpha)^2\\
令\varphi'(\alpha)=-16(1-4\alpha)-4(1-2\alpha)=0\Rightarrow
\alpha_1=\frac{5}{18}\\
\pmb x^{(2)}=\pmb x^{(1)}+\alpha_1\pmb p^{(1)}=\begin{bmatrix}-1/9 \\4/9 \end{bmatrix}
$$

类似地，第2, 3次迭代，
$$
\pmb x^{(3)}=\pmb x^{(2)}+\alpha_2\pmb p^{(2)}=\frac{2}{27}\begin{bmatrix}1\\1 \end{bmatrix}\\
\pmb x^{(4)}=\pmb x^{(3)}+\alpha_3\pmb p^{(3)}=\frac{2}{243}\begin{bmatrix}-1\\4 \end{bmatrix}\\
$$
达到精度要求$$||\nabla f(\pmb x^{(4)})||=\frac{8}{243}\sqrt{5}<0.1$$，于是近似解$$\overline{\pmb x}=\frac{2}{243}(-1,4)^{\rm T}$$。实际上，问题的最优解为$$\pmb x^*=(0,0)^{\rm T}$$。



**定理** 设$$f(\pmb x)$$是连续可微实函数，解集合$$\Omega=\{\overline{\pmb x}|\nabla f(\overline{\pmb x})=\pmb 0 \}$$，梯度下降法产生的序列$$\{\pmb x^{(k)}\}$$包含于某个紧集，则序列$$\{\pmb x^{(k)}\}$$的每个聚点$$\hat{\pmb x}\in \Omega$$。



**锯齿现象** 用梯度下降法极小化目标函数时，相邻的两个搜索方向是正交的，

> 因为
> $$
> \varphi(\alpha)= f(\pmb x^{(k)}+\alpha\pmb p^{(k)}), \\
> \pmb p^{(k)}=-\nabla f(\pmb x^{(k)}), \\
> 令\ \varphi'(\alpha)=\pmb p^{(k){\rm T}} \nabla f(\pmb x^{(k)}+\alpha\pmb p^{(k)})=0\Rightarrow \alpha=\alpha_k \\
> \Rightarrow \pmb p^{(k){\rm T}}\nabla f(\pmb x^{(k+1)})=0\Rightarrow -\pmb p^{(k){\rm T}}\pmb p^{(k+1)}=0
> $$
> 即方向$$\pmb p^{(k)},\pmb p^{(k+1)}$$正交

这表明迭代产生的序列$$\{\pmb x^{(k)}\}$$所循路径是“之”字形的，如下图所示。特别是当$$\pmb x^{(k)}$$接近极小点$$\overline{\pmb x}$$时，每次迭代移动的步长很小，于是出现了**锯齿现象**，影响了收敛速率。

![Banana-SteepDesc.gif](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Banana-SteepDesc.gif/400px-Banana-SteepDesc.gif)

从局部看，最速下降方向是函数值下降最快的方向，选择这样的方向进行搜索是有利的；但从全局看，由于锯齿现象的影响，收敛速率大为减慢。梯度下降法并不是收敛最快的方法，相反，从全局看，它的收敛是比较慢的。因此梯度下降法一般适用于计算过程的前期迭代，而不适用于后期接近极小点的情形。





# 熵和交叉熵（信息论）

**熵(entropy)**最早是物理学的概念，用于表示一个热力学系统的无序程度。在信息论中，熵用来衡量一个随机事件的不确定性。



## 自信息和熵

**自信息(self information)**表示一个随机事件所包含的信息量。一个随机事件发生的概率越高，其自信息越低。如果一个事件必然发生，其自信息为0。

对于一个随机变量$$X$$（取值集合为$$\mathcal{X}$$，概率分布为$$p(x),x∈\mathcal{X}$$），当$$X=x$$时的自信息$$I(x)$$定义为
$$
I(x) = − \log p(x)
$$
在自信息的定义中，对数的底可以使用2、自然常数e或是10。当底为2时，自信息的单位为bit；当底为e时，自信息的单位为nat。默认底为2。

对于分布为$$p(x)$$的随机变量$$X$$，其自信息的数学期望，即熵$$H(X)$$定义为
$$
H(X)=E(I(x))=E(−\log p(x))
=−\sum_{x\in \mathcal{X}} p(x)\log p(x)
$$

其中当$$p(x_i)=0$$时，约定$$0\log0=0$$（因为$$x\to 0\Rightarrow x\log x\to 0$$）。

熵越高则随机变量的信息越多，熵越低则信息越少。如果变量$$X$$当且仅当在$$x$$时$$p(x)=1$$，则熵为0。也就是说，对于一个确定的信息，其熵为0，信息量也为0。如果其概率分布为一个均匀分布，则熵最大。



@设随机变量$$X$$服从参数为$$p$$的伯努利分布，那么$$H(X)$$与$$p$$的关系为
$$
H(X)=-p\log p-(1-p)\log (1-p)
$$
![Screenshot from 2020-10-15 10-28-12.png](https://i.loli.net/2020/10/15/51GPfgUSVjkL98e.png)



## 交叉熵

**交叉熵(cross entropy)**可以用于衡量两个概率分布$$p(x),q(x)$$之间的差异，定义为
$$
H(p, q) = E (− \log q(x))= − \sum_x p(x) \log q(x)
$$
在给定$$p$$的情况下，如果$$q$$和$$p$$越接近，交叉熵越小；如果$$q$$和$$p$$越远，交叉熵就越大。

> 对于分布为$$p(x)$$的随机变量，熵$$H(p)$$表示其最优编码长度。交叉熵是按照概率分布$$q$$的最优编码对真实分布为$$p$$的信息进行编码的长度。



@设离散概率分布$$p(x),q_1(x),q_2(x)$$为
$$
\begin{array}{c|ccccc}
x & 0 & 1 & 2\\
\hline
p(x) & 0 & 1 & 0
\end{array},\
\begin{array}{c|ccccc}
x & 0 & 1 & 2\\
\hline
q_1(x) & 0.05 & 0.9 & 0.05
\end{array},\
\begin{array}{c|ccccc}
x & 0 & 1 & 2\\
\hline
q_2(x) & 0.3 & 0.4 & 0.3
\end{array}
$$
求交叉熵$$H(p,p),H(p,q_1),H(p,q_2)$$。
$$
H(p,p)=-0\log 0-1\log 1-0\log 0 = 0\\
H(p,q_1)=-0\log 0.05 -1\log 0.9-0\log0.05=0.152\\
H(p,q_2)=-0\log 0.05 -1\log 0.4-0\log0.05=1.322\\
$$




# 参考

+ The Matrix Cookbook
+ Convex Optimization, Stephen Boyd
+ Elements of Information Theory, Thomas M. Cover

