# 具有不等式约束的拉格朗日乘子法

> 原文：[`machinelearningmastery.com/lagrange-multiplier-approach-with-inequality-constraints/`](https://machinelearningmastery.com/lagrange-multiplier-approach-with-inequality-constraints/)

在之前的文章中，我们介绍了[拉格朗日乘子法](https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/)用于寻找具有等式约束的函数的局部最小值或最大值。同样的方法也可以应用于具有不等式约束的情况。

在本教程中，您将发现拉格朗日乘子法应用于当存在不等式约束时寻找函数的局部最小值或最大值的方法，也可以与等式约束一起使用。

完成本教程后，您将了解

+   如何找到具有等式约束的函数的局部最大值或最小值

+   具有等式约束的拉格朗日乘子法

让我们开始吧。

![具有不等式约束的拉格朗日乘子法](img/3bcfc4cf00942cfcadcc9024019a6378.png)

具有不等式约束的拉格朗日乘子法

图片由 [Christine Roy](https://unsplash.com/@agent_illustrateur) 提供，保留所有权利。

## 前提条件

对于本教程，我们假设您已审阅：

+   [函数的导数](https://machinelearningmastery.com/a-gentle-introduction-to-function-derivatives/)

+   [多个变量的函数、偏导数和梯度向量](https://machinelearningmastery.com/a-gentle-introduction-to-partial-derivatives-and-gradient-vectors)

+   [优化的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-optimization-mathematical-programming)

+   [梯度下降](https://machinelearningmastery.com/a-gentle-introduction-to-gradient-descent-procedure)

以及

+   [拉格朗日乘子法的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/)

您可以通过点击上述链接来回顾这些概念。

## 受限优化和拉格朗日函数

从我们[之前的帖子](https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/)扩展，受限优化问题通常可以看作是

$$

\begin{aligned}

\min && f(X) \\

\textrm{subject to} && g(X) &= 0 \\

&& h(X) &\ge 0 \\

&& k(X) &\le 0

\end{aligned}

$$

其中 $X$ 是标量或向量值。这里，$g(X)=0$ 是等式约束，而 $h(X)\ge 0$，$k(X)\le 0$ 是不等式约束。请注意，我们在优化问题中总是使用 $\ge$ 和 $\le$，而不是 $\gt$ 和 $\lt$，因为前者定义了数学中的**闭集**，我们应该从中寻找 $X$ 的值。这些约束在优化问题中可以有很多种。

等式约束易于处理，但不等式约束则不然。因此，简化处理的一种方法是将不等式转化为等式，引入**松弛变量**：

$$

\begin{aligned}

\min && f(X) \\

\textrm{subject to} && g(X) &= 0 \\

&& h(X) – s² &= 0 \\

&& k(X) + t² &= 0

\end{aligned}

$$

当某些事物是负的时候，向其添加一定的正量将使其等于零，反之亦然。这个量就是松弛变量；上述的$s²$和$t²$就是例子。我们特意将$s²$和$t²$项放在那里以表示它们不能为负。

引入松弛变量后，我们可以使用拉格朗日乘子法来解决问题，其中拉格朗日函数定义如下：

$$

L(X, \lambda, \theta, \phi) = f(X) – \lambda g(X) – \theta (h(X)-s²) + \phi (k(X)+t²)

$$

了解对于问题的最优解$X^*$来说，不等式约束要么等式成立（此时松弛变量为零），要么不成立是很有用的。这些等式成立的不等式约束被称为激活约束。否则，是非激活约束。从这个意义上说，你可以认为等式约束总是激活的。

## 互补松弛条件

我们需要知道约束条件是否处于激活状态，这是因为克拉斯-库恩-塔克 (KKT) 条件。准确地说，KKT 条件描述了在$X^*$是受约束优化问题的最优解时会发生什么：

1.  拉格朗日函数的梯度为零

1.  所有约束条件均满足

1.  不等式约束满足互补松弛条件

最重要的是互补松弛条件。虽然我们了解到，通过等式约束的优化问题可以使用拉格朗日乘子来解决，即拉格朗日函数的梯度在最优解处为零，互补松弛条件通过说在最优解$X^*$处，要么拉格朗日乘子为零，要么对应的不等式约束是激活的，将此扩展到不等式约束的情况。

使用互补松弛条件有助于我们探索解决优化问题的不同情况。最好通过一个例子来解释。

## 示例 1：均值-方差组合优化

这是一个来自金融领域的例子。如果我们有 1 美元并且打算进行两种不同的投资，其中它们的回报被建模为双变量高斯分布。我们应该如何分配资金以最小化整体回报的方差？

此优化问题，也称为马尔科维茨均值-方差组合优化，可以表述为：

$$

\begin{aligned}

\min && f(w_1, w_2) &= w_1²\sigma_1²+w_2²\sigma_2²+2w_1w_2\sigma_{12} \\

\textrm{subject to} && w_1+w_2 &= 1 \\

&& w_1 &\ge 0 \\

&& w_1 &\le 1

\end{aligned}

$$

其中最后两个约束用于将每项投资的权重限制在 0 到 1 美元之间。假设 $\sigma_1²=0.25$，$\sigma_2²=0.10$，$\sigma_{12} = 0.15$，那么拉格朗日函数定义为：

$$

\begin{aligned}

L(w_1,w_2,\lambda,\theta,\phi) =& 0.25w_1²+0.1w_2²+0.3w_1w_2 \\

&- \lambda(w_1+w_2-1) \\

&- \theta(w_1-s²) – \phi(w_1-1+t²)

\end{aligned}

$$

我们有梯度：

$$

\begin{aligned}

\frac{\partial L}{\partial w_1} &= 0.5w_1+0.3w_2-\lambda-\theta-\phi \\

\frac{\partial L}{\partial w_2} &= 0.2w_2+0.3w_1-\lambda \\

\frac{\partial L}{\partial\lambda} &= 1-w_1-w_2 \\

\frac{\partial L}{\partial\theta} &= s²-w_1 \\

\frac{\partial L}{\partial\phi} &= 1-w_1-t²

\end{aligned}

$$

从这一点开始，必须考虑互补松弛条件。我们有两个松弛变量 $s$ 和 $t$，相应的拉格朗日乘子是 $\theta$ 和 $\phi$。现在我们需要考虑一个松弛变量是否为零（对应的不等式约束是激活的）或拉格朗日乘子是否为零（约束是不激活的）。有四种可能的情况：

1.  $\theta=\phi=0$ 和 $s²>0$，$t²>0$

1.  $\theta\ne 0$ 但 $\phi=0$，并且 $s²=0$，$t²>0$

1.  $\theta=0$ 但 $\phi\ne 0$，并且 $s²>0$，$t²=0$

1.  $\theta\ne 0$ 和 $\phi\ne 0$，并且 $s²=t²=0$

对于案例 1，使用 $\partial L/\partial\lambda=0$，$\partial L/\partial w_1=0$ 和 $\partial L/\partial w_2=0$ 我们得到

$$

\begin{align}

w_2 &= 1-w_1 \\

0.5w_1 + 0.3w_2 &= \lambda \\

0.3w_1 + 0.2w_2 &= \lambda

\end{align}

$$

我们得到 $w_1=-1$，$w_2=2$，$\lambda=0.1$。但通过 $\partial L/\partial\theta=0$，我们得到 $s²=-1$，这没有解 ($s²$ 不能为负)。因此这个案例不可行。

对于案例 2，通过 $\partial L/\partial\theta=0$ 我们得到 $w_1=0$。因此，从 $\partial L/\partial\lambda=0$ 我们知道 $w_2=1$。并且通过 $\partial L/\partial w_2=0$，我们找到 $\lambda=0.2$，从 $\partial L/\partial w_1$ 我们得到 $\phi=0.1$。在这种情况下，目标函数的值为 0.1

对于案例 3，通过 $\partial L/\partial\phi=0$ 我们得到 $w_1=1$。因此，从 $\partial L/\partial\lambda=0$ 我们知道 $w_2=0$。并且通过 $\partial L/\partial w_2=0$，我们得到 $\lambda=0.3$，从 $\partial L/\partial w_1$ 我们得到 $\theta=0.2$。在这种情况下，目标函数的值为 0.25

对于案例 4，通过 $\partial L/\partial\theta=0$ 我们得到 $w_1=0$，但通过 $\partial L/\partial\phi=0$ 我们得到 $w_1=1$。因此这个案例不可行。

比较案例 2 和案例 3 的目标函数值，我们看到案例 2 的值更低。因此，我们将其作为优化问题的解，最优解为 $w_1=0$，$w_2=1$。

作为练习，你可以重新尝试上述问题，$\sigma_{12}=-0.15$。解决方案将是 0.0038，当 $w_1=\frac{5}{13}$，两个不等式约束不激活时。

### 想开始学习机器学习中的微积分吗？

立即参加我的免费 7 天电子邮件速成课程（附示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 示例 2：水填充算法

这是一个来自通信工程的例子。如果我们有一个信道（例如，一个无线带宽），其噪声功率为 $N$，信号功率为 $S$，则信道容量（以每秒比特数计）与 $\log_2(1+S/N)$ 成正比。如果我们有 $k$ 个类似的信道，每个信道都有其自身的噪声和信号水平，则所有信道的总容量是 $\sum_i \log_2(1+S_i/N_i)$ 的总和。

假设我们使用的电池只能提供 1 瓦特的功率，这个功率必须分配到 $k$ 个信道（记作 $p_1,\cdots,p_k$）。每个信道可能有不同的衰减，因此最终，信号功率会被每个信道的增益 $g_i$ 折扣。然后，利用这 $k$ 个信道可以达到的最大总容量被表述为一个优化问题

$$

\begin{aligned}

\max && f(p_1,\cdots,p_k) &= \sum_{i=1}^k \log_2\left(1+\frac{g_ip_i}{n_i}\right) \\

\textrm{受限于} && \sum_{i=1}^k p_i &= 1 \\

&& p_1,\cdots,p_k &\ge 0 \\

\end{aligned}

$$

为了方便微分，我们注意到 $\log_2x=\log x/\log 2$ 和 $\log(1+g_ip_i/n_i)=\log(n_i+g_ip_i)-\log(n_i)$，因此目标函数可以替换为

$$

f(p_1,\cdots,p_k) = \sum_{i=1}^k \log(n_i+g_ip_i)

$$

假设我们有 $k=3$ 个信道，每个信道的噪声水平分别为 1.0、0.9、1.0，信道增益为 0.9、0.8、0.7，则优化问题为

$$

\begin{aligned}

\max && f(p_1,p_2,p_k) &= \log(1+0.9p_1) + \log(0.9+0.8p_2) + \log(1+0.7p_3)\\

\textrm{受限于} && p_1+p_2+p_3 &= 1 \\

&& p_1,p_2,p_3 &\ge 0

\end{aligned}

$$

我们这里有三个不等式约束。拉格朗日函数定义为

$$

\begin{aligned}

& L(p_1,p_2,p_3,\lambda,\theta_1,\theta_2,\theta_3) \\

=\ & \log(1+0.9p_1) + \log(0.9+0.8p_2) + \log(1+0.7p_3) \\

& – \lambda(p_1+p_2+p_3-1) \\

& – \theta_1(p_1-s_1²) – \theta_2(p_2-s_2²) – \theta_3(p_3-s_3²)

\end{aligned}

$$

梯度因此为

$$

\begin{aligned}

\frac{\partial L}{\partial p_1} & = \frac{0.9}{1+0.9p_1}-\lambda-\theta_1 \\

\frac{\partial L}{\partial p_2} & = \frac{0.8}{0.9+0.8p_2}-\lambda-\theta_2 \\

\frac{\partial L}{\partial p_3} & = \frac{0.7}{1+0.7p_3}-\lambda-\theta_3 \\

\frac{\partial L}{\partial\lambda} &= 1-p_1-p_2-p_3 \\

\frac{\partial L}{\partial\theta_1} &= s_1²-p_1 \\

\frac{\partial L}{\partial\theta_2} &= s_2²-p_2 \\

\frac{\partial L}{\partial\theta_3} &= s_3²-p_3 \\

\end{aligned}

$$

但现在我们有 3 个松弛变量，需要考虑 8 种情况：

1.  $\theta_1=\theta_2=\theta_3=0$，因此 $s_1²,s_2²,s_3²$ 均非零

1.  $\theta_1=\theta_2=0$ 但 $\theta_3\ne 0$，因此仅有 $s_3²=0$

1.  $\theta_1=\theta_3=0$ 但 $\theta_2\ne 0$，因此仅有 $s_2²=0$

1.  $\theta_2=\theta_3=0$ 但 $\theta_1\ne 0$，因此仅有 $s_1²=0$

1.  $\theta_1=0$ 但 $\theta_2,\theta_3$ 非零，因此仅有 $s_2²=s_3²=0$

1.  $\theta_2=0$ 但 $\theta_1,\theta_3$ 非零，因此仅有 $s_1²=s_3²=0$

1.  $\theta_3=0$ 但 $\theta_1,\theta_2$ 非零，因此仅有 $s_1²=s_2²=0$

1.  所有的 $\theta_1,\theta_2,\theta_3$ 都非零，因此 $s_1²=s_2²=s_3²=0$

我们可以立即得出案例 8 不可行，因为从 $\partial L/\partial\theta_i=0$ 我们可以使 $p_1=p_2=p_3=0$，但无法使 $\partial L/\partial\lambda=0$。

对于案例 1，我们有

$$

\frac{0.9}{1+0.9p_1}=\frac{0.8}{0.9+0.8p_2}=\frac{0.7}{1+0.7p_3}=\lambda

$$

从 $\partial L/\partial p_1=\partial L/\partial p_2=\partial L/\partial p_3=0$。结合 $p_3=1-p_1-p_2$ 从 $\partial L/\partial\lambda=0$，我们找到解为 $p_1=0.444$，$p_2=0.430$，$p_3=0.126$，目标函数 $f(p_1,p_2,p_3)=0.639$。

对于案例 2，我们有 $p_3=0$ 从 $\partial L/\partial\theta_3=0$。进一步，使用 $p_2=1-p_1$ 从 $\partial L/\partial\lambda=0$，以及

$$

\frac{0.9}{1+0.9p_1}=\frac{0.8}{0.9+0.8p_2}=\lambda

$$

从 $\partial L/\partial p_1=\partial L/\partial p_2=0$，我们可以解得 $p_1=0.507$ 和 $p_2=0.493$。目标函数 $f(p_1,p_2,p_3)=0.634$。

同样在案例 3 中，$p_2=0$，我们解得 $p_1=0.659$ 和 $p_3=0.341$，目标函数 $f(p_1,p_2,p_3)=0.574$。

在案例 4 中，我们有 $p_1=0$，$p_2=0.652$，$p_3=0.348$，目标函数 $f(p_1,p_2,p_3)=0.570$。

案例 5 中我们有 $p_2=p_3=0$，因此 $p_3=1$。因此我们有目标函数 $f(p_1,p_2,p_3)=0.0.536$。

同样在案例 6 和案例 7 中，我们分别有 $p_2=1$ 和 $p_1=1$。目标函数分别为 0.531 和 0.425。

比较所有这些情况，我们发现目标函数的最大值出现在案例 1。因此，该优化问题的解是

$p_1=0.444$，$p_2=0.430$，$p_3=0.126$，目标函数 $f(p_1,p_2,p_3)=0.639$。

## 扩展阅读

在上述例子中，我们将松弛变量引入了拉格朗日函数，一些书籍可能更倾向于不添加松弛变量，而是将拉格朗日乘子限制为正值。在这种情况下，你可能会看到拉格朗日函数写作

$$

L(X, \lambda, \theta, \phi) = f(X) – \lambda g(X) – \theta h(X) + \phi k(X)

$$

但需要 $\theta\ge 0;\phi\ge 0$。

拉格朗日函数在原对偶方法中也很有用，用于找到最大值或最小值。如果目标或约束是非线性的，这特别有帮助，因为解决方案可能不容易找到。

一些涵盖此主题的书籍包括：

+   [凸优化](https://amzn.com/0521833787) 由 Stephen Boyd 和 Lieven Vandenberghe 著，2004

+   [深度学习](https://amzn.com/0262035618) 第四章，作者 Ian Goodfellow 等，2016

## 总结

在本教程中，你了解了如何将拉格朗日乘子方法应用于不等式约束。具体而言，你学到了：

+   拉格朗日乘子和拉格朗日函数在不等式约束存在时

+   如何利用 KKT 条件解决给定不等式约束的优化问题
