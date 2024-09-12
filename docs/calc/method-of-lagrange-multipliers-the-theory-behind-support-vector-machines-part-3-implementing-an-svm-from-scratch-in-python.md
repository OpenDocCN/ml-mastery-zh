# 拉格朗日乘数法：支持向量机背后的理论（第三部分：在 Python 中从头开始实现 SVM）

> 原文：[`machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-3-implementing-an-svm-from-scratch-in-python/`](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-3-implementing-an-svm-from-scratch-in-python/)

支持向量机（SVM）分类器背后的数学是美丽的。重要的是不仅要学习 SVM 的基本模型，还要知道如何从头开始实现整个模型。这是我们关于 SVM 的系列教程的延续。在本系列的 [第一部分](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case) 和 [第二部分](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-2-the-non-separable-case) 中，我们讨论了线性 SVM 背后的数学模型。在本教程中，我们将展示如何使用 Python 的 SciPy 库中提供的优化例程构建 SVM 线性分类器。

完成本教程后，您将了解：

+   如何使用 SciPy 的优化例程

+   如何定义目标函数

+   如何定义界限和线性约束

+   如何在 Python 中实现自己的 SVM 分类器

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/12/Untitled.png)

拉格朗日乘数法：支持向量机背后的理论（第三部分：在 Python 中从头开始实现 SVM）

由 Thomas Sayre 创作的雕塑 Gyre，摄影师 Mehreen Saeed，部分权利保留。

## 教程概述

本教程分为 2 部分；它们是：

1.  SVM 的优化问题

1.  在 Python 中解决优化问题

    1.  定义目标函数

    1.  定义界限和线性约束

1.  使用不同的 C 值解决问题

## 先决条件

对于本教程，假设您已经熟悉以下主题。您可以单击各个链接获取更多详细信息。

+   [优化/数学规划的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-optimization-mathematical-programming/)

+   [拉格朗日乘数法的温和介绍](https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/)

+   [拉格朗日乘数法与不等式约束](https://machinelearningmastery.com/lagrange-multiplier-approach-with-inequality-constraints/)

+   [拉格朗日乘数法：支持向量机背后的理论（第一部分：可分离情况）](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case))

+   [拉格朗日乘子的法则：支持向量机背后的理论（第二部分：不可分离情况）](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-2-the-non-separable-case)

## 符号和假设

基本的 SVM 机器假设是一个二分类问题。假设我们有 $m$ 个训练点，每个点是一个 $n$ 维向量。我们将使用以下符号：

+   $m$: 总训练点数

+   $n$: 每个训练点的维度

+   $x$: 数据点，是一个 $n$ 维向量

+   $i$: 用于索引训练点的下标。 $0 \leq i < m$

+   $k$: 用于索引训练点的下标。 $0 \leq k < m$

+   $j$: 用于索引训练点每个维度的下标

+   $t$: 数据点的标签。它是一个 $m$ 维向量，其中 $t_i \in \{-1, +1\}$

+   $T$: 转置操作符

+   $w$: 权重向量，表示超平面的系数。它也是一个 $n$ 维向量

+   $\alpha$: 拉格朗日乘子向量，也是一个 $m$ 维向量

+   $C$: 用户定义的惩罚因子/正则化常数

## SVM 优化问题

SVM 分类器最大化以下拉格朗日对偶：

$$

L_d = -\frac{1}{2} \sum_i \sum_k \alpha_i \alpha_k t_i t_k (x_i)^T (x_k) + \sum_i \alpha_i

$$

上述函数受以下约束条件的限制：

\begin{eqnarray}

0 \leq \alpha_i \leq C, & \forall i\\

\sum_i \alpha_i t_i = 0& \\

\end{eqnarray}

我们要做的就是找到与每个训练点相关的拉格朗日乘子 $\alpha$，同时满足上述约束条件。

### 想开始学习机器学习的微积分？

现在就拿我的免费 7 天电子邮件速成课程（附样例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## SVM 的 Python 实现

我们将使用 SciPy 优化包来找到拉格朗日乘子的最佳值，并计算软间隔和分离超平面。

### 导入部分和常量

让我们编写优化、绘图和合成数据生成的导入部分。

Python

```py
import numpy as np    
# For optimization
from scipy.optimize import Bounds, BFGS                     
from scipy.optimize import LinearConstraint, minimize   
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For generating dataset
import sklearn.datasets as dt
```

我们还需要以下常量来检测所有数值接近零的 alpha，因此我们需要定义自己的零阈值。

Python

```py
ZERO = 1e-7
```

### 定义数据点和标签

让我们定义一个非常简单的数据集、相应的标签和一个简单的绘图例程。可选地，如果给定一串 alpha 给绘图函数，它还将标记所有支持向量及其对应的 alpha 值。仅供回顾，支持向量是那些 $\alpha>0$ 的点。

Python

```py
dat = np.array([[0, 3], [-1, 0], [1, 2], [2, 1], [3,3], [0, 0], [-1, -1], [-3, 1], [3, 1]])
labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])

def plot_x(x, t, alpha=[], C=0):
    sns.scatterplot(dat[:,0], dat[:, 1], style=labels,
    hue=labels, markers=['s', 'P'],
    palette=['magenta', 'green'])
    if len(alpha) > 0:
        alpha_str = np.char.mod('%.1f', np.round(alpha, 1))
        ind_sv = np.where(alpha > ZERO)[0]
        for i in ind_sv:   
            plt.gca().text(dat[i,0], dat[i, 1]-.25, alpha_str[i] )

plot_x(dat, labels)
```

![](https://machinelearningmastery.com/wp-content/uploads/2021/12/svm1.png)

## `minimize()` 函数

让我们来看看 `scipy.optimize` 库中的 `minimize()` 函数。它需要以下参数：

+   需要最小化的目标函数。在我们的情况下是拉格朗日对偶。

+   变量的初始值，关于这些变量进行最小化。在这个问题中，我们需要确定拉格朗日乘子 $\alpha$。我们将随机初始化所有 $\alpha$。

+   用于优化的方法。我们将使用 `trust-constr`。

+   对 $\alpha$ 的线性约束。

+   $\alpha$ 的边界。

### 定义目标函数

我们的目标函数是上述定义的 $L_d$，需要最大化。由于我们使用 `minimize()` 函数，所以需要将 $L_d$ 乘以 (-1) 来进行最大化。其实现如下。目标函数的第一个参数是优化时的变量。我们还需要训练点和相应的标签作为额外参数。

你可以通过使用矩阵来简化下面的 `lagrange_dual()` 函数的代码。然而，在本教程中，它保持非常简单，以使一切更加清晰。

Python

```py
# Objective function
def lagrange_dual(alpha, x, t):
    result = 0
    ind_sv = np.where(alpha > ZERO)[0]
    for i in ind_sv:
        for k in ind_sv:
            result = result + alpha[i]*alpha[k]*t[i]*t[k]*np.dot(x[i, :], x[k, :]) 
    result = 0.5*result - sum(alpha)     
    return result
```

### 定义线性约束

对于每个点的 alpha 的线性约束为：

$$

\sum_i \alpha_i t_i = 0

$$

我们也可以写成：

$$

\alpha_0 t_0 + \alpha_1 t_1 + \ldots \alpha_m t_m = 0

$$

`LinearConstraint()` 方法要求所有约束以矩阵形式书写，即：

\begin{equation}

0 =

\begin{bmatrix}

t_0 & t_1 & \ldots t_m

\end{bmatrix}

\begin{bmatrix}

\alpha_0\\ \alpha_1 \\ \vdots \\ \alpha_m

\end{bmatrix}

= 0

\end{equation}

第一个矩阵是 `LinearConstraint()` 方法中的第一个参数。左边界和右边界是第二个和第三个参数。

Python

```py
linear_constraint = LinearConstraint(labels, [0], [0])
print(linear_constraint)
```

输出

```py
<scipy.optimize._constraints.LinearConstraint object at 0x12c87f5b0>
```

### 定义边界

alpha 的边界通过 `Bounds()` 方法定义。所有 alpha 都被限制在 0 和 $C$ 之间。以下是 $C=10$ 的示例。

Python

```py
bounds_alpha = Bounds(np.zeros(dat.shape[0]), np.full(dat.shape[0], 10))
print(bounds_alpha)
```

输出

```py
Bounds(array([0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([10, 10, 10, 10, 10, 10, 10, 10, 10]))
```

### 定义查找 Alphas 的函数

让我们编写一个例程来寻找给定参数 `x`、`t` 和 `C` 时的最优 `alpha` 值。目标函数需要额外的参数 `x` 和 `t`，这些参数通过 `minimize()` 的 args 传递。

Python

```py
def optimize_alpha(x, t, C):
    m, n = x.shape
    np.random.seed(1)
    # Initialize alphas to random values
    alpha_0 = np.random.rand(m)*C
    # Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    # Define the bounds
    bounds_alpha = Bounds(np.zeros(m), np.full(m, C))
    # Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args = (x, t), method='trust-constr', 
                      hess=BFGS(), constraints=[linear_constraint],
                      bounds=bounds_alpha)
    # The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha
```

### 确定超平面

超平面的表达式为：

$$

w^T x + w_0 = 0

$$

对于超平面，我们需要权重向量 $w$ 和常数 $w_0$。权重向量由以下公式给出：

$$

w = \sum_i \alpha_i t_i x_i

$$

如果训练点过多，最好只使用 $\alpha>0$ 的支持向量来计算权重向量。

对于 $w_0$，我们将从每个支持向量 $s$ 计算，对于这些支持向量 $\alpha_s < C$，然后取其平均值。对于单个支持向量 $x_s$，$w_0$ 由以下公式给出：

$$

w_0 = t_s – w^T x_s

$$

支持向量的 alpha 不能在数值上完全等于 C。因此，我们可以从 C 中减去一个小常数，以找到所有 $\alpha_s < C$ 的支持向量。这在 `get_w0()` 函数中完成。

Python

```py
def get_w(alpha, t, x):
    m = len(x)
    # Get all support vectors
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alpha[i]*t[i]*x[i, :]        
    return w

def get_w0(alpha, t, x, w, C):
    C_numeric = C-ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO)&(alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + t[s] - np.dot(x[s, :], w)
    # Take the average    
    w0 = w0 / len(ind_sv)
    return w0
```

### 分类测试点

要对测试点 $x_{test}$ 进行分类，我们使用 $y(x_{test})$ 的符号，如下：

$$

\text{label}_{x_{test}} = \text{sign}(y(x_{test})) = \text{sign}(w^T x_{test} + w_0)

$$

让我们编写相应的函数，可以将测试点的数组与 $w$ 和 $w_0$ 作为参数传入，并对各种点进行分类。我们还添加了第二个函数来计算错误分类率：

Python

```py
def classify_points(x_test, w, w0):
    # get y(x_test)
    predicted_labels = np.sum(x_test*w, axis=1) + w0
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels==0] = 1
    return predicted_labels

def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors/total*100
```

### 绘制边界和超平面

我们还将定义绘制超平面和软边界的函数。

Python

```py
def plot_hyperplane(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    y_coord = -w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, y_coord, color='red')

def plot_margin(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    ypos_coord = 1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, ypos_coord, '--', color='green') 
    yneg_coord = -1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, yneg_coord, '--', color='magenta')
```

## 强化 SVM

现在是运行 SVM 的时候了。`display_SVM_result()` 函数将帮助我们可视化一切。我们将 alpha 初始化为随机值，定义 C，并在此函数中找到最佳的 alpha 值。我们还将绘制超平面、边界和数据点。支持向量也将通过其对应的 alpha 值进行标记。图的标题将是错误的百分比和支持向量的数量。

Python

```py
def display_SVM_result(x, t, C):
    # Get the alphas
    alpha = optimize_alpha(x, t, C)   
    # Get the weights
    w = get_w(alpha, t, x)
    w0 = get_w0(alpha, t, x, w, C)
    plot_x(x, t, alpha, C)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plot_hyperplane(w, w0)
    plot_margin(w, w0)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Get the misclassification error and display it as title
    predictions = classify_points(x, w, w0)
    err = misclassification_rate(t, predictions)
    title = 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%'
    title = title + ',  total SV = ' + str(len(alpha[alpha > ZERO]))
    plt.title(title)

display_SVM_result(dat, labels, 100)    
plt.show()
```

![](https://machinelearningmastery.com/wp-content/uploads/2021/12/svm2.png)

## `C` 的影响

如果你将 `C` 的值更改为 $\infty$，那么软边界将变成硬边界，不容忍错误。在这种情况下，我们定义的问题是不可解的。让我们生成一组人工点并观察 `C` 对分类的影响。为了理解整个问题，我们将使用一个简单的数据集，其中正例和负例是可分的。

以下是通过 `make_blobs()` 生成的点：

Python

```py
dat, labels = dt.make_blobs(n_samples=[20,20],
                           cluster_std=1,
                           random_state=0)
labels[labels==0] = -1
plot_x(dat, labels)
```

![](https://machinelearningmastery.com/wp-content/uploads/2021/12/svm3.png)

现在我们定义不同的 C 值并运行代码。

Python

```py
fig = plt.figure(figsize=(8,25))

i=0
C_array = [1e-2, 100, 1e5]

for C in C_array:
    fig.add_subplot(311+i)    
    display_SVM_result(dat, labels, C)  
    i = i + 1
```

### ![](https://machinelearningmastery.com/wp-content/uploads/2021/12/svm4.png)对结果的评论

上述例子很好地展示了当 $C$ 增加时，边界变小。较高的 $C$ 值对错误施加了更严格的惩罚。较小的值允许更宽的边界和更多的错误分类。因此，$C$ 定义了边界最大化和分类错误之间的权衡。

## 整合代码

这是整合后的代码，你可以将其粘贴到你的 Python 文件中并在本地运行。你可以尝试不同的 $C$ 值，并尝试 `minimize()` 函数中作为参数给出的不同优化方法。

Python

```py
import numpy as np    
# For optimization
from scipy.optimize import Bounds, BFGS                     
from scipy.optimize import LinearConstraint, minimize   
# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
# For generating dataset
import sklearn.datasets as dt

ZERO = 1e-7

def plot_x(x, t, alpha=[], C=0):
    sns.scatterplot(dat[:,0], dat[:, 1], style=labels,
    hue=labels, markers=['s', 'P'],
    palette=['magenta', 'green'])
    if len(alpha) > 0:
        alpha_str = np.char.mod('%.1f', np.round(alpha, 1))
        ind_sv = np.where(alpha > ZERO)[0]
        for i in ind_sv:   
            plt.gca().text(dat[i,0], dat[i, 1]-.25, alpha_str[i] )

# Objective function
def lagrange_dual(alpha, x, t):
    result = 0
    ind_sv = np.where(alpha > ZERO)[0]
    for i in ind_sv:
        for k in ind_sv:
            result = result + alpha[i]*alpha[k]*t[i]*t[k]*np.dot(x[i, :], x[k, :]) 
    result = 0.5*result - sum(alpha)     
    return result 

def optimize_alpha(x, t, C):
    m, n = x.shape
    np.random.seed(1)
    # Initialize alphas to random values
    alpha_0 = np.random.rand(m)*C
    # Define the constraint
    linear_constraint = LinearConstraint(t, [0], [0])
    # Define the bounds
    bounds_alpha = Bounds(np.zeros(m), np.full(m, C))
    # Find the optimal value of alpha
    result = minimize(lagrange_dual, alpha_0, args = (x, t), method='trust-constr', 
                      hess=BFGS(), constraints=[linear_constraint],
                      bounds=bounds_alpha)
    # The optimized value of alpha lies in result.x
    alpha = result.x
    return alpha

def get_w(alpha, t, x):
    m = len(x)
    # Get all support vectors
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alpha[i]*t[i]*x[i, :]        
    return w

def get_w0(alpha, t, x, w, C):
    C_numeric = C-ZERO
    # Indices of support vectors with alpha<C
    ind_sv = np.where((alpha > ZERO)&(alpha < C_numeric))[0]
    w0 = 0.0
    for s in ind_sv:
        w0 = w0 + t[s] - np.dot(x[s, :], w)
    # Take the average    
    w0 = w0 / len(ind_sv)
    return w0

def classify_points(x_test, w, w0):
    # get y(x_test)
    predicted_labels = np.sum(x_test*w, axis=1) + w0
    predicted_labels = np.sign(predicted_labels)
    # Assign a label arbitrarily a +1 if it is zero
    predicted_labels[predicted_labels==0] = 1
    return predicted_labels

def misclassification_rate(labels, predictions):
    total = len(labels)
    errors = sum(labels != predictions)
    return errors/total*100

def plot_hyperplane(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    y_coord = -w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, y_coord, color='red')

def plot_margin(w, w0):
    x_coord = np.array(plt.gca().get_xlim())
    ypos_coord = 1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, ypos_coord, '--', color='green') 
    yneg_coord = -1/w[1] - w0/w[1] - w[0]/w[1] * x_coord
    plt.plot(x_coord, yneg_coord, '--', color='magenta')  

def display_SVM_result(x, t, C):
    # Get the alphas
    alpha = optimize_alpha(x, t, C)   
    # Get the weights
    w = get_w(alpha, t, x)
    w0 = get_w0(alpha, t, x, w, C)
    plot_x(x, t, alpha, C)
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plot_hyperplane(w, w0)
    plot_margin(w, w0)
    plt.xlim(xlim)
    plt.ylim(ylim)
    # Get the misclassification error and display it as title
    predictions = classify_points(x, w, w0)
    err = misclassification_rate(t, predictions)
    title = 'C = ' + str(C) + ',  Errors: ' + '{:.1f}'.format(err) + '%'
    title = title + ',  total SV = ' + str(len(alpha[alpha > ZERO]))
    plt.title(title)

dat = np.array([[0, 3], [-1, 0], [1, 2], [2, 1], [3,3], [0, 0], [-1, -1], [-3, 1], [3, 1]])
labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1])                  
plot_x(dat, labels)
plt.show()
display_SVM_result(dat, labels, 100)    
plt.show()

dat, labels = dt.make_blobs(n_samples=[20,20],
                           cluster_std=1,
                           random_state=0)
labels[labels==0] = -1
plot_x(dat, labels)

fig = plt.figure(figsize=(8,25))

i=0
C_array = [1e-2, 100, 1e5]

for C in C_array:
    fig.add_subplot(311+i)    
    display_SVM_result(dat, labels, C)  
    i = i + 1
```

## 进一步阅读

本节提供了更多关于该主题的资源，如果你想深入了解，可以参考这些资源。

### 书籍

+   [模式识别与机器学习](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) 由 Christopher M. Bishop 著

### 文章

+   [机器学习中的支持向量机](https://machinelearningmastery.com/support-vector-machines-for-machine-learning/)

+   [支持向量机模式识别教程](https://www.di.ens.fr/~mallat/papiers/svmtutorial.pdf) 由 Christopher J.C. Burges 著

### API 参考

+   [SciPy 的优化库](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html)

+   [Scikit-learn 的样本生成库 (sklearn.datasets)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)

+   [NumPy 随机数生成器](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)

## 总结

在本教程中，你学会了如何从头开始实现 SVM 分类器。

具体来说，你学到了：

+   如何为 SVM 优化问题编写目标函数和约束

+   如何编写代码从拉格朗日乘子确定超平面

+   C 对确定边界的影响

你对本文讨论的支持向量机（SVM）有任何问题吗？在下面的评论中提问，我会尽力回答。
