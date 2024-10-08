# NumPy 和 SciPy 中的科学函数

> 原文：[`machinelearningmastery.com/scientific-functions-in-numpy-and-scipy/`](https://machinelearningmastery.com/scientific-functions-in-numpy-and-scipy/)

Python 是一个通用计算语言，但在科学计算中非常受欢迎。由于 Python 生态系统中的一些库，它在许多情况下可以替代 R 和 Matlab。在机器学习中，我们广泛使用一些数学或统计函数，并且我们经常发现 NumPy 和 SciPy 非常有用。接下来，我们将简要概述 NumPy 和 SciPy 提供了什么以及一些使用技巧。

完成本教程后，你将了解到：

+   NumPy 和 SciPy 为你的项目提供了什么

+   如何使用 numba 快速加速 NumPy 代码

**通过我的新书[《机器学习的 Python》](https://machinelearningmastery.com/python-for-machine-learning/)**，**快速启动你的项目**，包括*一步步的教程*和所有示例的*Python 源代码*文件。

让我们开始吧！！[](../Images/8a66d78bdafab3162c4d63b3e4be46ec.png)

NumPy 和 SciPy 中的科学函数

照片由[Nothing Ahead](https://www.pexels.com/photo/magnifying-glass-on-textbook-4494641/)提供。保留所有权利。

## 概述

本教程分为三个部分：

+   NumPy 作为张量库

+   SciPy 的函数

+   使用 numba 加速

## NumPy 作为张量库

虽然 Python 中的列表和元组是我们本地管理数组的方式，但 NumPy 提供了更接近 C 或 Java 的数组功能，意味着我们可以强制所有元素为相同的数据类型，并且在高维数组的情况下，每个维度中的形状都是规则的。此外，在 NumPy 数组中执行相同的操作通常比在 Python 本地中更快，因为 NumPy 中的代码经过高度优化。

NumPy 提供了上千个函数，你应该查阅 NumPy 的文档以获取详细信息。以下备忘单中可以找到一些常见用法：

![](https://machinelearningmastery.com/wp-content/uploads/2022/04/cheatsheet.png)

NumPy 备忘单。版权所有 2022 MachineLearningMastery.com

NumPy 中有一些很酷的功能值得一提，因为它们对机器学习项目很有帮助。

例如，如果我们想绘制一个 3D 曲线，我们会计算$x$和$y$范围内的$z=f(x,y)$，然后在$xyz$空间中绘制结果。我们可以用以下方法生成范围：

```py
import numpy as np
x = np.linspace(-1, 1, 100)
y = np.linspace(-2, 2, 100)
```

对于$z=f(x,y)=\sqrt{1-x²-(y/2)²}$，我们可能需要一个嵌套的 for 循环来扫描数组`x`和`y`中的每个值并进行计算。但在 NumPy 中，我们可以使用`meshgrid`将两个 1D 数组扩展为两个 2D 数组，通过匹配索引，我们得到所有组合，如下所示：

```py
import matplotlib.pyplot as plt 
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.linspace(-2, 2, 100)

# convert vector into 2D arrays
xx, yy = np.meshgrid(x,y)
# computation on matching
z = np.sqrt(1 - xx**2 - (yy/2)**2)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([0,2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
plt.show()
```

![](img/63be7afc5683451c422e60eb0b8d1e70.png)

上述中，由 `meshgrid()` 生成的 2D 数组 `xx` 在同一列上具有相同的值，`yy` 在同一行上具有相同的值。因此，`xx` 和 `yy` 上的逐元素操作实际上是对 $xy$ 平面的操作。这就是为什么它有效，以及我们如何绘制上面的椭球体。

NumPy 另一个很好的功能是扩展维度的函数。神经网络中的卷积层通常期望 3D 图像，即 2D 像素和作为第三维度的不同颜色通道。它适用于使用 RGB 通道的彩色图像，但在灰度图像中我们只有一个通道。例如，scikit-learn 中的数字数据集：

```py
from sklearn.datasets import load_digits
images = load_digits()["images"]
print(images.shape)
```

```py
(1797, 8, 8)
```

这表明这个数据集有 1797 张图像，每张图像为 8×8 像素。这是一个灰度数据集，显示每个像素的黑暗值。我们将第四轴添加到这个数组中（即，将一个 3D 数组转换为 4D 数组），以便每张图像为 8x8x1 像素：

```py
...

# image has axes 0, 1, and 2, adding axis 3
images = np.expand_dims(images, 3)
print(images.shape)
```

```py
(1797, 8, 8, 1)
```

在处理 NumPy 数组时，一个方便的特性是布尔索引和花式索引。例如，如果我们有一个 2D 数组：

```py
import numpy as np

X = np.array([
    [ 1.299,  0.332,  0.594, -0.047,  0.834],
    [ 0.842,  0.441, -0.705, -1.086, -0.252],
    [ 0.785,  0.478, -0.665, -0.532, -0.673],
    [ 0.062,  1.228, -0.333,  0.867,  0.371]
])
```

我们可以检查列中的所有值是否都是正值：

```py
...
y = (X > 0).all(axis=0)
print(y)
```

```py
array([ True,  True, False, False, False])
```

这仅显示前两列都是正值。注意这是一个长度为 5 的一维数组，其大小与数组 `X` 的轴 1 相同。如果我们在轴 1 上使用这个布尔数组作为索引，我们只选择索引为正的子数组：

```py
...
y = X[:, (X > 0).all(axis=0)
print(y)
```

```py
array([[1.299, 0.332],
       [0.842, 0.441],
       [0.785, 0.478],
       [0.062, 1.228]])
```

如果用整数列表代替上述布尔数组，我们根据与列表匹配的索引从 `X` 中选择。NumPy 称之为花式索引。如下所示，我们可以选择前两列两次并形成一个新数组：

```py
...
y = X[:, [0,1,1,0]]
print(y)
```

```py
array([[1.299, 0.332, 0.332, 1.299],
       [0.842, 0.441, 0.441, 0.842],
       [0.785, 0.478, 0.478, 0.785],
       [0.062, 1.228, 1.228, 0.062]])
```

## SciPy 的函数

SciPy 是 NumPy 的姊妹项目。因此，你通常会看到 SciPy 函数期望 NumPy 数组作为参数或返回一个。SciPy 提供了许多不常用或更高级的函数。

SciPy 函数被组织在子模块下。一些常见的子模块包括：

+   `scipy.cluster.hierarchy`: 层次聚类

+   `scipy.fft`: 快速傅里叶变换

+   `scipy.integrate`: 数值积分

+   `scipy.interpolate`: 插值和样条函数

+   `scipy.linalg`: 线性代数

+   `scipy.optimize`: 数值优化

+   `scipy.signal`: 信号处理

+   `scipy.sparse`: 稀疏矩阵表示

+   `scipy.special`: 一些特殊的数学函数

+   `scipy.stats`: 统计，包括概率分布

但不要假设 SciPy 可以覆盖所有内容。例如，对于时间序列分析，最好依赖于 `statsmodels` 模块。

我们在其他文章中已经讨论了许多使用 `scipy.optimize` 的例子。它是一个很棒的工具，可以使用例如牛顿法来找到函数的最小值。NumPy 和 SciPy 都有 `linalg` 子模块用于线性代数，但 SciPy 中的函数更高级，如进行 QR 分解或矩阵指数的函数。

也许 SciPy 最常用的功能是 `stats` 模块。在 NumPy 和 SciPy 中，我们可以生成具有非零相关性的多变量高斯随机数。

```py
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mean = [0, 0]             # zero mean
cov = [[1, 0.8],[0.8, 1]] # covariance matrix
X1 = np.random.default_rng().multivariate_normal(mean, cov, 5000)
X2 = multivariate_normal.rvs(mean, cov, 5000)

fig = plt.figure(figsize=(12,6))
ax = plt.subplot(121)
ax.scatter(X1[:,0], X1[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("NumPy")

ax = plt.subplot(122)
ax.scatter(X2[:,0], X2[:,1], s=1)
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])
ax.set_title("SciPy")

plt.show()
```

![](img/28987164ebb562fe0132bb780ba659ca.png)

但如果我们想要引用分布函数本身，最好依赖于 SciPy。例如，著名的 68-95-99.7 规则是指标准正态分布，我们可以通过 SciPy 的累积分布函数获取确切的百分比：

```py
from scipy.stats import norm
n = norm.cdf([1,2,3,-1,-2,-3])
print(n)
print(n[:3] - n[-3:])
```

```py
[0.84134475 0.97724987 0.9986501  0.15865525 0.02275013 0.0013499 ]
[0.68268949 0.95449974 0.9973002 ]
```

因此我们看到，在正态分布中，我们期望 68.269%的概率值会落在均值的一个标准差范围内。相反，我们有百分位点函数作为累积分布函数的逆函数：

```py
...
print(norm.ppf(0.99))
```

```py
2.3263478740408408
```

这意味着，如果值服从正态分布，我们期望有 99%的概率（单尾概率）值不会超过均值的 2.32 倍标准差。

这些都是 SciPy 如何超越 NumPy 的例子。

### 想要开始使用 Python 进行机器学习吗？

立即参加我的免费 7 天邮件速成课程（包括示例代码）。

点击注册，还可以获得课程的免费 PDF 电子书版本。

## 使用 numba 提速

NumPy 比原生 Python 更快，因为许多操作是用 C 实现的并使用了优化的算法。但有时我们希望做一些事情时，NumPy 仍然太慢。

如果你有 GPU，你可以要求 `numba` 进一步优化，通过并行化或将操作移到 GPU 上来加速。你需要先安装 `numba` 模块：

```py
pip install numba
```

如果你需要将 `numba` 编译成 Python 模块，可能需要一段时间。之后，如果你有一个纯 NumPy 操作的函数，你可以添加 `numba` 装饰器来加速它：

```py
import numba

@numba.jit(nopython=True)
def numpy_only_function(...)
    ...
```

它的作用是使用即时编译器来向量化操作，以便它可以更快地运行。如果你的函数在程序中运行很多次（例如，梯度下降中的更新函数），你可以看到最佳的性能提升，因为编译器的开销可以得到摊销。

例如，下面是一个将 784 维数据转换为 2 维的 t-SNE 算法实现。我们不会详细解释 t-SNE 算法，但它需要很多次迭代才能收敛。以下代码展示了我们如何使用 `numba` 来优化内部循环函数（同时也演示了一些 NumPy 的用法）。完成需要几分钟时间。你可以尝试在之后移除 `@numba.jit` 装饰器。这将需要相当长的时间。

```py
import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import numba

def tSNE(X, ndims=2, perplexity=30, seed=0, max_iter=500, stop_lying_iter=100, mom_switch_iter=400):
    """The t-SNE algorithm

	Args:
		X: the high-dimensional coordinates
		ndims: number of dimensions in output domain
    Returns:
        Points of X in low dimension
    """
    momentum = 0.5
    final_momentum = 0.8
    eta = 200.0
    N, _D = X.shape
    np.random.seed(seed)

    # normalize input
    X -= X.mean(axis=0) # zero mean
    X /= np.abs(X).max() # min-max scaled

    # compute input similarity for exact t-SNE
    P = computeGaussianPerplexity(X, perplexity)
    # symmetrize and normalize input similarities
    P = P + P.T
    P /= P.sum()
    # lie about the P-values
    P *= 12.0
    # initialize solution
    Y = np.random.randn(N, ndims) * 0.0001
    # perform main training loop
    gains = np.ones_like(Y)
    uY = np.zeros_like(Y)
    for i in range(max_iter):
        # compute gradient, update gains
        dY = computeExactGradient(P, Y)
        gains = np.where(np.sign(dY) != np.sign(uY), gains+0.2, gains*0.8).clip(0.1)
        # gradient update with momentum and gains
        uY = momentum * uY - eta * gains * dY
        Y = Y + uY
        # make the solution zero-mean
        Y -= Y.mean(axis=0)
        # Stop lying about the P-values after a while, and switch momentum
        if i == stop_lying_iter:
            P /= 12.0
        if i == mom_switch_iter:
            momentum = final_momentum
        # print progress
        if (i % 50) == 0:
            C = evaluateError(P, Y)
            now = datetime.datetime.now()
            print(f"{now} - Iteration {i}: Error = {C}")
    return Y

@numba.jit(nopython=True)
def computeExactGradient(P, Y):
    """Gradient of t-SNE cost function

	Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        dY, a numpy array of shape (N,D)
	"""
    N, _D = Y.shape
    # compute squared Euclidean distance matrix of Y, the Q matrix, and the normalization sum
    DD = computeSquaredEuclideanDistance(Y)
    Q = 1/(1+DD)
    sum_Q = Q.sum()
    # compute gradient
    mult = (P - (Q/sum_Q)) * Q
    dY = np.zeros_like(Y)
    for n in range(N):
        for m in range(N):
            if n==m: continue
            dY[n] += (Y[n] - Y[m]) * mult[n,m]
    return dY

@numba.jit(nopython=True)
def evaluateError(P, Y):
    """Evaluate t-SNE cost function

    Args:
        P: similarity matrix
        Y: low-dimensional coordinates
    Returns:
        Total t-SNE error C
    """
    DD = computeSquaredEuclideanDistance(Y)
    # Compute Q-matrix and normalization sum
    Q = 1/(1+DD)
    np.fill_diagonal(Q, np.finfo(np.float32).eps)
    Q /= Q.sum()
    # Sum t-SNE error: sum P log(P/Q)
    error = P * np.log( (P + np.finfo(np.float32).eps) / (Q + np.finfo(np.float32).eps) )
    return error.sum()

@numba.jit(nopython=True)
def computeGaussianPerplexity(X, perplexity):
    """Compute Gaussian Perplexity

    Args:
        X: numpy array of shape (N,D)
        perplexity: double
    Returns:
        Similarity matrix P
    """
    # Compute the squared Euclidean distance matrix
    N, _D = X.shape
    DD = computeSquaredEuclideanDistance(X)
    # Compute the Gaussian kernel row by row
    P = np.zeros_like(DD)
    for n in range(N):
        found = False
        beta = 1.0
        min_beta = -np.inf
        max_beta = np.inf
        tol = 1e-5

        # iterate until we get a good perplexity
        n_iter = 0
        while not found and n_iter < 200:
            # compute Gaussian kernel row
            P[n] = np.exp(-beta * DD[n])
            P[n,n] = np.finfo(np.float32).eps
            # compute entropy of current row
            # Gaussians to be row-normalized to make it a probability
            # then H = sum_i -P[i] log(P[i])
            #        = sum_i -P[i] (-beta * DD[n] - log(sum_P))
            #        = sum_i P[i] * beta * DD[n] + log(sum_P)
            sum_P = P[n].sum()
            H = beta * (DD[n] @ P[n]) / sum_P + np.log(sum_P)
            # Evaluate if entropy within tolerance level
            Hdiff = H - np.log2(perplexity)
            if -tol < Hdiff < tol:
                found = True
                break
            if Hdiff > 0:
                min_beta = beta
                if max_beta in (np.inf, -np.inf):
                    beta *= 2
                else:
                    beta = (beta + max_beta) / 2
            else:
                max_beta = beta
                if min_beta in (np.inf, -np.inf):
                    beta /= 2
                else:
                    beta = (beta + min_beta) / 2
            n_iter += 1
        # normalize this row
        P[n] /= P[n].sum()
    assert not np.isnan(P).any()
    return P

@numba.jit(nopython=True)
def computeSquaredEuclideanDistance(X):
    """Compute squared distance
    Args:
        X: numpy array of shape (N,D)
    Returns:
        numpy array of shape (N,N) of squared distances
    """
    N, _D = X.shape
    DD = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1, N):
            diff = X[i] - X[j]
            DD[j][i] = DD[i][j] = diff @ diff
    return DD

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# pick 1000 samples from the dataset
rows = np.random.choice(X_test.shape[0], 1000, replace=False)
X_data = X_train[rows].reshape(1000, -1).astype("float")
X_label = y_train[rows]
# run t-SNE to transform into 2D and visualize in scatter plot
Y = tSNE(X_data, 2, 30, 0, 500, 100, 400)
plt.figure(figsize=(8,8))
plt.scatter(Y[:,0], Y[:,1], c=X_label)
plt.show()
```

## 进一步阅读

本节提供了更多关于该主题的资源，如果你希望更深入地了解。

#### API 文档

+   [NumPy 用户指南](https://numpy.org/doc/stable/user/index.html#user)

+   [SciPy 用户指南](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide)

+   [Numba 文档](https://numba.pydata.org/numba-doc/dev/index.html)

## **总结**

在本教程中，你简要了解了 NumPy 和 SciPy 提供的函数。

具体来说，你学到了：

+   如何使用 NumPy 数组

+   SciPy 提供的一些帮助函数

+   如何通过使用来自 numba 的 JIT 编译器加快 NumPy 代码的运行速度
