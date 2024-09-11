# 正弦和余弦的导数

> 原文：[`machinelearningmastery.com/derivative-of-the-sine-and-cosine/`](https://machinelearningmastery.com/derivative-of-the-sine-and-cosine/)

许多机器学习算法涉及不同目的的优化过程。优化是指通过改变输入值来最小化或最大化目标函数的问题。

优化算法依赖于导数来理解如何改变（增加或减少）目标函数的输入值，以最小化或最大化目标函数。因此，考虑中的目标函数必须是*可导*的。

两个基本的三角函数，即正弦和余弦，为理解求导的操作提供了良好的机会。这两个函数变得特别重要，如果我们将它们视为更复杂函数的基本构建块。

在本教程中，你将发现如何找到正弦和余弦函数的导数。

完成本教程后，你将了解：

+   如何通过应用代数、三角学和极限中的几个规则来找到正弦和余弦函数的导数。

+   如何在 Python 中求正弦函数和余弦函数的导数。

让我们开始吧。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_cover-scaled.jpg)

正弦和余弦的导数

图片由 [Tim Marshall](https://unsplash.com/photos/9tta3btd8hE) 提供，部分版权保留。

## **教程概述**

本教程分为三部分，它们是：

+   正弦函数的导数

+   余弦函数的导数

+   在 Python 中求导

## **正弦函数的导数**

某个函数 *f* 在特定点 *x* 的导数 *f’*(*x*) 可以定义为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_1.png)

我们将从考虑正弦函数开始。因此，首先将 *f*(*x*) = sin *x* 代入：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_2.png)

如果我们查看三角恒等式，我们发现可以应用*加法公式*来展开 sin(*x* + *h*) 项：

sin(*x* + *y*) = sin *x* cos *y* + cos *x* sin *y*

确实，通过将 *y* 替换为 *h*，我们可以定义 sin *x* 的导数为：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_3.png)

我们可以通过应用限制法则进一步简化表达式，该法则指出函数和的极限等于其极限的和：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_4.png)

我们可以进一步简化，通过提取任何与 *x* 相关的共同因子。这样，我们可以将表达式因式分解以得到两个不依赖于 *x* 的独立极限的和：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_5.png)

求解这两个极限中的每一个将给出*sin x*的导数。

让我们从处理第一个极限开始。

[回顾](https://machinelearningmastery.com/what-you-need-to-know-before-you-get-started-a-brief-tour-of-calculus-pre-requisites/) 我们可以在单位圆上以弧度表示角度 *h*。则 *h* 的正弦值由垂直于 x 轴的 *BC* 表示，该点与单位圆相交：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_diagrams_1.png)

在单位圆上表示角度 *h*

我们将比较不同扇形和三角形的面积，边缘与角*h*相对，以推测当*h*的值接近零时，((sin *h*) / *h*) 的行为。为此，首先考虑扇形 *OAB* 的面积：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_diagrams_2.png)

计算扇形 *OAB* 的面积

扇形的面积可以用圆的半径 *r* 和弧长 *AB*，即 *h*，来定义。由于考虑的圆是 *单位* 圆，因此 *r* = 1：

area_of_sector_OAB = *r h* / 2 = *h* / 2

我们可以将我们刚刚找到的扇形 *OAB* 的面积与同一扇形内的 *三角形 OAB* 的面积进行比较。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_diagrams_3.png)

计算三角形 *OAB* 的面积

这个三角形的面积是以其高度 *BC* = sin *h* 和底边长度 *OA* = 1 为定义的：

area_of_triangle_OAB = (*BC*) (*OA*) / 2 = (sin *h*) / 2

由于我们可以清楚地看到，我们刚才考虑的三角形 *OAB* 的面积小于它所包含的扇形的面积，因此我们可以说：

(sin *h*) / 2 < *h* / 2

(sin *h*) / *h* < 1

这是我们获得的关于((sin *h*) */* *h*)的行为的第一条信息，这告诉我们它的上限值不会超过 1。

现在我们考虑第二个三角形 *OAB'*，其面积大于扇形 *OAB* 的面积。我们可以使用这个三角形来提供关于((sin *h*) */* *h*) 的第二条信息，即它的下限值：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_diagrams_04.png)

比较相似的三角形 *OAB* 和 *OAB’*

应用相似三角形的性质将 *OAB’* 关联到 *OCB*，提供了计算三角形面积所需的长度 *B’A* 的信息：

*B’A* / *OA* = *BC* / *OC* = (sin *h*) / (cos *h*)

因此，三角形 *OAB’* 的面积可以计算为：

area_of_triangle_OAB’ = (*B’A*) (*OA*) / 2 = (sin *h*) / (2 cos *h*)

比较三角形 *OAB’* 和扇形 *OAB* 的面积，我们可以看到前者现在更大：

*h* / 2 < (sin *h*) / (2 cos *h*)

cos *h* < (sin *h*) / *h*

这是我们所需的第二条信息，它告诉我们 ((sin *h*) */* *h*) 的下界值不会低于 cos *h*。我们还知道，随着 *h* 趋近于 0，cos *h* 的值趋近于 1。

因此，将这两条信息结合起来，我们发现随着 *h* 越来越小，((sin *h*) */* *h*) 的值被其上下界限 *挤压* 到 1。这实际上被称为 *挤压* 或 *夹挤* 定理。

现在我们继续解决第二个极限。

通过应用标准的代数规则：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_6.png)

我们可以如下操作第二个极限：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_7.png)

我们可以通过应用三角学中的毕达哥拉斯恒等式 sin² *h* = 1 – cos² *h* 来用正弦表达这个极限：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_8.png)

随后应用另一种极限法则，该法则指出乘积的极限等于各个极限的乘积：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_9.png)

我们已经解决了这个乘积的第一个极限，并发现其值为 1。

这个乘积的第二个极限的特征是分母中有一个 cos *h*，随着 *h* 的减小，cos *h* 接近 1。因此，第二个极限的分母随着 *h* 趋近于 0，接近值 2。另一方面，分子中的正弦项在 *h* 趋近于 0 时达到 0。这不仅使第二个极限，也使整个乘积极限趋向于 0：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_10.png)

综合考虑所有因素，我们最终可以得出以下结论：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_11.png)

sin’(*x*) = (1) (cos *x*) + (0) (sin *x*)

sin’(*x*) = cos *x*

这最终告诉我们 sin *x* 的导数就是 cos *x*。

## **余弦函数的导数**

同样，我们可以通过重新利用找到正弦函数导数的知识来计算余弦函数的导数。将 *f*(*x*) = cos *x* 代入：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_12.png)

现在应用*加法公式*来展开 cos(*x* + *h*) 项，如下所示：

cos(*x* + *y*) = cos *x* cos *y* + sin *x* sin *y*

这又导致了两个极限的求和：

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_13.png)

我们可以很快发现，我们已经在寻找正弦函数导数的过程中评估了这两个极限；第一个极限趋近于 1，而第二个极限趋近于 0，随着 *h* 的值变小：

cos’(*x*) = (1) (-sin *x*) + (0) (cos *x*)

cos’(*x*) = -sin *x*

这最终告诉我们余弦函数的导数恰好是 -sin *x*。

我们刚刚找到的导数的重要性在于它们定义了函数在某个特定角度 *h* 下的*变化率*。例如，如果我们回顾周期性正弦函数的图形，我们可以观察到它的第一个正峰值恰好与 π / 2 弧度的角度重合。

![](https://machinelearningmastery.com/wp-content/uploads/2021/06/derivative_14.png)

周期性正弦函数的线性图

我们可以利用正弦函数的导数来直接计算图表上这个峰值处切线的变化率或斜率：

sin’(π / 2) = cos(π / 2) = 0

我们发现这一结果与正弦函数的峰值确实是一个变化率为零的静止点这一事实相符。

类似的练习可以很容易地进行，以计算不同角度下切线的变化率，适用于正弦和余弦函数。

## **在 Python 中求导**

在这一部分，我们将使用 Python 计算正弦和余弦函数的导数。

为此，我们将利用 SymPy 库，它允许我们以符号形式处理数学对象的计算。这意味着 SymPy 库将使我们能够定义和操作未评估变量的正弦和余弦函数。我们可以通过使用 Python 中的 *symbols* 来定义变量为符号，而求导则使用 *diff* 函数。

在进一步操作之前，我们首先加载所需的库。

```py
from sympy import diff
from sympy import sin
from sympy import cos
from sympy import symbols
```

现在，我们可以定义一个符号形式的变量 *x*，这意味着我们可以使用 *x* 而不需要为其赋值。

```py
# define variable as symbol
x = symbols('x')
```

接下来，我们可以使用`diff`函数找出正弦和余弦函数关于*x*的导数。

```py
# find the first derivative of sine and cosine with respect to x
print('The first derivative of sine is:', diff(sin(x), x))
print('The first derivative of cosine is:', diff(cos(x), x))
```

我们发现`diff`函数正确地返回了*cos*(*x*)作为正弦的导数，以及–*sin*(*x*)作为余弦的导数。

Python

```py
The first derivative of sine is: cos(x)
The first derivative of cosine is: -sin(x)
```

`diff`函数也可以进行多次导数。例如，我们可以通过将*x*传递两次来找出正弦和余弦的第二导数。

```py
# find the second derivative of sine and cosine with respect to x
print('The second derivative of sine is:', diff(sin(x), x, x))
print('The second derivative of cosine is:', diff(cos(x), x, x))
```

这意味着，在寻找第二导数时，我们实际上是在计算每个函数的导数的导数。例如，要找出正弦函数的第二导数，我们需要对*cos*(*x*)，即它的第一次导数，进行导数计算。我们也可以通过类似的方式计算余弦函数的第二导数，即对–*sin*(*x*)，它的第一次导数，进行导数计算。

```py
The second derivative of sine is: -sin(x)
The second derivative of cosine is: -cos(x)
```

我们还可以将数字 2 传递给`diff`函数，以指示我们感兴趣于找到第二导数。

```py
# find the second derivative of sine and cosine with respect to x
print('The second derivative of sine is:', diff(sin(x), x, 2))
print('The second derivative of cosine is:', diff(cos(x), x, 2))
```

将这些内容结合起来，下面列出了寻找正弦和余弦函数导数的完整示例。

```py
# finding the derivative of the sine and cosine functions
from sympy import diff
from sympy import sin
from sympy import cos
from sympy import symbols

# define variable as symbol
x = symbols('x')

# find the first derivative of sine and cosine with respect to x
print('The first derivative of sine is:', diff(sin(x), x))
print('The first derivative of cosine is:', diff(cos(x), x))

# find the second derivative of sine and cosine with respect to x
print('\nThe second derivative of sine is:', diff(sin(x), x, x))
print('The second derivative of cosine is:', diff(cos(x), x, x))

# find the second derivative of sine and cosine with respect to x
print('\nThe second derivative of sine is:', diff(sin(x), x, 2))
print('The second derivative of cosine is:', diff(cos(x), x, 2))
```

## **进一步阅读**

本节提供了更多的资源，如果你希望深入了解这个主题。

### **书籍**

+   [微积分的搭便车指南](https://www.amazon.com/Hitchhikers-Calculus-Classroom-Resource-Materials/dp/1470449625/ref=as_li_ss_tl?dchild=1&keywords=The+Hitchhiker%27s+Guide+to+Calculus&qid=1606170787&sr=8-1&linkCode=sl1&tag=inspiredalgor-20&linkId=f8875fa9736746bf29d78fc0c55553d8&language=en_US)，2019 年。

+   [优化算法](https://www.amazon.com/Algorithms-Optimization-Press-Mykel-Kochenderfer/dp/0262039427/ref=sr_1_1?dchild=1&keywords=algorithms+for+optimization&qid=1624019308&sr=8-1)，2019 年。

## **总结**

在本教程中，你发现了如何找出正弦和余弦函数的导数。

具体来说，你学到了：

+   如何通过应用代数、三角学和极限的一些规则来找出正弦和余弦函数的导数。

+   如何在 Python 中找到正弦和余弦函数的导数。

你有任何问题吗？

在下面的评论中提出你的问题，我会尽力回答。
