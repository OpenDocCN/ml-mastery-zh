# 从数据到地图：使用 Python 可视化 Ames 房价

> 原文：[`machinelearningmastery.com/data-to-map-geospatial/`](https://machinelearningmastery.com/data-to-map-geospatial/)

地理空间可视化已成为理解和表示地理上下文中数据的必要工具。它在各种实际应用中发挥着重要作用，从城市规划和环境研究到房地产和交通。例如，城市规划者可能会使用地理空间数据来优化公共交通路线，而房地产专业人士则可以利用这些数据分析特定区域的房产价值趋势。使用 Python，你可以利用像 geopandas、Matplotlib 和 contextily 这样的库来创建引人注目的可视化。在本章中，你将深入了解一个可视化 Iowa Ames 房价的代码片段，逐步拆解每一步以理解其目的和功能。

让我们开始吧。

![](img/12266ea57807a05f0f5ec96cce79477f.png)

从数据到地图：使用 Python 可视化 Ames 房价

图片来源于[Annie Spratt](https://unsplash.com/photos/white-and-green-state-maps-AFB6S2kibuk)。保留部分权利。

## 概述

本文分为六部分；它们是：

+   安装必需的 Python 包

+   导入必要的库

+   加载和准备数据

+   设置坐标参考系统（CRS）

+   创建凸包

+   可视化数据

## 安装必需的 Python 包

在我们深入探索 Python 的地理空间可视化之前，正确设置你的开发环境至关重要。在 Windows 上，你可以打开命令提示符或 PowerShell。如果你使用的是 macOS 或 Linux，终端应用程序是你进入命令行世界的入口。此外，为了确保你能够访问所有必需的 Python 库，必须访问 Python 包索引（PyPI），这是 Python 包的官方第三方软件库。

要安装必需的软件包，你可以在终端或命令行界面中使用以下命令：

```py
pip install pandas
pip install geopandas
pip install matplotlib
pip install contextily
pip install shapely
```

一旦你成功安装了所需的软件包，你就可以导入必要的库，并开始你的地理空间可视化之旅。

**启动你的项目**，请参考我的书籍 [《数据科学入门指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/)。它提供了**自学教程**和**可运行的代码**。

## 导入必要的库

在深入可视化之前，导入将支持我们可视化的必要库是至关重要的。

```py
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
```

我们将使用几个 Python 库，包括：

+   **pandas：** 用于数据操作和分析。

+   **geopandas：** 用于处理地理空间数据。

+   **matplotlib：** 用于创建静态、动画和交互式可视化。

+   **contextily：** 用于将底图添加到我们的图表中。

+   **shapely：** 用于处理和分析平面几何对象。

## 加载和准备数据

[`Ames.csv` 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv) 包含有关 Iowa, Ames 房屋销售的详细信息。这包括房屋的各种属性，如大小、年龄和状况，以及其地理坐标（纬度和经度）。这些地理坐标对我们的地理空间可视化至关重要，因为它们使我们能够在地图上绘制每个房屋，为销售价格提供空间背景。

```py
# Load the dataset
Ames = pd.read_csv('Ames.csv')

# Convert the DataFrame to a GeoDataFrame
geometry = [Point(xy) for xy in zip(Ames['Longitude'], Ames['Latitude'])]
geo_df = gpd.GeoDataFrame(Ames, geometry=geometry)
```

通过将 pandas DataFrame 转换为 GeoDataFrame，我们可以在数据集上利用地理空间功能，将原始数据转化为适合地理空间分析和可视化的格式。

## 设置坐标参考系统（CRS）

坐标参考系统（CRS）是准确的地理空间操作和制图的基本方面，决定了我们的数据在地球表面上的对齐方式。不同 CRS 下两点之间的距离会有所不同，地图的显示也会有所不同。在我们的示例中，我们使用“EPSG:4326”这一标注设置了 GeoDataFrame 的 CRS，该标注对应于广泛使用的 WGS 84（或世界大地测量系统 1984）纬度-经度坐标系统。

```py
# Set the CRS for the GeoDataFrame
geo_df.crs = "EPSG:4326"
```

WGS 84 是一个于 1984 年建立的全球参考系统，是卫星定位、GPS 和各种映射应用的事实标准。它使用三维坐标系统，以纬度和经度定义地球表面的位置信息，高度表示相对于**参考椭球体**的高度。

除了 WGS 84 之外，还有许多坐标参考系统满足不同的映射需求。选择包括提供适用于区域映射的平面笛卡尔坐标的通用横轴墨卡托（UTM），例如用于基于网页的映射的欧洲石油勘测组（EPSG）选项，如“EPSG:3857”，以及提供美国内州特定系统的州平面坐标系统（SPCS）。选择合适的 CRS 取决于规模、精度和数据的地理范围等因素，以确保地理空间分析和可视化的精确性。

## 创建凸包

凸包提供了一个包围所有数据点的边界，提供了我们数据地理分布的视觉表示。

```py
# Create a convex hull around the points
convex_hull = geo_df.unary_union.convex_hull
convex_hull_geo = gpd.GeoSeries(convex_hull, crs="EPSG:4326")
convex_hull_transformed = convex_hull_geo.to_crs(epsg=3857)
buffered_hull = convex_hull_transformed.buffer(500)
```

从“EPSG:4326”到“EPSG:3857”的转换至关重要，原因有几个：

+   **基于网页的可视化：** “EPSG:3857”被优化用于像 Google Maps 和 OpenStreetMap 这样的网页映射应用程序。通过将数据转换为该 CRS，我们确保它可以正确叠加在基于网页的底图上。

+   **以米为单位的缓冲：** 缓冲操作在凸包周围添加边距。在“EPSG:4326”中，坐标是以度为单位的，这使得以米为单位的缓冲变得困难。通过转换到“EPSG:3857”，我们可以准确地将凸包缓冲 500 米，为 Ames 提供清晰的边界。

通过缓冲凸包，我们不仅可视化了数据的分布，还为可视化提供了地理背景，突出显示了关注区域。

### 想要开始学习数据科学的初学者指南吗？

立即参加我的免费邮件速成课程（包括示例代码）。

点击注册并获得课程的免费 PDF 电子书版本。

## 数据可视化

数据准备好后，是时候通过可视化来展现它。我们将在地图上绘制单个房屋的销售价格，使用颜色梯度表示不同的价格范围。

```py
# Plotting the map with Sale Prices, a basemap, and the buffered convex hull as a border
fig, ax = plt.subplots(figsize=(12, 8))
geo_df.to_crs(epsg=3857).plot(column='SalePrice', cmap='coolwarm', ax=ax, legend=True, 
                              markersize=20)
buffered_hull.boundary.plot(ax=ax, color='black', label='Buffered Boundary of Ames')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
ax.set_axis_off()
ax.legend(loc='upper right')
colorbar = ax.get_figure().get_axes()[1]
colorbar.set_ylabel('Sale Price', rotation=270, labelpad=20, fontsize=15)
plt.title('Sale Prices of Individual Houses in Ames, Iowa with Buffered Boundary', 
          fontsize=18)
plt.show()
```

使用的颜色梯度，**‘coolwarm’**，是一个发散的颜色映射。这意味着它有两种不同的颜色代表光谱的两个端点，中间有一个中性色。在我们的可视化中：

+   **较冷的颜色（蓝色）** 代表房价较低的房屋。

+   **较暖的颜色（红色）** 表示房价较高的房屋。

这种颜色映射选择使读者能够快速识别房产价值高低的区域，提供了对 Ames 房屋销售价格分布的见解。缓冲边界进一步强调了关注区域，为可视化提供了背景。

![](https://machinelearningmastery.com/wp-content/uploads/2024/01/Ames_Map.png)

这张地图是几个组件的组合：由 contextily 从 OpenStreetMap 导入的基图展示了特定经纬度的地形。彩色点基于来自 pandas DataFrame 的数据，但通过 geopandas 转换为地理 CRS，应该与基图对齐。

## **进一步阅读**

本节提供了更多相关资源，如果你想深入了解。

#### 教程

+   [GeoPandas 介绍](https://geopandas.org/en/stable/getting_started/introduction.html)

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据词典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这篇文章中，我们深入探讨了使用 Python 进行地理空间可视化的复杂性，重点关注了爱荷华州 Ames 房屋销售价格的可视化。通过对代码的详细逐步解析，我们揭示了从初始数据加载和准备到最终可视化的各个阶段。理解地理空间可视化技术不仅仅是学术上的练习；它具有深远的现实世界意义。掌握这些技术可以使各领域的专业人士，如城市规划和房地产领域的从业者，能够做出基于地理背景的明智数据驱动决策。随着城市的增长和世界变得越来越数据驱动，将数据叠加到地理地图上将对制定未来策略和洞察力至关重要。

具体来说，通过本教程，你学到了：

+   如何利用关键 Python 库进行地理空间可视化。

+   数据准备和转换在地理空间操作中的关键作用。

+   可视化地理空间数据的有效技巧，包括设置颜色渐变和整合底图的细节。

你有任何问题吗？请在下面的评论中提问，我会尽力回答。
