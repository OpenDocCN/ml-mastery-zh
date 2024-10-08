# 用数据寻找价值：奢华房地产决策中的凝聚力

> 原文：[`machinelearningmastery.com/finding-value-with-data-the-cohesive-force-behind-luxury-real-estate-decisions/`](https://machinelearningmastery.com/finding-value-with-data-the-cohesive-force-behind-luxury-real-estate-decisions/)

房地产行业是一个庞大的利益相关者网络，包括代理商、房主、投资者、开发商、城市规划师和技术创新者，每个人都带来独特的视角和目标。在这个错综复杂的生态系统中，数据作为将这些不同利益结合在一起的关键元素，促进了合作与创新。PropTech，即房地产技术，通过将信息技术应用于房地产，展示了这种协同作用，利用数据科学的力量改变了物业的研究、购买、销售和管理方式。

从房地产列表的数字化初期到包括虚拟现实、物联网智能家居和区块链增强交易的当前格局，PropTech 的历程反映了一个越来越由数据驱动的行业。这一演变的标志不是技术本身，而是数据科学如何利用信息来简化操作、提升用户体验，并在每一个环节引入效率。

PropTech 的变革性影响核心是数据科学，它擅长从庞大的数据集中提取可操作的见解。它为房地产领域的所有参与者——从优化项目位置的开发商到塑造可持续城市环境的城市规划师——提供了基于扎实数据分析的决策工具。通过复杂的数据管理和描述性分析，数据科学为房地产行业的战略规划和运营改进奠定了基础。

探索数据科学在奢华住宅买家中的应用，你可以看到 PropTech 更广泛影响的一个具体示例。这个叙述不仅展示了数据驱动策略在评估奢华物业中的细致好处，还强调了数据在促进房地产行业更加整合、高效和以消费者为响应的基本作用。

让我们开始吧。

![](img/9b7e118c29f275d395d874c943989fbe.png)

用数据寻找价值：奢华房地产决策中的凝聚力

图片由 [Breno Assis](https://unsplash.com/photos/aerial-photography-of-rural-r3WAWU5Fi5Q) 提供。保留所有权利。

## 概述

本文分为三部分；它们是：

+   Folium：互动地图指南

+   用数据科学赋能奢华住宅买家：在市场中寻找价值

+   可视化机会：绘制通向可及奢华房地产的路径

## Folium：互动地图指南

Folium 是一个强大的 Python 库，通过交互式地图将地理空间数据生动呈现。利用 Leaflet.js，Folium 允许用户通过几行 Python 代码创建丰富的动态可视化，使其成为房地产专业人士和数据科学家的宝贵工具。它的主要优势在于其交互性，允许缩放和点击查看细节，以及与 Python 数据科学工具栈（特别是 pandas）的无缝集成，使数据驱动的地图可视化变得简单易行。

对于房地产行业，Folium 使得可以将房产数据可视化到地理背景中，提供对市场趋势、房产估值和人口统计叠加的无与伦比的清晰度。Folium 地图可以轻松保存为 HTML 文件，方便与客户或利益相关者共享。这一功能使复杂的地理空间分析变得更加民主化，通过互动地图增强演示、报告和房源列表，提供房地产全景的全面视图。

Folium 是 PyPI 中的一个包。要安装它，你可以在终端或命令行界面中使用以下命令：

```py
pip install folium
```

一旦成功安装所需的包，你就可以按照下面的示例继续操作。

**启动你的项目**，请参考我的书籍 [《数据科学初学者指南》](https://machinelearning.samcart.com/products/beginners-guide-data-science/)。它提供了**自学教程**和**可用代码**。

## 通过数据科学赋能奢侈品买家：在市场中发现价值

在今天的房地产市场中，潜在的购房者，尤其是对奢侈品领域感兴趣的人，面临的重大挑战是找到既满足他们审美和舒适度偏好又提供实质性价值的房产。这就是数据科学发挥关键作用的地方，它将寻找完美住宅的艰巨任务转变为一个充满见解和发现的旅程。

数据科学在房地产领域的力量在于其分析大量信息的能力，揭示那些一开始可能不易察觉的模式、趋势和机会。对于奢侈品买家来说，这意味着能够识别出既代表奢华又具有价值的房产，确保他们的投资既稳健又美观。

你的第一步是确定 Ames 中最昂贵的前 10 套房产。这个初步筛选作为你的起点，展示市场上被认为最有价值的房产。为此，你将使用 Python 中的 pandas 库来加载数据集并进行必要的分析。

下面是标志着你数据驱动旅程开始的代码：

```py
# Import the pandas library and load the dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Identify the 10 most expensive homes based on SalePrice with key features
top_10_expensive_homes = Ames.nlargest(10, 'SalePrice')
features = ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces']
top_10_df = top_10_expensive_homes[features]
print(top_10_df)
```

这块代码有效地筛选了 Ames 数据集，以提取 10 个最昂贵的房屋，重点关注定义奢华生活的关键属性：“SalePrice”（售价）、“GrLivArea”（地上生活面积）、“OverallQual”（整体材料和装修质量）、“KitchenQual”（厨房质量）、“TotRmsAbvGrd”（地上房间总数）和“Fireplaces”（壁炉）。这些特征为区分单纯的美学吸引力和真正的质量奠定了基础。

```py
      SalePrice  GrLivArea  OverallQual KitchenQual  TotRmsAbvGrd  Fireplaces
1007     755000       4316           10          Ex            10           2
1325     625000       3627           10          Gd            10           1
65       615000       2470           10          Ex             7           2
584      611657       2364            9          Ex            11           2
1639     610000       2674           10          Ex             8           2
615      591587       2338            9          Gd             8           2
2087     584500       3500            9          Ex            11           1
1112     555000       2402           10          Ex            10           2
802      538000       3279            8          Ex            12           1
407      535000       2698           10          Ex            11           1
```

为了进一步精细化你的搜索，你应用了体现奢华生活精髓的具体标准。你关注那些总体质量评级为 10 的房屋，表示最高水平的工艺和材料。此外，你还寻找厨房质量（KitchenQual 为“Ex”）优异和奢华舒适的重要特征：至少有两个壁炉的房产。这些标准帮助你筛选出真正代表 Ames 奢华巅峰的房产。

让我们执行下一块代码，将 10 个最昂贵的房屋筛选到符合这些精英标准的房产中：

```py
# Refine the search with highest quality, excellent kitchen, and 2 fireplaces
elite = top_10_df.query('OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >= 2').copy()
print(elite)
```

```py
      SalePrice  GrLivArea  OverallQual KitchenQual  TotRmsAbvGrd  Fireplaces
1007     755000       4316           10          Ex            10           2
65       615000       2470           10          Ex             7           2
1639     610000       2674           10          Ex             8           2
1112     555000       2402           10          Ex            10           2
```

这次精细化搜索将你的关注范围缩小到四个精英房产，这些房产不仅是 Ames 最昂贵的房产之一，而且符合奢华的严格标准。

随着你深入分析，介绍**特征工程**变得至关重要。特征工程是从现有数据中创建新变量或特征的过程，这些变量或特征更好地代表潜在问题。在你的案例中，特征工程有助于提升对房地产价值的理解。其中一个创新特征是**每平方英尺价格（PSF）**。PSF 指标成为你分析工具中的关键工具，提供了超越单纯挂牌价格的价值视角。通过计算每个房产的 PSF，你可以更公平地比较房产，无论其大小或绝对价格。这一度量标准揭示了每平方英尺生活空间的投资价值，为评估奢华房产的真正价值提供了标准化的尺度。

让我们将此计算应用到你精英选中的房屋中：

```py
# Introduce PSF to rank the options
elite['PSF'] = elite['SalePrice']/elite['GrLivArea']
print(elite.sort_values(by='PSF'))
```

这一行动产生了以下见解，使你可以根据相对价值对房产进行排名：

```py
      SalePrice  GrLivArea  OverallQual KitchenQual  TotRmsAbvGrd  Fireplaces         PSF
1007     755000       4316           10          Ex            10           2  174.930491
1639     610000       2674           10          Ex             8           2  228.122663
1112     555000       2402           10          Ex            10           2  231.057452
65       615000       2470           10          Ex             7           2  248.987854
```

在分析 PSF 后，很明显，并非所有奢华住宅都是相同的。尽管列表中的房产价格最高，但 PSF 最低的房产提供了最佳的价值，突显了该指标在评估奢华物业中的重要性。

现在你可以识别出符合奢华标准并根据 PSF 指标呈现出卓越价值的其他房产。通过扩大标准，包括数据集中总体质量评级为 10、厨房质量优秀、至少两个壁炉，但 PSF 低于$175 的所有房屋，目标是发现那些以更可接近价格提供奢华的房屋。

下面是你如何进行这次扩展分析的步骤：

```py
# Cross check entire homes to search for better value
Ames['PSF'] = Ames['SalePrice']/Ames['GrLivArea']
value = Ames.query('PSF < 175 & OverallQual == 10 & KitchenQual == "Ex" & Fireplaces >=2')
print(value[['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'PSF']])
```

这个精细的搜索产生了有趣的结果：

```py
      SalePrice  GrLivArea  OverallQual KitchenQual  TotRmsAbvGrd  Fireplaces         PSF
1007     755000       4316           10          Ex            10           2  174.930491
2003     475000       3608           10          Ex            12           2  131.651885
```

在对艾姆斯数据集进行的全面搜索中，你发现了两个不仅体现了奢华顶点的卓越设施和工艺，而且在奢侈品市场中也是价值的典范的属性。值得注意的是，其中一处房产的每平方英尺价格（PSF）显著低于你设定的门槛，为奢华购房者提供了绝佳的机会。这一发现突显了数据科学在房地产中的强大作用，使买家能够找到提供卓越居住体验和卓越财务价值的房屋。

从数值分析转向空间可视化，让我们转向 Folium 在艾姆斯的地理背景中绘制这些突出的房产。这一步骤将为你的发现提供视觉背景，并展示数据科学在提升房地产选择过程中的实际应用，使寻找完美奢侈房屋的过程既有信息支持，又充满视觉吸引力。

## 可视化机会：绘制通向可达奢侈房地产的路径

确定了两个突出的属性后，你的下一步是利用 Folium 将这些发现呈现在互动地图上。这种可视化不仅将每个属性置于艾姆斯（Ames，Iowa）的地理背景中，还通过允许你直接在地图上嵌入详细信息来丰富你的分析。

使用 Folium，你可以为这些杰出的房产创建标记，为潜在买家和利益相关者提供一目了然的全面概览。每个标记包含关于房产的关键数据点，包括销售价格、总生活面积、质量评级以及计算出的每平方英尺价格（PSF），提供了一种直观且引人入胜的方式来探索这些奢侈房屋。下面，你详细描述了将这些丰富信息添加到地图中的过程，确保每个房产不仅仅是地图上的一个点，而是通向其独特故事和价值主张的门户。

```py
# Visualize the results using folium
import folium
final_observation_indexes = value.index.tolist()

# Filter the dataset for these observations to get their latitude and longitude
final_locations = Ames.loc[final_observation_indexes, ['Latitude', 'Longitude']]

# Create a Folium map centered around the average location of the final observations
map_center = [final_locations['Latitude'].mean(), final_locations['Longitude'].mean()]
value_map = folium.Map(location=map_center, zoom_start=12)

# Add information to markers
for idx, row in final_locations.iterrows():
    # Extract additional information for the popup
    info = value.loc[idx, ['SalePrice', 'GrLivArea', 'OverallQual', 'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'PSF']]
    popup_text = f"""<b>Index:</b> {idx}<br>
                     <b>SalePrice:</b> {info['SalePrice']}<br>
                     <b>GrLivArea:</b> {info['GrLivArea']} sqft<br>
                     <b>OverallQual:</b> {info['OverallQual']}<br>
                     <b>KitchenQual:</b> {info['KitchenQual']}<br>
                     <b>TotRmsAbvGrd:</b> {info['TotRmsAbvGrd']}<br>
                     <b>Fireplaces:</b> {info['Fireplaces']}<br>
                     <b>PSF:</b> ${info['PSF']:.2f} /sqft"""
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=folium.Popup(popup_text, max_width=250)).add_to(value_map)

# Save the map to an HTML file on working directory
value_map.save('value_map.html')
```

执行此代码不仅会将互动标记填充到 Folium 地图中，还会将数据驱动探索的成果封装在一个 HTML 文件中，轻松保存到你的工作目录作为`value_map.html`。这个文件作为一个具体的实物，随时可以共享和进一步探索，邀请利益相关者以直观和动态的格式参与你的发现。以下是该文件的静态输出：

![](https://machinelearningmastery.com/wp-content/uploads/2024/03/Screenshot-2024-03-03-at-16.10.53.png)

当你总结你的分析之旅时，这种可视化成为一个关键资源，引导潜在的购房者穿越一个充满隐藏宝石的市场，这些宝石通过数据科学的细致应用得以照亮。这种方法不仅赋予买家对奢华房地产价值的更深刻理解，还促使更有信息、更具战略性和最终更有成效的决策过程。

热图是 Folium 提供的另一种强大的可视化技术。它提供了在特定地理区域内的数据密度或强度的生动表现，使你能够识别 Ames 房地产市场中的活动或兴趣热点。

```py
# Import HeatMap from folium
from folium.plugins import HeatMap

# Filter out rows with NaN values in 'Latitude' or 'Longitude'
Ames_Heat = Ames.dropna(subset=['Latitude', 'Longitude'])

# Group by 'Neighborhood' and calculate mean 'Latitude' and 'Longitude'
neighborhood_locs = Ames_Heat.groupby('Neighborhood').agg({'Latitude':'mean', 'Longitude':'mean'}).reset_index()

# Create a map centered around Ames, Iowa
ames_map_center = [Ames_Heat['Latitude'].mean(), Ames_Heat['Longitude'].mean()]
ames_heatmap = folium.Map(location=ames_map_center, zoom_start=12)

# Extract latitude and longitude data for the heatmap
heat_data = [[row['Latitude'], row['Longitude']] for index, row in Ames_Heat.iterrows()]

# Create and add a HeatMap layer to the map
HeatMap(heat_data, radius=12).add_to(ames_heatmap)

# Add one black flag per neighborhood to the map
for index, row in neighborhood_locs.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Neighborhood'],
        icon=folium.Icon(color='black', icon='flag')
    ).add_to(ames_heatmap)

# Save the map to an HTML file in the working directory
ames_heatmap.save('ames_heatmap.html')
```

在输出中，你战略性地放置了标记来标示 Ames 中的每个社区，为其位置提供了即时的视觉提示。这些标记通过其黑色与每个社区的全名区分，作为你探索过程中的导航指南。此外，热图利用颜色渐变来表示属性的密度，温暖的颜色代表更高的房屋浓度。这种颜色编码不仅增强了你地图的视觉吸引力，还提供了对市场活动的直观理解，并为开发者和买家提供了潜在的兴趣区域。

![](https://machinelearningmastery.com/wp-content/uploads/2024/03/Screenshot-2024-03-03-at-16.43.50.png)

利用热图可视化的洞察可以使开发者有针对性地瞄准低密度社区。通过关注新兴社区，并优先考虑创新设计和施工效率，开发者可以引入一种新的经济型奢华住宅维度。这一策略迎合了对高质量居住空间的需求，并通过使奢华住宅更易获得，扩大市场。这个结合的数据科学、地理空间分析和领域专家的信息策略，凸显了这些学科在塑造未来可及奢华房地产中的变革潜力，确保更多人能够触及高质量生活。

### 想开始学习数据科学入门指南吗？

现在就参加我的免费电子邮件速成课程（包含示例代码）。

点击注册，还可获得免费 PDF 电子书版本的课程。

## **进一步**阅读

#### Python 库

+   [Folium: 交互式映射库](https://pypi.org/project/folium/)

#### 教程

+   [利用数据科学实现房地产卓越](https://dataforest.ai/blog/leveraging-data-science-for-real-estate-excellence#:~:text=Data%20science%20in%20real%20estate%20empowers%20businesses%20by%20utilizing%20predictive,and%20employing%20machine%20learning%20algorithms.)

#### **资源**

+   [Ames 数据集](https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv)

+   [Ames 数据字典](https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt)

## **总结**

在这次全面的探索中，你深入探讨了数据科学和交互式地图在房地产市场中的变革性作用，特别是集中于爱荷华州艾姆斯的奢侈品市场。通过对数据分析和可视化技术的细致应用，你揭示了宝贵的洞察，这些洞察不仅使奢侈品购房者受益，还为开发商在可负担奢侈房地产领域内创新开辟了新途径。

具体而言，你学到了：

+   数据科学在识别奢侈物业中的应用。

+   特征工程的引入以及每平方英尺价格（PSF）的计算，作为评估相对价值的创新方法。

+   如何利用强大的 Python 库**Folium**创建动态视觉效果，以提升房地产决策。

你有任何问题吗？请在下面的评论中提出你的问题，我会尽力回答。
