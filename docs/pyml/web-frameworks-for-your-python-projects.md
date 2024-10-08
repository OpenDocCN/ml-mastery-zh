# 你的 Python 项目的 web 框架

> 原文：[`machinelearningmastery.com/web-frameworks-for-your-python-projects/`](https://machinelearningmastery.com/web-frameworks-for-your-python-projects/)

当我们完成一个 Python 项目并推出供其他人使用时，最简单的方法是将项目呈现为命令行程序。如果你想让它更友好，可能需要为你的程序开发一个 GUI，以便人们可以在运行时通过鼠标点击进行互动。开发 GUI 可能很困难，因为人机交互的模型复杂。因此，折衷方案是为你的程序创建一个网页界面。这相比于纯命令行程序需要额外的工作，但不像使用 Qt5 库那样繁重。在这篇文章中，我们将展示网页界面的细节以及如何轻松地为你的程序提供一个界面。

完成本教程后，你将学到：

+   从一个简单的例子看 Flask 框架

+   使用 Dash 完全用 Python 构建交互式网页

+   一个 web 应用程序如何运行

**通过我的新书** [《Python 机器学习》](https://machinelearningmastery.com/python-for-machine-learning/)，**启动你的项目**，包括 *逐步教程* 和所有示例的 *Python 源代码* 文件。

让我们开始吧!![](img/3c9ae620ba0715f364dadb96217dea84.png)

你的 Python 项目的 web 框架

图片由 [Quang Nguyen Vinh](https://www.pexels.com/photo/photo-of-people-on-a-boat-during-dawn-2150371/) 提供。保留了一些权利。

## 概述

本教程分为五个部分，它们是：

+   Python 和网络

+   Flask 用于 web API 应用程序

+   Dash 用于交互式小部件

+   Dash 中的轮询

+   结合 Flask 和 Dash

## Python 和网络

网络通过超文本传输协议（HTTP）进行服务。Python 的标准库支持与 HTTP 的交互。如果你只是想用 Python 运行一个 web 服务器，没有比进入一个文件目录并运行命令更简单的了。

```py
python -m http.server
```

这通常会在 8000 端口启动一个 web 服务器。如果目录中存在 `index.html`，那将是我们在相同计算机上使用地址 `http://localhost:8000/` 打开浏览器时提供的默认页面。

这个内置的 web 服务器非常适合快速设置 web 服务器（例如，让网络上的另一台计算机下载一个文件）。但如果我们想做更多的事情，比如拥有一些动态内容，它将不够用。

在深入细节之前，让我们回顾一下我们在谈到网页界面时希望实现的目标。首先，现代网页将是一个与用户互动的界面，用于传播信息。这不仅意味着从服务器发送信息，还包括接收用户的输入。浏览器能够以美观的方式呈现信息。

另外，我们还可以使用不带浏览器的网页。例如，使用 web 协议下载文件。在 Linux 中，我们有著名的 `wget` 工具来完成这个任务。另一个例子是查询信息或向服务器传递信息。例如，在 AWS EC2 实例中，你可以在地址 `http://169.254.169.254/latest/meta-data/` 检查机器实例的 [元数据](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html)（其中 169.254.169.254 是 EC2 机器上可用的特殊 IP 地址）。在 Linux 实例中，我们可以使用 `curl` 工具进行检查。其输出将不是 HTML，而是机器可读的纯文本格式。有时，我们将其称为 web API，因为我们像使用远程执行函数一样使用它。

这两种是网页应用中的不同范式。第一种需要编写用户与服务器之间交互的代码。第二种需要在 URL 上设置各种端点，以便用户可以使用不同的地址请求不同的内容。在 Python 中，有第三方库可以完成这两种任务。

### 想开始学习 Python 机器学习吗？

现在就参加我的免费 7 天电子邮件速成课程（附示例代码）。

点击注册并获取免费的 PDF 电子书版本课程。

## Flask 用于 Web API 应用程序

允许我们用 Python 编写程序来构建基于网页的应用程序的工具被称为 **web 框架**。有很多这样的框架。Django 可能是最著名的一个。然而，不同的 web 框架的学习曲线可能差异很大。一些 web 框架假设你使用的是模型-视图设计，你需要理解其背后的原理才能明白如何使用它。

作为机器学习从业者，你可能希望做一些快速的、不太复杂的，但又足够强大以满足许多使用场景的事情。Flask 可能是这个类别中的一个不错选择。

Flask 是一个轻量级的 web 框架。你可以将它作为一个命令运行，并将其用作 Python 模块。假设我们想编写一个 web 服务器，报告任何用户指定时区的当前时间。可以通过 Flask 以简单的方式实现：

```py
from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown time zone: {timezone}\n"

app.run()
```

将以上内容保存到 `server.py` 或任何你喜欢的文件名中，然后在终端中运行它。你将看到以下内容：

```py
 * Serving Flask app 'time now' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

这意味着你的脚本现在正在 `http://127.0.0.1:5000` 作为一个 web 服务器运行。它将永远服务于 web 请求，直到你用 Ctrl-C 中断它。

如果你打开另一个终端并查询 URL，例如，在 Linux 中使用 `curl`：

```py
$ curl http://127.0.0.1:5000/now/Asia/Tokyo
2022-04-20 13:29:42 +0900 JST
```

你会在屏幕上看到以你请求的时区（在这个例子中是 Asia/Tokyo）打印的时间，你可以在维基百科上查看所有支持的时区列表 [在维基百科](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)。函数返回的字符串将是 URL 返回的内容。如果时区无法识别，你会看到“未知时区”消息，如上面代码中的 `except` 块所返回的。

如果我们想稍微扩展一下，假设在未提供时区的情况下使用 UTC，我们只需向函数中添加另一个装饰器：

```py
from datetime import datetime
import pytz
from flask import Flask

app = Flask("time now")

@app.route('/now', defaults={'timezone': ''})
@app.route("/now/<path:timezone>")
def timenow(timezone):
    try:
        if not timezone:
            zone = pytz.utc
        else:
            zone = pytz.timezone(timezone)
        now = datetime.now(zone)
        return now.strftime("%Y-%m-%d %H:%M:%S %z %Z\n")
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}\n"

app.run()
```

重启服务器后，我们可以看到如下结果：

```py
$ curl http://127.0.0.1:5000/now/Asia/Tokyo
2022-04-20 13:37:27 +0900 JST
$ curl http://127.0.0.1:5000/now/Asia/Tok
Unknown timezone: Asia/Tok
$ curl http://127.0.0.1:5000/now
2022-04-20 04:37:29 +0000 UTC
```

如今，许多这样的应用程序会返回一个 JSON 字符串以表示更复杂的数据，但从技术上讲，任何东西都可以被传递。如果你希望创建更多的 Web API，只需定义你的函数以返回数据，并像上面的例子一样用`@app.route()`进行装饰。

## 用于交互式小部件的 Dash

Flask 提供的 Web 端点非常强大。很多 Web 应用程序都是这样做的。例如，我们可以使用 HTML 编写网页用户界面，并用 Javascript 处理用户交互。一旦用户触发事件，我们可以让 Javascript 处理任何 UI 更改，并通过发送数据到一个端点创建一个 AJAX 调用并等待回复。AJAX 调用是异步的；因此，当接收到 Web 服务器的响应（通常在几分之一秒内）时，Javascript 会再次被触发，以进一步更新 UI，让用户了解情况。

然而，随着网页界面的复杂度越来越高，编写 Javascript 代码可能会变得繁琐。因此，有许多**客户端**库可以简化这一过程。有些库简化了 Javascript 编程，例如 jQuery。有些库改变了 HTML 和 Javascript 的交互方式，例如 ReactJS。但由于我们正在用 Python 开发机器学习项目，能够在不使用 Javascript 的情况下开发一个交互式网页应用将是非常棒的。Dash 就是为此而设的工具。

让我们考虑一个机器学习的例子：我们希望使用 MNIST 手写数字数据集来训练一个手写数字识别器。LeNet5 模型在这项任务中非常有名。但我们希望让用户微调 LeNet5 模型，重新训练它，然后用于识别。训练一个简单的 LeNet5 模型只需几行代码：

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical

# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data to (n_samples, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")

# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# LeNet5 model
model = Sequential([
    Conv2D(6, (5,5), activation="tanh",
           input_shape=(28,28,1), padding="same"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(16, (5,5), activation="tanh"),
    AveragePooling2D((2,2), strides=2),
    Conv2D(120, (5,5), activation="tanh"),
    Flatten(),
    Dense(84, activation="tanh"),
    Dense(10, activation="softmax")
])

# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)
```

在这段代码中，我们可以更改几个超参数，例如激活函数、训练的优化器、训练轮次和批量大小。我们可以在 Dash 中创建一个界面，让用户更改这些参数并重新训练模型。这个界面将以 HTML 呈现，但用 Python 编码：

```py
...
from flask import Flask
from dash import Dash, html, dcc

# default values
model_data = {
    "activation": "relu",
    "optimizer": "adam",
    "epochs": 100,
    "batchsize": 32,
}
...
server = Flask("mlm")
app = Dash(server=server)
app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            children="LeNet5 training",
            style={"textAlign": "center"}
        ),
        html.Div(
            className="flex-container",
            children=[
                html.Div(children=[
                    html.Div(id="activationdisplay", children="Activation:"),
                    dcc.Dropdown(
                        id="activation",
                        options=[
                            {"label": "Rectified linear unit", "value": "relu"},
                            {"label": "Hyperbolic tangent", "value": "tanh"},
                            {"label": "Sigmoidal", "value": "sigmoid"},
                        ],
                        value=model_data["activation"]
                    )
                ]),
                html.Div(children=[
                    html.Div(id="optimizerdisplay", children="Optimizer:"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=[
                            {"label": "Adam", "value": "adam"},
                            {"label": "Adagrad", "value": "adagrad"},
                            {"label": "Nadam", "value": "nadam"},
                            {"label": "Adadelta", "value": "adadelta"},
                            {"label": "Adamax", "value": "adamax"},
                            {"label": "RMSprop", "value": "rmsprop"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "FTRL", "value": "ftrl"},
                        ],
                        value=model_data["optimizer"]
                    ),
                ]),
                html.Div(children=[
                    html.Div(id="epochdisplay", children="Epochs:"),
                    dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
                               value=model_data["epochs"], id="epochs"),
                ]),
                html.Div(children=[
                    html.Div(id="batchdisplay", children="Batch size:"),
                    dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
                               value=model_data["batchsize"], id="batchsize"),
                ]),
            ]
        ),
        html.Button(id="train", n_clicks=0, children="Train"),
    ]
)
```

在这里，我们设置了一个基于 Flask 服务器的 Dash 应用程序。上面的代码主要用于设置 Dash 应用程序的**布局**，该布局将在网页浏览器中显示。布局顶部有一个标题，底部有一个按钮（标签为“Train”），中间有一个包含多个选项小部件的大框。布局中有一个用于激活函数的下拉框，一个用于训练优化器的下拉框，以及两个滑块，一个用于轮次，一个用于批量大小。布局如下所示：

![](img/4432753c56de312f60d25feba0400887.png)

如果你熟悉 HTML 开发，你可能注意到我们上面使用了许多`<div>`元素。此外，我们还向一些元素提供了`style`参数，以改变它们在浏览器中的渲染方式。实际上，我们将这些 Python 代码保存到文件`server.py`中，并创建了一个文件`assets/main.css`，其内容如下：

CSS

```py
.flex-container {
    display: flex;
    padding: 5px;
    flex-wrap: nowrap;
    background-color: #EEEEEE;
}

.flex-container > * {
    flex-grow: 1
}
```

当运行此代码时，我们可以使四个不同的用户选项水平对齐。

在创建了 HTML 前端之后，关键是让用户通过从下拉列表中选择或移动滑块来更改超参数。然后，在用户点击“训练”按钮后，我们启动模型训练。我们定义训练函数如下：

```py
...
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

def train():
    activation = model_data["activation"]
    model = Sequential([
        Conv2D(6, (5, 5), activation=activation,
               input_shape=(28, 28, 1), padding="same"),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(16, (5, 5), activation=activation),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(120, (5, 5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=model_data["optimizer"],
                  metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop])
    return model, history
```

这个函数依赖于一个外部字典`model_data`来获取参数和数据集，例如`X_train`和`y_train`，这些是在函数外部定义的。它将创建一个新模型，训练它，并返回带有训练历史的模型。我们只需在浏览器上的“训练”按钮被点击时运行此函数即可。我们在`fit()`函数中设置`verbose=0`，以要求训练过程不要向屏幕打印任何内容，因为它应该在服务器上运行，而用户则在浏览器中查看。用户无法看到服务器上的终端输出。我们还可以进一步显示训练周期中的损失和评估指标历史。这是我们需要做的：

```py
...
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State

...
app.layout = html.Div(
    id="parent",
    children=[
        ...
        html.Button(id="train", n_clicks=0, children="Train"),
        dcc.Graph(id="historyplot"),
    ]
)

...
@app.callback(Output("historyplot", "figure"),
              Input("train", "n_clicks"),
              State("activation", "value"),
              State("optimizer", "value"),
              State("epochs", "value"),
              State("batchsize", "value"),
              prevent_initial_call=True)
def train_action(n_clicks, activation, optimizer, epoch, batchsize):
    model_data.update({
        "activation": activation,
        "optimizer": optimizer,
        "epoch": epoch,
        "batchsize": batchsize,
    })
    model, history = train()
    model_data["model"] = model  # keep the trained model
    history = pd.DataFrame(history.history)
    fig = px.line(history, title="Model training metrics")
    fig.update_layout(xaxis_title="epochs",
                      yaxis_title="metric value", legend_title="metrics")
    return fig
```

我们首先在网页上添加一个`Graph`组件来显示我们的训练指标。`Graph`组件不是标准的 HTML 元素，而是 Dash 组件。Dash 提供了许多这样的组件，作为其主要特性。Dash 是 Plotly 的姊妹项目，Plotly 是一个类似于 Bokeh 的可视化库，将交互式图表渲染到 HTML 中。`Graph`组件用于显示 Plotly 图表。

然后我们定义了一个函数`train_action()`，并用我们 Dash 应用程序的回调函数装饰它。函数`train_action()`接受多个输入（模型超参数）并返回一个输出。在 Dash 中，输出通常是一个字符串，但我们在这里返回一个 Plotly 图形对象。回调装饰器要求我们指定输入和输出。这些是由其 ID 字段指定的网页组件，以及作为输入或输出的属性。在此示例中，除了输入和输出，我们还需要一些称为“状态”的额外数据。

在 Dash 中，输入是触发操作的因素。在这个示例中，Dash 中的一个按钮会记住它被按下的次数，这个次数存储在组件的属性`n_clicks`中。所以我们将这个属性的变化声明为触发该函数的因素。类似地，当这个函数返回时，图形对象将替换`Graph`组件。状态参数作为非触发参数提供给这个函数。指定输出、输入和状态的顺序非常重要，因为这是回调装饰器所期望的，以及我们定义的函数的参数顺序。

我们不会详细解释 Plotly 的语法。如果你了解了像 Bokeh 这样的可视化库的工作原理，查阅 Plotly 的文档后，应该不会很难将你的知识适应到 Plotly 上。

但是，我们需要提到 Dash 回调函数的一点：当网页首次加载时，所有回调函数会被调用一次，因为组件是新创建的。由于所有组件的属性从不存在到有了一些值，因此它们会触发事件。如果我们不希望它们在页面加载时被调用（例如，在这种情况下，我们不希望耗时的训练过程在用户确认超参数之前开始），我们需要在装饰器中指定`prevent_initial_call=True`。

我们可以进一步通过使超参数选择变得交互化来迈出一步。这是礼貌的，因为你会对用户的操作提供反馈。由于我们已经为每个选择组件的标题有一个`<div>`元素，我们可以利用它来提供反馈，通过创建以下函数：

```py
...

@app.callback(Output(component_id="epochdisplay", component_property="children"),
              Input(component_id="epochs", component_property="value"))
def update_epochs(value):
    return f"Epochs: {value}"

@app.callback(Output("batchdisplay", "children"),
              Input("batchsize", "value"))
def update_batchsize(value):
    return f"Batch size: {value}"

@app.callback(Output("activationdisplay", "children"),
              Input("activation", "value"))
def update_activation(value):
    return f"Activation: {value}"

@app.callback(Output("optimizerdisplay", "children"),
              Input("optimizer", "value"))
def update_optimizer(value):
    return f"Optimizer: {value}"
```

这些函数很简单，返回一个字符串，这个字符串会成为`<div>`元素的“子元素”。我们还展示了第一个函数装饰器中的命名参数，以防你希望更明确。

把所有内容整合在一起，以下是可以通过网页接口控制模型训练的完整代码：

```py
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask

server = Flask("mlm")
app = Dash(server=server)
# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model_data = {
    "activation": "relu",
    "optimizer": "adam",
    "epochs": 100,
    "batchsize": 32,
}

def train():
    activation = model_data["activation"]
    model = Sequential([
        Conv2D(6, (5, 5), activation=activation,
               input_shape=(28, 28, 1), padding="same"),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(16, (5, 5), activation=activation),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(120, (5, 5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=model_data["optimizer"],
                  metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop])
    return model, history

app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            children="LeNet5 training",
            style={"textAlign": "center"}
        ),
        html.Div(
            className="flex-container",
            children=[
                html.Div(children=[
                    html.Div(id="activationdisplay"),
                    dcc.Dropdown(
                        id="activation",
                        options=[
                            {"label": "Rectified linear unit", "value": "relu"},
                            {"label": "Hyperbolic tangent", "value": "tanh"},
                            {"label": "Sigmoidal", "value": "sigmoid"},
                        ],
                        value=model_data["activation"]
                    )
                ]),
                html.Div(children=[
                    html.Div(id="optimizerdisplay"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=[
                            {"label": "Adam", "value": "adam"},
                            {"label": "Adagrad", "value": "adagrad"},
                            {"label": "Nadam", "value": "nadam"},
                            {"label": "Adadelta", "value": "adadelta"},
                            {"label": "Adamax", "value": "adamax"},
                            {"label": "RMSprop", "value": "rmsprop"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "FTRL", "value": "ftrl"},
                        ],
                        value=model_data["optimizer"]
                    ),
                ]),
                html.Div(children=[
                    html.Div(id="epochdisplay"),
                    dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
                               value=model_data["epochs"], id="epochs"),
                ]),
                html.Div(children=[
                    html.Div(id="batchdisplay"),
                    dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
                               value=model_data["batchsize"], id="batchsize"),
                ]),
            ]
        ),
        html.Button(id="train", n_clicks=0, children="Train"),
        dcc.Graph(id="historyplot"),
    ]
)

@app.callback(Output(component_id="epochdisplay", component_property="children"),
              Input(component_id="epochs", component_property="value"))
def update_epochs(value):
    model_data["epochs"] = value
    return f"Epochs: {value}"

@app.callback(Output("batchdisplay", "children"),
              Input("batchsize", "value"))
def update_batchsize(value):
    model_data["batchsize"] = value
    return f"Batch size: {value}"

@app.callback(Output("activationdisplay", "children"),
              Input("activation", "value"))
def update_activation(value):
    model_data["activation"] = value
    return f"Activation: {value}"

@app.callback(Output("optimizerdisplay", "children"),
              Input("optimizer", "value"))
def update_optimizer(value):
    model_data["optimizer"] = value
    return f"Optimizer: {value}"

@app.callback(Output("historyplot", "figure"),
              Input("train", "n_clicks"),
              State("activation", "value"),
              State("optimizer", "value"),
              State("epochs", "value"),
              State("batchsize", "value"),
              prevent_initial_call=True)
def train_action(n_clicks, activation, optimizer, epoch, batchsize):
    model_data.update({
        "activation": activation,
        "optimizer": optimizer,
        "epcoh": epoch,
        "batchsize": batchsize,
    })
    model, history = train()
    model_data["model"] = model  # keep the trained model
    history = pd.DataFrame(history.history)
    fig = px.line(history, title="Model training metrics")
    fig.update_layout(xaxis_title="epochs",
                      yaxis_title="metric value", legend_title="metrics")
    return fig

# run server, with hot-reloading
app.run_server(debug=True, threaded=True)
```

上述代码的最后一行是运行 Dash 应用程序，就像我们在上一节中运行 Flask 应用程序一样。`run_server()` 函数的 `debug=True` 参数用于“热重载”，这意味着每当 Dash 检测到我们的脚本已更改时，它会重新加载所有内容。这在我们在另一个窗口编辑代码时非常方便，因为它不需要我们终止 Dash 服务器并重新运行。`threaded=True` 是要求 Dash 服务器在处理多个请求时以多线程运行。一般来说，不建议 Python 程序使用多线程，因为全局解释器锁的问题。但在 Web 服务器环境中，由于大多数时候服务器在等待 I/O，所以是可以接受的。如果不是多线程，选项将是多进程运行。我们不能在单线程和单进程中运行服务器，因为即使我们只为一个用户提供服务，浏览器也会同时启动多个 HTTP 查询（例如，请求我们上面创建的 CSS 文件时加载网页）。

## 在 Dash 中进行轮询

如果我们用中等数量的 epochs 运行上述 Dash 应用程序，它将花费相当长的时间来完成。我们希望看到它运行，而不仅仅在完成后更新图表。有一种方法可以要求 Dash 向我们的浏览器**推送**更新，但这需要一个插件（例如，[dash_devices](https://pypi.org/project/dash-devices/) 包可以做到这一点）。但我们也可以要求浏览器**拉取**任何更新。这种设计称为**轮询**。

在我们上面定义的 `train()` 函数中，我们设置 `verbose=0` 来跳过终端输出。但是我们仍然需要了解训练过程的进度。在 Keras 中，这可以通过自定义回调函数来完成。我们可以如下定义一个：

```py
...
from tensorflow.keras.callbacks import Callback

train_status = {
    "running": False,
    "epoch": 0,
    "batch": 0,
    "batch metric": None,
    "last epoch": None,
}

class ProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        train_status["running"] = True
        train_status["epoch"] = 0
    def on_train_end(self, logs=None):
        train_status["running"] = False
    def on_epoch_begin(self, epoch, logs=None):
        train_status["epoch"] = epoch
        train_status["batch"] = 0
    def on_epoch_end(self, epoch, logs=None):
        train_status["last epoch"] = logs
    def on_train_batch_begin(self, batch, logs=None):
        train_status["batch"] = batch
    def on_train_batch_end(self, batch, logs=None):
        train_status["batch metric"] = logs

def train():
    ...
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop, ProgressCallback()])
    return model, history
```

如果我们为 Keras 模型的 `fit()` 函数提供此类的实例，这个类的成员函数将在训练周期、epoch 或批次的开始或结束时被调用。在函数内部我们可以做很多事情。在 epoch 或批次结束时，函数的 `logs` 参数是损失和验证指标的字典。因此，我们定义了一个全局字典对象来记住这些指标。

现在，我们可以随时检查字典 `train_status` 来了解模型训练的进度，我们可以修改我们的网页来显示它：

```py
...

app.layout = html.Div(
    id="parent",
    children=[
        ...
        html.Button(id="train", n_clicks=0, children="Train"),
        html.Pre(id="progressdisplay"),
        dcc.Interval(id="trainprogress", n_intervals=0, interval=1000),
        dcc.Graph(id="historyplot"),
    ]
)

import json

@app.callback(Output("progressdisplay", "children"),
              Input("trainprogress", "n_intervals"))
def update_progress(n):
    return json.dumps(train_status, indent=4)
```

我们创建一个不可见组件 `dcc.Interval()`，它每隔 1000 毫秒（= 1 秒）自动更改其属性 `n_intervals`。然后我们在我们的“Train”按钮下创建一个 `<pre>` 元素，并命名为 `progressdisplay`。每当 `Interval` 组件触发时，我们将 `train_status` 字典转换为 JSON 字符串并显示在那个 `<pre>` 元素中。如果你愿意，你可以创建一个小部件来显示这些信息。Dash 提供了几个小部件。

仅仅通过这些更改，当您的模型训练完成时，您的浏览器将看起来像这样：

![](img/c52255159aa96eda68efef591a3fb33b.png)

以下是完整的代码。不要忘记你还需要 `assets/main.css` 文件以正确渲染网页：

```py
import json

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping

import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask

server = Flask("mlm")
app = Dash(server=server)

# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model_data = {
    "activation": "relu",
    "optimizer": "adam",
    "epochs": 100,
    "batchsize": 32,
}

train_status = {
    "running": False,
    "epoch": 0,
    "batch": 0,
    "batch metric": None,
    "last epoch": None,
}

class ProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        train_status["running"] = True
        train_status["epoch"] = 0
    def on_train_end(self, logs=None):
        train_status["running"] = False
    def on_epoch_begin(self, epoch, logs=None):
        train_status["epoch"] = epoch
        train_status["batch"] = 0
    def on_epoch_end(self, epoch, logs=None):
        train_status["last epoch"] = logs
    def on_train_batch_begin(self, batch, logs=None):
        train_status["batch"] = batch
    def on_train_batch_end(self, batch, logs=None):
        train_status["batch metric"] = logs

def train():
    activation = model_data["activation"]
    model = Sequential([
        Conv2D(6, (5, 5), activation=activation,
               input_shape=(28, 28, 1), padding="same"),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(16, (5, 5), activation=activation),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(120, (5, 5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=model_data["optimizer"],
                  metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop, ProgressCallback()])
    return model, history

app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            children="LeNet5 training",
            style={"textAlign": "center"}
        ),
        html.Div(
            className="flex-container",
            children=[
                html.Div(children=[
                    html.Div(id="activationdisplay"),
                    dcc.Dropdown(
                        id="activation",
                        options=[
                            {"label": "Rectified linear unit", "value": "relu"},
                            {"label": "Hyperbolic tangent", "value": "tanh"},
                            {"label": "Sigmoidal", "value": "sigmoid"},
                        ],
                        value=model_data["activation"]
                    )
                ]),
                html.Div(children=[
                    html.Div(id="optimizerdisplay"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=[
                            {"label": "Adam", "value": "adam"},
                            {"label": "Adagrad", "value": "adagrad"},
                            {"label": "Nadam", "value": "nadam"},
                            {"label": "Adadelta", "value": "adadelta"},
                            {"label": "Adamax", "value": "adamax"},
                            {"label": "RMSprop", "value": "rmsprop"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "FTRL", "value": "ftrl"},
                        ],
                        value=model_data["optimizer"]
                    ),
                ]),
                html.Div(children=[
                    html.Div(id="epochdisplay"),
                    dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
                               value=model_data["epochs"], id="epochs"),
                ]),
                html.Div(children=[
                    html.Div(id="batchdisplay"),
                    dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
                               value=model_data["batchsize"], id="batchsize"),
                ]),
            ]
        ),
        html.Button(id="train", n_clicks=0, children="Train"),
        html.Pre(id="progressdisplay"),
        dcc.Interval(id="trainprogress", n_intervals=0, interval=1000),
        dcc.Graph(id="historyplot"),
    ]
)

@app.callback(Output(component_id="epochdisplay", component_property="children"),
              Input(component_id="epochs", component_property="value"))
def update_epochs(value):
    return f"Epochs: {value}"

@app.callback(Output("batchdisplay", "children"),
              Input("batchsize", "value"))
def update_batchsize(value):
    return f"Batch size: {value}"

@app.callback(Output("activationdisplay", "children"),
              Input("activation", "value"))
def update_activation(value):
    return f"Activation: {value}"

@app.callback(Output("optimizerdisplay", "children"),
              Input("optimizer", "value"))
def update_optimizer(value):
    return f"Optimizer: {value}"

@app.callback(Output("historyplot", "figure"),
              Input("train", "n_clicks"),
              State("activation", "value"),
              State("optimizer", "value"),
              State("epochs", "value"),
              State("batchsize", "value"),
              prevent_initial_call=True)
def train_action(n_clicks, activation, optimizer, epoch, batchsize):
    model_data.update({
        "activation": activation,
        "optimizer": optimizer,
        "epoch": epoch,
        "batchsize": batchsize,
    })
    model, history = train()
    model_data["model"] = model  # keep the trained model
    history = pd.DataFrame(history.history)
    fig = px.line(history, title="Model training metrics")
    fig.update_layout(xaxis_title="epochs",
                      yaxis_title="metric value", legend_title="metrics")
    return fig

@app.callback(Output("progressdisplay", "children"),
              Input("trainprogress", "n_intervals"))
def update_progress(n):
    return json.dumps(train_status, indent=4)

# run server, with hot-reloading
app.run_server(debug=True, threaded=True)
```

## 结合 Flask 和 Dash

你也可以提供一个网页界面来 **使用** 训练好的模型吗？当然可以。如果模型接受一些数字输入，这会更容易，因为我们只需在页面上提供一个输入框元素。在这种情况下，由于这是一个手写数字识别模型，我们需要一种方法在浏览器中提供图像，并将其传递给服务器上的模型。只有这样，我们才能获得结果并显示出来。我们可以选择两种方式来实现这一点：我们可以让用户上传一个数字图像供模型识别，或者让用户直接在浏览器中绘制图像。

在 HTML5 中，我们有一个 `<canvas>` 元素，允许我们在网页上绘制或显示像素。我们可以利用这个元素让用户在上面绘制，然后将其转换为 28×28 的数字矩阵，并将其发送到服务器端，让模型进行预测并显示预测结果。

这样做不是 Dash 的工作，因为我们想要读取 `<canvas>` 元素并将其转换为正确格式的矩阵。我们将在 Javascript 中完成这项工作。但之后，我们会在一个网页 URL 中调用模型，就像我们在文章开头所描述的那样。一个带有参数的查询会被发送，服务器的响应将是我们的模型识别出的数字。

在后台，Dash 使用 Flask，根 URL 指向 Dash 应用程序。我们可以创建一个使用模型的 Flask 端点，如下所示：

```py
...
@server.route("/recognize", methods=["POST"])
def recognize():
    if not model_data.get("model"):
        return "Please train your model."
    matrix = json.loads(request.form["matrix"])
    matrix = np.asarray(matrix).reshape(1, 28, 28)
    proba = model_data["model"].predict(matrix).reshape(-1)
    result = np.argmax(proba)
    return "Digit "+str(result)
```

正如我们所回忆的，变量 `server` 是我们构建 Dash 应用程序的 Flask 服务器。我们使用其装饰器创建一个端点。由于我们要传递一个 28×28 矩阵作为参数，因此我们使用 HTTP POST 方法，这对于大块数据更为适合。POST 方法提供的数据不会成为 URL 的一部分。因此，我们没有在 `@server.route()` 装饰器中设置路径参数。相反，我们通过 `request.form["matrix"]` 读取数据，其中 `"matrix"` 是我们传递的参数名称。然后我们假设字符串为 JSON 格式，将其转换为数字列表，并进一步转换为 NumPy 数组，然后传递给模型以预测数字。我们将训练好的模型保存在 `model_data["model"]` 中，但我们可以通过检查该训练模型是否存在并在不存在时返回错误消息，使上述代码更健壮。

要修改网页，我们只需添加一些组件：

```py
app.layout = html.Div(
    id="parent",
    children=[
        ...
        dcc.Graph(id="historyplot"),
        html.Div(
            className="flex-container",
            id="predict",
            children=[
                html.Div(
                    children=html.Canvas(id="writing"),
                    style={"textAlign": "center"}
                ),
                html.Div(id="predictresult", children="?"),
                html.Pre(
                    id="lastinput",
                ),
            ]
        ),
        html.Div(id="dummy", style={"display": "none"}),
    ]
)
```

底部是一个隐藏的 `<div>` 元素，我们稍后将使用它。主要部分是另一个 `<div>` 元素，其中包含三个项目，即一个 `<canvas>` 元素（ID 为 `"writing"`），一个 `<div>` 元素（ID 为 `"predictresult"`）用于显示结果，以及一个 `<pre>` 元素（ID 为 `"lastinput"`）用于显示我们传递给服务器的矩阵。

由于这些元素不是由 Dash 处理的，我们不需要在 Python 中创建更多的函数。相反，我们需要创建一个 JavaScript 文件 `assets/main.js` 以便与这些组件进行交互。Dash 应用程序会自动加载 `assets` 目录下的所有内容，并在网页加载时将其发送给用户。我们可以用纯 JavaScript 编写这些内容，但为了使代码更简洁，我们将使用 jQuery。因此，我们需要告诉 Dash 我们将在这个 Web 应用程序中使用 jQuery：

```py
...
app = Dash(server=server,
           external_scripts=[
               "https://code.jquery.com/jquery-3.6.0.min.js"
           ])
```

`external_scripts` 参数是一个 URL 列表，这些 URL 指向将在网页加载**之前**作为附加脚本加载的资源。因此，我们通常会在这里提供库，但将我们自己的代码保持在外部。

我们自己的 JavaScript 代码将是一个单独的函数，因为它在网页完全加载后被调用：

JavaScript

```py
function pageinit() {
	// Set up canvas object
	var canvas = document.getElementById("writing");
	canvas.width = parseInt($("#writing").css("width"));
	canvas.height = parseInt($("#writing").css("height"));
	var context = canvas.getContext("2d");  // to remember drawing
	context.strokeStyle = "#FF0000";        // draw in bright red
	context.lineWidth = canvas.width / 15;  // thickness adaptive to canvas size

	...
};
```

我们首先在 JavaScript 中设置 `<canvas>` 元素。这些设置是特定于我们需求的。首先，我们将以下内容添加到 `assets/main.css` 中：

CSS

```py
canvas#writing {
    width: 300px;
    height: 300px;
    margin: auto;
    padding: 10px;
    border: 3px solid #7f7f7f;
    background-color: #FFFFFF;
}
```

这将宽度和高度固定为 300 像素，以使我们的画布成为正方形，同时进行其他美观上的微调。由于最终我们会将手写的内容转换为 28×28 像素的图像，以适应模型的期望，所以我们在画布上写的每一笔都不能过于细。因此，我们将笔画宽度设置为与画布大小相关。

仅有这些还不足以使我们的画布可用。假设我们从未在移动设备上使用它，而只在桌面浏览器上使用，绘图是通过鼠标点击和移动完成的。我们需要定义鼠标点击在画布上执行的操作。因此，我们将以下功能添加到 JavaScript 代码中：

JavaScript

```py
function pageinit() {
	...

	// Canvas reset by timeout
	var timeout = null; // holding the timeout event
	var reset = function() {
		// clear the canvas
		context.clearRect(0, 0, canvas.width, canvas.height);
	}

	// Set up drawing with mouse
	var mouse = {x:0, y:0}; // to remember the coordinate w.r.t. canvas
	var onPaint = function() {
		clearTimeout(timeout);
		// event handler for mouse move in canvas
		context.lineTo(mouse.x, mouse.y);
		context.stroke();
	};

	// HTML5 Canvas mouse event - in case of desktop browser
	canvas.addEventListener("mousedown", function(e) {
		clearTimeout(timeout);
		// mouse down, begin path at current mouse position
		context.moveTo(mouse.x, mouse.y);
		context.beginPath();
		// all mouse move from now on should be painted
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);
	canvas.addEventListener("mousemove", function(e) {
		// mouse move remember position w.r.t. canvas
		mouse.x = e.pageX - this.offsetLeft;
		mouse.y = e.pageY - this.offsetTop;
	}, false);
	canvas.addEventListener("mouseup", function(e) {
		clearTimeout(timeout);
		// all mouse move from now on should NOT be painted
		canvas.removeEventListener("mousemove", onPaint, false);
		// read drawing into image
		var img = new Image(); // on load, this will be the canvas in same WxH
		img.onload = function() {
			// Draw the 28x28 to top left corner of canvas
			context.drawImage(img, 0, 0, 28, 28);
			// Extract data: Each pixel becomes a RGBA value, hence 4 bytes each
			var data = context.getImageData(0, 0, 28, 28).data;
			var input = [];
			for (var i=0; i<data.length; i += 4) {
				// scan each pixel, extract first byte (R component)
				input.push(data[i]);
			};

			// TODO: use "input" for prediction
		};
		img.src = canvas.toDataURL("image/png");
		timeout = setTimeout(reset, 5000); // clear canvas after 5 sec
	}, false);
};
```

这有点啰嗦，但基本上我们要求监听画布上的三个鼠标事件，即按下鼠标按钮、移动鼠标和释放鼠标按钮。这三个事件组合在一起就是我们在画布上绘制一笔的方式。

首先，我们添加到 `<canvas>` 元素上的 `mousemove` 事件处理器仅仅是为了记住 JavaScript 对象 `mouse` 中当前的鼠标位置。

然后在 `mousedown` 事件处理器中，我们从最新的鼠标位置开始绘图上下文。由于绘图已经开始，所有后续的鼠标移动都应该在画布上绘制。我们定义了 `onPaint` 函数，以将线段扩展到画布上当前的鼠标位置。现在这个函数被注册为 `mousemove` 事件的附加事件处理器。

最后，`mouseup` 事件处理程序用于处理用户完成一次绘制并释放鼠标按钮的情况。所有后续的鼠标移动不应在画布上绘制，因此我们需要移除 `onPaint` 函数的事件处理程序。然后，当我们完成一次绘制时，这 **可能是** 一个完成的数字，因此我们想将其提取为 28×28 像素版本。这可以很容易完成。我们只需在 Javascript 中创建一个新的 `Image` 对象，并将整个画布加载到其中。当完成后，Javascript 会自动调用与之关联的 `onload` 函数。在其中，我们将这个 `Image` 对象转化为 28×28 像素，并绘制到我们 `context` 对象的左上角。然后我们逐像素读取它（每个像素将是 0 到 255 的 RGB 值，但由于我们使用红色绘制，我们只关心红色通道）到 Javascript 数组 `input` 中。我们只需将这个 `input` 数组传递给我们的模型，然后可以进行预测。

我们不想创建任何额外的按钮来清除我们的画布或提交我们的数字进行识别。因此，我们希望如果用户在 5 秒内没有绘制任何新内容，画布会自动清除。这是通过 Javascript 函数 `setTimeout()` 和 `clearTimeout()` 实现的。我们创建一个 `reset` 函数来清除画布，该函数将在 `mouseup` 事件后 5 秒触发。而这个计划调用的 `reset` 函数会在超时之前发生绘制事件时被取消。同样，每当发生 `mouseup` 事件时，识别也会自动进行。

给定我们有一个 28×28 像素的输入数据被转化为一个 Javascript 数组，我们可以直接使用我们用 Flask 创建的 `recognize` 端点。如果我们能看到我们传递给 `recognize` 的内容以及它返回的结果会很有帮助。所以我们在 ID 为 `lastinput` 的 `<pre>` 元素中显示输入数据，并在 ID 为 `predictresult` 的 `<div>` 元素中显示 `recognize` 端点返回的结果。这可以通过稍微扩展 `mouseup` 事件处理程序轻松完成。

JavaScript

```py
function pageinit() {
	canvas.addEventListener("mouseup", function(e) {
		...
		img.onload = function() {
            ...
			var input = [];
			for (var i=0; i<data.length; i += 4) {
				// scan each pixel, extract first byte (R component)
				input.push(data[i]);
			};
			var matrix = [];
			for (var i=0; i<input.length; i+=28) {
				matrix.push(input.slice(i, i+28).toString());
			};
			$("#lastinput").html("[[" + matrix.join("],<br/>[") + "]]");
			// call predict function with the matrix
			predict(input);
		};
		img.src = canvas.toDataURL("image/png");
		setTimeout(reset, 5000); // clear canvas after 5 sec
	}, false);

	function predict(input) {
		$.ajax({
			type: "POST",
			url: "/recognize",
			data: {"matrix": JSON.stringify(input)},
			success: function(result) {
				$("#predictresult").html(result);
			}
		});
	};
};
```

我们定义了一个新的 Javascript 函数 `predict()`，它会发起一个 AJAX 调用到我们用 Flask 设置的 `recognize` 端点。它使用 POST 方法，数据 `matrix` 赋值为 Javascript 数组的 JSON 版本。我们不能直接在 HTTP 请求中传递数组，因为一切必须被序列化。当 AJAX 调用返回时，我们更新 `<div>` 元素以显示结果。

这个 `predict()` 函数是由 `mouseup` 事件处理程序调用的，当我们完成将 28×28 像素图像转化为数字数组时。同时，我们将一个版本写入 `<pre>` 元素，仅用于显示目的。

到这里，我们的应用程序已经完成。但我们仍然需要在 Dash 应用程序加载时调用`pageinit()`函数。实际上，Dash 应用程序使用 React 来进行延迟渲染，因此我们不应该将`pageinit()`函数挂钩到`document.onload`事件处理程序上，否则我们会发现我们要找的组件不存在。正确的方法是在 Dash 应用程序完全加载时调用 JavaScript 函数是设置一个**客户端回调**，即由浏览器端 JavaScript 处理的回调，而不是服务器端的 Python。我们在 Python 程序`server.py`中添加以下函数调用：

```py
...
app.clientside_callback(
    "pageinit",
    Output("dummy", "children"),
    Input("dummy", "children")
)
```

`clientside_callback()`函数不是作为装饰器使用，而是作为完整的函数调用。它将 JavaScript 函数作为第一个参数，将`Output`和`Input`对象作为第二和第三个参数，类似于回调装饰器的情况。由于这个原因，我们在网页布局中创建了一个隐藏的虚拟组件，以帮助在页面加载时触发 JavaScript 函数，所有 Dash 回调会被调用一次，除非`prevent_initial_call=True`作为回调的一个参数。

现在我们一切就绪。我们可以运行`server.py`脚本来启动我们的 Web 服务器，它将加载`assets/`目录下的两个文件。打开浏览器访问 Dash 应用程序报告的 URL，我们可以更改超参数并训练模型，然后使用模型进行预测。

综合起来，以下是我们 JavaScript 部分的完整代码，保存为`assets/main.js`：

JavaScript

```py
function pageinit() {
	// Set up canvas object
	var canvas = document.getElementById("writing");
	canvas.width = parseInt($("#writing").css("width"));
	canvas.height = parseInt($("#writing").css("height"));
	var context = canvas.getContext("2d");  // to remember drawing
	context.strokeStyle = "#FF0000";        // draw in bright red
	context.lineWidth = canvas.width / 15;  // thickness adaptive to canvas size

	// Canvas reset by timeout
	var timeout = null; // holding the timeout event
	var reset = function() {
		// clear the canvas
		context.clearRect(0, 0, canvas.width, canvas.height);
	}

	// Set up drawing with mouse
	var mouse = {x:0, y:0}; // to remember the coordinate w.r.t. canvas
	var onPaint = function() {
		clearTimeout(timeout);
		// event handler for mousemove in canvas
		context.lineTo(mouse.x, mouse.y);
		context.stroke();
	};

	// HTML5 Canvas mouse event - in case of desktop browser
	canvas.addEventListener("mousedown", function(e) {
		clearTimeout(timeout);
		// mousedown, begin path at mouse position
		context.moveTo(mouse.x, mouse.y);
		context.beginPath();
		// all mousemove from now on should be painted
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);
	canvas.addEventListener("mousemove", function(e) {
		// mousemove remember position w.r.t. canvas
		mouse.x = e.pageX - this.offsetLeft;
		mouse.y = e.pageY - this.offsetTop;
	}, false);
	canvas.addEventListener("mouseup", function(e) {
		clearTimeout(timeout);
		// all mousemove from now on should NOT be painted
		canvas.removeEventListener("mousemove", onPaint, false);
		// read drawing into image
		var img = new Image(); // on load, this will be the canvas in same WxH
		img.onload = function() {
			// Draw the 28x28 to top left corner of canvas
			context.drawImage(img, 0, 0, 28, 28);
			// Extract data: Each pixel becomes a RGBA value, hence 4 bytes each
			var data = context.getImageData(0, 0, 28, 28).data;
			var input = [];
			for (var i=0; i<data.length; i += 4) {
				// scan each pixel, extract first byte (R component)
				input.push(data[i]);
			};
			var matrix = [];
			for (var i=0; i<input.length; i+=28) {
				matrix.push(input.slice(i, i+28).toString());
			};
			$("#lastinput").html("[[" + matrix.join("],\n[") + "]]");
			// call predict function with the matrix
			predict(input);
		};
		img.src = canvas.toDataURL("image/png");
		timeout = setTimeout(reset, 5000); // clear canvas after 5 sec
	}, false);

	function predict(input) {
		$.ajax({
			type: "POST",
			url: "/recognize",
			data: {"matrix": JSON.stringify(input)},
			success: function(result) {
				$("#predictresult").html(result);
			}
		});
	};
};
```

以下是 CSS 的完整代码，`assets/main.css`（`pre#lastinput`部分是使用较小的字体显示我们的输入矩阵）：

CSS

```py
.flex-container {
    display: flex;
    padding: 5px;
    flex-wrap: nowrap;
    background-color: #EEEEEE;
}

.flex-container > * {
    flex-grow: 1
}

canvas#writing {
    width: 300px;
    height: 300px;
    margin: auto;
    padding: 10px;
    border: 3px solid #7f7f7f;
    background-color: #FFFFFF;
}

pre#lastinput {
    font-size: 50%;
}
```

以下是主要的 Python 程序，`server.py`：

```py
import json

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping

import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask, request

server = Flask("mlm")
app = Dash(server=server,
           external_scripts=[
               "https://code.jquery.com/jquery-3.6.0.min.js"
           ])

# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model_data = {
    "activation": "relu",
    "optimizer": "adam",
    "epochs": 100,
    "batchsize": 32,
    "model": load_model("lenet5.h5"),
}
train_status = {
    "running": False,
    "epoch": 0,
    "batch": 0,
    "batch metric": None,
    "last epoch": None,
}

class ProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        train_status["running"] = True
        train_status["epoch"] = 0
    def on_train_end(self, logs=None):
        train_status["running"] = False
    def on_epoch_begin(self, epoch, logs=None):
        train_status["epoch"] = epoch
        train_status["batch"] = 0
    def on_epoch_end(self, epoch, logs=None):
        train_status["last epoch"] = logs
    def on_train_batch_begin(self, batch, logs=None):
        train_status["batch"] = batch
    def on_train_batch_end(self, batch, logs=None):
        train_status["batch metric"] = logs

def train():
    activation = model_data["activation"]
    model = Sequential([
        Conv2D(6, (5, 5), activation=activation,
               input_shape=(28, 28, 1), padding="same"),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(16, (5, 5), activation=activation),
        AveragePooling2D((2, 2), strides=2),
        Conv2D(120, (5, 5), activation=activation),
        Flatten(),
        Dense(84, activation=activation),
        Dense(10, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer=model_data["optimizer"],
                  metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,
                              restore_best_weights=True)
    history = model.fit(
            X_train, y_train, validation_data=(X_test, y_test),
            epochs=model_data["epochs"],
            batch_size=model_data["batchsize"],
            verbose=0, callbacks=[earlystop, ProgressCallback()])
    return model, history

app.layout = html.Div(
    id="parent",
    children=[
        html.H1(
            children="LeNet5 training",
            style={"textAlign": "center"}
        ),
        html.Div(
            className="flex-container",
            children=[
                html.Div(children=[
                    html.Div(id="activationdisplay"),
                    dcc.Dropdown(
                        id="activation",
                        options=[
                            {"label": "Rectified linear unit", "value": "relu"},
                            {"label": "Hyperbolic tangent", "value": "tanh"},
                            {"label": "Sigmoidal", "value": "sigmoid"},
                        ],
                        value=model_data["activation"]
                    )
                ]),
                html.Div(children=[
                    html.Div(id="optimizerdisplay"),
                    dcc.Dropdown(
                        id="optimizer",
                        options=[
                            {"label": "Adam", "value": "adam"},
                            {"label": "Adagrad", "value": "adagrad"},
                            {"label": "Nadam", "value": "nadam"},
                            {"label": "Adadelta", "value": "adadelta"},
                            {"label": "Adamax", "value": "adamax"},
                            {"label": "RMSprop", "value": "rmsprop"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "FTRL", "value": "ftrl"},
                        ],
                        value=model_data["optimizer"]
                    ),
                ]),
                html.Div(children=[
                    html.Div(id="epochdisplay"),
                    dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
                               value=model_data["epochs"], id="epochs"),
                ]),
                html.Div(children=[
                    html.Div(id="batchdisplay"),
                    dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
                               value=model_data["batchsize"], id="batchsize"),
                ]),
            ]
        ),
        html.Button(id="train", n_clicks=0, children="Train"),
        html.Pre(id="progressdisplay"),
        dcc.Interval(id="trainprogress", n_intervals=0, interval=1000),
        dcc.Graph(id="historyplot"),
        html.Div(
            className="flex-container",
            id="predict",
            children=[
                html.Div(
                    children=html.Canvas(id="writing"),
                    style={"textAlign": "center"}
                ),
                html.Div(id="predictresult", children="?"),
                html.Pre(
                    id="lastinput",
                ),
            ]
        ),
        html.Div(id="dummy", style={"display": "none"}),
    ]
)

@app.callback(Output(component_id="epochdisplay", component_property="children"),
              Input(component_id="epochs", component_property="value"))
def update_epochs(value):
    model_data["epochs"] = value
    return f"Epochs: {value}"

@app.callback(Output("batchdisplay", "children"),
              Input("batchsize", "value"))
def update_batchsize(value):
    model_data["batchsize"] = value
    return f"Batch size: {value}"

@app.callback(Output("activationdisplay", "children"),
              Input("activation", "value"))
def update_activation(value):
    model_data["activation"] = value
    return f"Activation: {value}"

@app.callback(Output("optimizerdisplay", "children"),
              Input("optimizer", "value"))
def update_optimizer(value):
    model_data["optimizer"] = value
    return f"Optimizer: {value}"

@app.callback(Output("historyplot", "figure"),
              Input("train", "n_clicks"),
              State("activation", "value"),
              State("optimizer", "value"),
              State("epochs", "value"),
              State("batchsize", "value"),
              prevent_initial_call=True)
def train_action(n_clicks, activation, optimizer, epoch, batchsize):
    model_data.update({
        "activation": activation,
        "optimizer": optimizer,
        "epoch": epoch,
        "batchsize": batchsize,
    })
    model, history = train()
    model_data["model"] = model  # keep the trained model
    history = pd.DataFrame(history.history)
    fig = px.line(history, title="Model training metrics")
    fig.update_layout(xaxis_title="epochs",
                      yaxis_title="metric value", legend_title="metrics")
    return fig

@app.callback(Output("progressdisplay", "children"),
              Input("trainprogress", "n_intervals"))
def update_progress(n):
    return json.dumps(train_status, indent=4)

app.clientside_callback(
    "function() { pageinit(); };",
    Output("dummy", "children"),
    Input("dummy", "children")
)

@server.route("/recognize", methods=["POST"])
def recognize():
    if not model_data.get("model"):
        return "Please train your model."
    matrix = json.loads(request.form["matrix"])
    matrix = np.asarray(matrix).reshape(1, 28, 28)
    proba = model_data["model"].predict(matrix).reshape(-1)
    result = np.argmax(proba)
    return "Digit "+str(result)

# run server, with hot-reloading
app.run_server(debug=True, threaded=True)
```

如果我们运行所有这些，我们应该看到如下屏幕：![](img/04723f0ab4a7bcaf5c76799e3aad525e.png)

## 深入阅读

目前有大量的 Web 框架可用，Flask 只是其中之一。另一个流行的框架是 CherryPy。如果你想深入了解，以下是相关资源。

#### 书籍

+   [Python Dash: Build Stunning Data Analysis and Visualization Apps with Plotly](https://www.amazon.com/dp/1718502222/) 由 Adam Schroeder、Christian Mayer 和 Ann Marie Ward 编写

+   [Interactive Dashboards and Data Apps with Plotly and Dash](https://www.amazon.com/dp/1800568916/) 由 Elias Dabbas 编写

+   [Flask Web Development](https://www.amazon.com/dp/1491991739)，由 Miguel Grinberg 编写，第 2 版

+   [Flask Framework Cookbook](https://www.amazon.com/dp/1789951291/)，由 Shalabh Aggarwal 编写，第 2 版

#### 文章

+   [Web Frameworks](https://wiki.python.org/moin/WebFrameworks)，Python.org wiki

#### APIs 和软件

+   [CherryPy](https://cherrypydocrework.readthedocs.io/)

+   [Django](https://www.djangoproject.com/)

+   [Flask](https://flask.palletsprojects.com/en/2.1.x/)

+   [Dash](https://dash.plotly.com/)

+   [Plotly](https://plotly.com/)

+   MDN [Javascript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

+   MDN [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

+   [jQuery](https://jquery.com/)

## 总结

在本教程中，你学习了如何使用 Dash 库在 Python 中轻松构建网页应用。你还学会了如何使用 Flask 创建一些网页 API。具体来说，你学习了

+   网页应用的机制

+   我们如何使用 Dash 来构建一个由网页组件触发的简单网页应用

+   我们如何使用 Flask 创建网页 API

+   如何在 Javascript 中构建网页应用，并在使用我们用 Flask 构建的网页 API 的浏览器上运行
