# 理解 LangChain LLM 输出解析器

> 原文：[`machinelearningmastery.com/understanding-langchain-llm-output-parser/`](https://machinelearningmastery.com/understanding-langchain-llm-output-parser/)

![理解 LangChain LLM 输出解析器](img/da4313cfc413dea84a5ed4b24931a0ff.png)

图片来自 [Freepik](https://www.freepik.com/free-photo/computer-program-code_1154343.htm#fromView=search&page=2&position=42&uuid=15ea92ef-e413-4e1e-8489-219875d1a2af) 的 nikitabuida

大型语言模型（LLM）彻底改变了人们的工作方式。通过帮助用户从文本提示中生成答案，LLM 可以完成许多任务，例如回答问题、总结、计划事件等。

然而，有时 LLM 的输出可能达不到我们的标准。例如，生成的文本可能完全错误，需要进一步的指导。这时 LLM 输出解析器可以提供帮助。

通过使用 LangChain 输出解析器标准化输出结果，我们可以对输出进行一些控制。那么它是如何工作的呢？让我们深入了解一下。

## 准备工作

在本文中，我们将依赖于 LangChain 包，因此需要在环境中安装这些包。为此，你可以使用以下`code`。

```py
pip install langchain langchain_core langchain_community langchain_openai python-dotenv
```

此外，我们将使用 OpenAI GPT 模型进行文本生成，因此确保你拥有对它们的 API 访问权限。你可以从 OpenAI 平台获取 API 密钥。

我会在 Visual Studio Code IDE 中工作，但你可以在任何你喜欢的 IDE 中工作。在项目文件夹中创建一个名为`.env`的文件，并将 OpenAI API 密钥放入其中。它应该如下所示。

```py
OPENAI_API_KEY = sk-XXXXXXXXXX
```

一切准备就绪后，我们将进入文章的核心部分。

## 输出解析器

我们可以使用 LangChain 的多种输出解析器来标准化我们的 LLM 输出。我们将尝试其中几种，以更好地理解输出解析器。

首先，我们会尝试 Pydantic Parser。这是一个输出解析器，我们可以用来控制和验证生成文本的输出。让我们通过一个例子更好地使用它们。在你的 IDE 中创建一个 Python 脚本，然后将下面的代码复制到你的脚本中。

```py
from typing import List
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

load_dotenv()

class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="The year of the movie was released")
    genre: List[str] = Field(description="The main genres of the movie")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary of the movie plot")
    review: str = Field(description="Critical review of the movie")

    @validator("year")
    def valid_year(cls, val):
        if val  2025:
            raise ValueError("Must a valid movie year")
        return val

    @validator("rating")
    def valid_rating(cls, val):
        if val  10:
            raise ValueError("Rating must be between 0 and 10")
        return val

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate(
    template="Generate a movie review for the following movie:\n{movie_title}\n\n{format_instructions}",
    input_variables=["movie_title"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | parser

movie_title = "The Matrix"
review = chain.invoke({"movie_title": movie_title})
print(review)
```

我们最初在上面的代码中导入了这些包，并使用 `load_dotenv` 加载了 OpenAI 密钥。之后，我们创建了一个名为 `MovieReview` 的类，该类包含我们想要的所有信息输出。输出将包括标题、年份、类型、评分、摘要和评论。对于每个输出，我们定义了所需输出的描述。

从输出中，我们创建一个用于验证年份和评分的验证器，以确保结果符合我们的期望。你也可以根据需要添加更多验证机制。

然后我们创建了一个提示模板，该模板接受我们的查询输入以及其应有的格式。

我们最后做的事情是创建模型链并传递查询以获取结果。值得注意的是，上面的 `chain` 变量使用 “|” 接受结构，这是 LangChain 中的一种独特方法。

总的来说，结果类似于下面的内容。

输出：

```py
title='The Matrix' year=1999 genre=['Action', 'Sci-Fi'] rating=9.0 summary='A computer hacker learns about the true nature of reality and his role in the war against its controllers.' review="The Matrix is a groundbreaking film that revolutionized the action genre with its innovative special effects and thought-provoking storyline. Keanu Reeves delivers a memorable performance as Neo, the chosen one who must navigate the simulated reality of the Matrix to save humanity. The film's blend of martial arts, philosophy, and dystopian themes make it a must-watch for any movie enthusiast."
```

正如你所见，输出符合我们想要的格式，并且结果通过了我们的验证方法。

Pedantic Parser 是我们可以使用的标准输出解析器。如果我们已经有特定的格式需求，可以使用其他输出解析器。例如，如果我们希望结果以逗号分隔的项呈现，可以使用 CSV Parser。

```py
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List six {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | output_parser

print(chain.invoke({"subject": "Programming Language"}))
```

输出：

```py
['Java', 'Python', 'C++', 'JavaScript', 'Ruby', 'Swift']
```

结果是一个以逗号分隔的值的列表。如果结果是逗号分隔的，你可以按任何方式扩展模板。

也可以将输出格式更改为日期时间格式。通过修改代码和提示，我们可以期望得到我们想要的结果。

```py
from dotenv import load_dotenv
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

output_parser = DatetimeOutputParser()

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="""Answer the users question:

    {question}

    {format_instructions}""",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)

chain = prompt | model | output_parser

print(chain.invoke({"question": "When is the Python Programming Language invented?"}))
```

输出：

```py
1991-02-20 00:00:00
```

你可以看到结果是日期时间格式。

以上就是关于 LangChain LLM 输出解析器的内容。你可以访问它们的文档来找到你需要的输出解析器，或者使用 Pydantic 自行结构化输出。

## 结论

在这篇文章中，我们了解了 LangChain 输出解析器，它可以将 LLM 生成的文本标准化。我们可以使用 Pydantic Parser 来结构化 LLM 输出并提供所需结果。LangChain 还提供了许多其他可能适合你情况的输出解析器，如 CSV 解析器和日期时间解析器。
