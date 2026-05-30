from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.3
)

# 3. Output Parser
parser = StrOutputParser()


# Sequence Runnable
# In this, we just connect components like a pipeline. The output of one component becomes the input of the
# next. So your ow becomes something like prompt to model to parser. This is the foundation, and honestly,
# most applications start like this.
chain = prompt | model | parser

result = chain.invoke("Explain Deep Learning")
print(result)

