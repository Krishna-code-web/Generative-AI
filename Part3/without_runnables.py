from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1.Model
model = ChatMistralAI(
    model="mistral-small-latest"
)

# 2.Prompt Template 
prompt_template = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 3.Output Parser
parser = StrOutputParser()

# Step-by-step manual flow(no runnable/ no chains)

# Format the prompt 
formatted_prompt = prompt_template.format_messages(topic="Machine Learning")

# Call the model (old style)
response = model.invoke(formatted_prompt)

# Parse the output
final_output = parser.parse(response.content)

print(final_output)