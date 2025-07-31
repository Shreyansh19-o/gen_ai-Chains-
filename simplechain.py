from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


prompt = PromptTemplate(
    template='Generate a 5 interresting facts about {topic}',
    input_variables=['topic']
)

model= ChatGoogleGenerativeAI(
    model="gemini-2.0-flash")

parser=StrOutputParser()

chain=prompt | model | parser
result = chain.invoke({"topic": "Football"})
print(result)
chain.get_graph().print_ascii()  # Print the graph structure of the chain