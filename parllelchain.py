from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda # Import RunnableLambda
import os

load_dotenv()

# Initialize HuggingFace Endpoint for model2
# Ensure your Hugging Face API token is set as an environment variable (e.g., HF_TOKEN)
# export HF_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

# Wrap HuggingFaceEndpoint with ChatHuggingFace
model2 = ChatHuggingFace(llm=llm)

# Initialize ChatGoogleGenerativeAI for model1
# Ensure your Google API Key is set as an environment variable (e.g., GOOGLE_API_KEY)
# export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
model1 = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash'
)

# Define prompt templates for different tasks
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# Create individual chains for each task in parallel
# Use RunnableLambda to extract the 'text' value from the input dictionary
# and pass it to the prompt. The prompt then creates messages for the LLM.
parallel_chain = RunnableParallel({
    'notes': (RunnableLambda(lambda x: x['text']) | prompt1 | model1 | parser),
    'quiz': (RunnableLambda(lambda x: x['text']) | prompt2 | model2 | parser)
})

# Create the merge chain
# The merge_chain will receive a dictionary from parallel_chain with 'notes' and 'quiz' keys.
# This dictionary is then passed to prompt3, which formats it into messages for model1.
merge_chain = (prompt3 | model1 | parser)

# Combine the parallel and merge chains
chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# Invoke the main chain with the input text
result = chain.invoke({'text':text})

print(result)

# Print the ASCII graph of the chain
chain.get_graph().print_ascii()
