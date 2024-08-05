from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0,
    # other params...
)

# streaming output example
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
for chunk in llm.stream(messages):
    print(chunk.content)

# one shot output example
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that programs using python",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
response = chain.invoke(
    {
        "input": "Create a python script for printing prime numbers between 1000 and 20000 series",
    }
)

response.pretty_print()
