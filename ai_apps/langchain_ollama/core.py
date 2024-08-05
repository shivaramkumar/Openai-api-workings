from string import Template

from langchain.prompts import PromptTemplate

from langchain_ollama import OllamaLLM


def get_template_from_prompt(prompt: Template) -> PromptTemplate:
    return PromptTemplate.from_template(prompt)


def get_llm(model="llama3") -> OllamaLLM:
    return OllamaLLM(model=model)


def get_results(llm: OllamaLLM, prompt: PromptTemplate, input_params: dict) -> str:
    chain = prompt | llm
    return chain.invoke(input_params)
