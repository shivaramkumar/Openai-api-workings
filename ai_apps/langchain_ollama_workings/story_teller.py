from core import get_llm, get_results, get_template_from_prompt

prompt = get_template_from_prompt(
    "Complete a {length} story using the given beginning."
    + "The genre should be {genre} and the story should have an apt ending. Beginning: {text}"
)
llm = get_llm()

print(
    get_results(
        llm=llm, prompt=prompt, input_params={"length": "short", "genre": "horror", "text": "Once there was a coder"}
    )
)
print(
    get_results(
        llm=llm, prompt=prompt, input_params={"length": "short", "genre": "rom-com", "text": "And the Queen died"}
    )
)
