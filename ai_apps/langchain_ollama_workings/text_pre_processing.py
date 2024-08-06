from core import get_llm, get_results, get_template_from_prompt

prompt = get_template_from_prompt(
    "Preprocess the given text by following the given steps in sequence."
    + "Follow only those steps that have a yes against them."
    + "Remove Number:{number},Remove punctuations : {punc} ,Word stemming : {stem}."
    + "Output just the preprocessed text. Text : {text}"
)

llm = get_llm(model="llama3")

print(
    get_results(
        llm=llm,
        prompt=prompt,
        input_params={"text": "Hey!! I got 12 out of 20 in Swimming", "number": "yes", "punc": "yes", "stem": "no"},
    )
)

print(
    get_results(
        llm=llm,
        prompt=prompt,
        input_params={
            "text": "22 13B is my flat no. Rohit will be joining us for the party",
            "number": "yes",
            "punc": "no",
            "stem": "yes",
        },
    )
)
