"""Example of how certain, uncertain generations vary with `k`."""

import torch
from transformers import AutoTokenizer
from src.mpt.configuration import ExtendedMptConfig
from src.mpt.modeling import ExtendedMptForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

WIKI_SNIPPET = """Alexander Grothendieck (/ˈɡroʊtəndiːk/; German pronunciation: [ˌalɛˈksandɐ ˈɡʁoːtn̩ˌdiːk] (listen); French: [ɡʁɔtɛndik]; 28 March 1928 – 13 November 2014) was a stateless (and then, since 1971, French) mathematician who became the leading figure in the creation of modern algebraic geometry.[7][8] His research extended the scope of the field and added elements of commutative algebra, homological algebra, sheaf theory, and category theory to its foundations, while his so-called "relative" perspective led to revolutionary advances in many areas of pure mathematics.[7][9] He is considered by many to be the greatest mathematician of the twentieth century.[10][11]

Grothendieck began his productive and public career as a mathematician in 1949. In 1958, he was appointed a research professor at the Institut des hautes études scientifiques (IHÉS) and remained there until 1970, when, driven by personal and political convictions, he left following a dispute over military funding. He received the Fields Medal in 1966 for advances in algebraic geometry, homological algebra, and K-theory.[12] He later became professor at the University of Montpellier[1] and, while still producing relevant mathematical work, he withdrew from the mathematical community and devoted himself to political and religious pursuits (first Buddhism and later, a more Christian vision).[13] In 1991, he moved to the French village of Lasserre in the Pyrenees, where he lived in seclusion, still working tirelessly on mathematics and his philosophical and religious thoughts until his death in 2014.[14]
"""

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
memory_ids = tokenizer(WIKI_SNIPPET, return_tensors="pt")["input_ids"]
model = ExtendedMptForCausalLM.from_pretrained(
    "mosaicml/mpt-7b", config=ExtendedMptConfig(), external_memories=memory_ids
).to(DEVICE)


# Generation evolves with k
PROMPT = "When did Alexander Grothendieck get his French citizenship?"
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 50, topk=0
)
print("Baseline Generation: ", tokenizer.decode(out[0]))

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 15, topk=5
)
print("Generation for k=5: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 15, topk=6
)
print("Generation for k=6: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 20, topk=7
)
print("Generation for k=7: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 15, topk=8
)
print("Generation for k=8: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 20, topk=30
)
print("Generation for k=30: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())


# Generation stable over k
PROMPT = "What was did Alexander Grothendieck's profession?"
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 25, topk=0
)
print("Baseline Generation: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 15, topk=2
)
print("Generation for k=2: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())

out = model.generate(
    input_ids.to(model.device), max_length=input_ids.size(-1) + 15, topk=8
)
print("Generation for k=8: ", tokenizer.decode(out[0][input_ids.size(-1) :]).strip())
