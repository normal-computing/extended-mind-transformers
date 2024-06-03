import torch
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type

def split_text(text, tokenizer, tokens_per_chunk, chunk_overlap):
    input_ids = tokenizer(text)["input_ids"]
    start_idx = 0
    cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    splits =[]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokens_per_chunk - chunk_overlap
        cur_idx = min(start_idx + tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits

def retrieve_topk(query, keys, k):
    query_embed= torch.tensor(get_embedding(query))
    keys_embed = torch.tensor([get_embedding(chunk) for chunk in keys])

    q_n = query_embed/query_embed.norm(dim=-1).reshape(-1, 1)
    k_n = keys_embed/keys_embed.norm(dim=-1).reshape(-1, 1)

    sim = q_n.matmul(k_n.transpose(0,1))
    if k>sim.size(-1):
        k = sim.size(-1)
    val, idx = torch.topk(sim, k=k, dim=1)
    return idx

def get_chunks(document, question, tokenizer, tokens_per_chunk=500, n_chunks=5, chunk_overlap=0):
    chunks = split_text(document, tokenizer, tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap)
    top_chunks = retrieve_topk(question, chunks, k=n_chunks)
    chunks = [chunks[i] for i in torch.flatten(top_chunks)]
    not_top = [i for i in range(len(chunks)) if i not in top_chunks]
    not_top_chunks = [chunks[i] for i in not_top]
    chunks.reverse()
    return chunks, not_top_chunks


EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

import os 
openai.organization = os.getenv('openai_org') 
openai.api_key = os.getenv('openai_key')

# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]