import torch
from torch.distributions import Categorical

@torch.no_grad()
def augmented_generate(model, tokenizer, input_tokens, default_k=1, step_k=2, entropy_thresh=1.0, n_tokens=30, dynamic_topk=True, verbose=True): 
    if dynamic_topk:
        default_k = max(min(round(len(model.memory_ids[0])/500), 12), 4) // 2
        step_k = max(min(round(len(model.memory_ids[0])/500), 12), 4) // 3
        if verbose:
            print(f"Default k: {default_k}, Step k: {step_k}, Memory Len: {len(model.memory_ids[0])}")

    out = model(input_tokens.to(model.device), use_cache=True, topk=default_k)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(-1).unsqueeze(-1)
    seq = torch.concat([input_tokens.to(model.device), next_token], dim=-1)
    
    entropies = []
    c=0
    for _ in range(n_tokens):
        out = model(next_token, past_key_values=past_key_values, use_cache=True, topk=default_k)
        entropy = Categorical(logits=out.logits[:, -1, :], validate_args=False).entropy()

        step_entropy = None
        if entropy > entropy_thresh:
            out = model(next_token, past_key_values=past_key_values, use_cache=True, topk=step_k+default_k)
            step_entropy = Categorical(logits=out.logits[:, -1, :], validate_args=False).entropy()
            c+=1

        next_token = out.logits[:, -1, :].argmax(-1).unsqueeze(-1)
        past_key_values = out.past_key_values
        seq = torch.concat([seq, next_token], dim=-1)
        entropies.append([entropy, step_entropy, tokenizer.batch_decode(next_token)])
    
    if verbose:
        print(f'{c} tokens regenerated')

    return tokenizer.batch_decode(seq)[0], out, entropies