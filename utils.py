from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, tqdm, numpy as np
import math, multiprocessing, functools

@torch.no_grad()
def loglikelihood_from_tensor(model: AutoModelForCausalLM, X: torch.Tensor) -> float:
    return torch.sum(model(X).logits[0].log_softmax(dim=-1).gather(1, X.reshape(-1, 1))).item()

@torch.no_grad()
def loglikelihood(model: AutoModelForCausalLM, I: list, prefix: int = [2],
                  prefix_cutoff: bool = False) -> float:
    x = [I] if len(prefix) == 0 else [prefix+I]
    X = torch.tensor(x, dtype=int).to(model.device)
    return torch.sum(model(X).logits[0,:-1].log_softmax(dim=-1).gather(1, X[:,1:].reshape(-1, 1))[prefix_cutoff*len(prefix):]).item()

@torch.no_grad()
def loglikelihood_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, I: list,
                        use_cache: bool = True, prefix: int = [2],
                        prefix_cutoff: bool = False) -> torch.Tensor:
    if len(prefix) == 0: x, prefix_len = I, np.zeros(len(I), dtype=int)
    elif isinstance(prefix[0], list):
        x, prefix_len = [prefix[i]+I[i] for i in range(len(I))], list(map(len, prefix))
    else: x, prefix_len = [prefix+i for i in I], np.full(len(I), len(prefix), dtype=int)
    X = tokenizer.prepare_for_model(x, padding=True, return_tensors="pt").to(model.device)
    J = X["input_ids"].reshape(len(I), -1, 1)[:,1:]
    P = model(**X, use_cache=use_cache).logits[:,:-1].log_softmax(dim=-1).gather(-1, J)
    indices = list(map(len, I)) if prefix_cutoff else [len(I[i])+prefix_len[i] for i in range(len(I))]
    return np.array([torch.sum(x[-i:]).item() for x, i in zip(P, indices)])

def rhloglikelihood(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list,
                    batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False,
                    online_memory: bool = True, use_tqdm: bool = False, **kwargs):
    if len(prefix) > 0:
        if isinstance(prefix[0], list):
            in_texts = [prefix[i]+texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = list(map(len, prefix))
        else:
            in_texts = [prefix+texts[i] for i in range(len(texts))]
            if prefix_cutoff: cutoff = np.full(len(in_texts), len(prefix), dtype=np.int32)
        if not prefix_cutoff: cutoff = np.zeros(len(in_texts), dtype=np.int32)
    else:
        in_texts = texts
        cutoff = np.ones(len(in_texts), dtype=np.int32)
    text_len = list(map(len, in_texts))
    d = max(text_len)
    torch.cuda.empty_cache()
    _rng = range(0, len(in_texts), batch_size)
    rng = tqdm.tqdm(_rng, desc="Computing log-likelihood") if use_tqdm else _rng
    skip = int(True)#not is_mamba_tokenizer(tokenizer))

    if online_memory:
        n = len(in_texts)
        lls = []
        with torch.no_grad():
            T = torch.zeros((batch_size, d), dtype=torch.int32).to(model.device)
            M = torch.zeros((batch_size, d), dtype=torch.int32).to(model.device)
            for batch_idx in rng:
                batch_size_ = min(batch_size, n - batch_idx)
                T_b, M_b = T[:batch_size_,:], M[:batch_size_,:]
                for i in range(batch_size_):
                    l = len(in_texts[batch_idx+i])
                    T_b[i,:l], T_b[i,l:] = torch.tensor(in_texts[batch_idx+i]), tokenizer.pad_token_id
                    M_b[i,:l], M_b[i,l:] = 1, 0
                # Get logits up to the last token (which would be the suffix) unless skip == 0,
                # in which case returns the whole thing.
                logits = model.forward(input_ids=T_b, attention_mask=M_b).logits[:,:-skip or None,:]
                log_probs = torch.log_softmax(logits, dim=-1)
                log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                      torch.arange(0, d-skip).unsqueeze(0), T_b[:,skip:]]
                log_probs *= M_b[:,skip:]
                lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx+i]-1:text_len[batch_idx+i]-1], dim=-1).reshape(1)
                                       for i in range(log_probs.shape[0])), dim=0)
                lls.append(lls_)

        lls = torch.cat(lls, dim=0)
        return lls.to("cpu")

    T = torch.full((len(texts), d), tokenizer.pad_token_id, dtype=torch.int32).to(model.device)
    M = torch.zeros((len(texts), d), dtype=torch.int32).to(model.device)
    for i in range(len(text_len)):
        T[i,:len(in_texts[i])] = torch.tensor(in_texts[i])
        M[i,:len(in_texts[i])] = 1

    n = T.shape[0]
    lls = []
    with torch.no_grad():
        for batch_idx in rng:
            batch_size_ = min(batch_size, n - batch_idx)
            input_ids_ = T[batch_idx:batch_idx+batch_size_]
            attention_mask_ = M[batch_idx:batch_idx+batch_size_]
            logits = model.forward(input_ids=input_ids_,
                                   attention_mask=attention_mask_).logits[:,:-skip or None,:]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                  torch.arange(0, d-skip).unsqueeze(0), input_ids_[:,skip:]]
            log_probs *= attention_mask_[:,skip:]
            lls_ = torch.cat(tuple(torch.sum(log_probs[i][cutoff[batch_idx+i]-1:text_len[batch_idx+i]-1], dim=-1).reshape(1)
                                   for i in range(log_probs.shape[0])), dim=0)
            lls.append(lls_)

    lls = torch.cat(lls, dim=0)

    return lls.to("cpu")

def hloglikelihood(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, texts: list,
                   batch_size: int = 16, prefix: list = [], prefix_cutoff: bool = False):
    if len(prefix) > 0:
        if isinstance(prefix[0], list): in_texts = [prefix[i]+texts[i] for i in range(len(texts))]
        else: in_texts = [prefix+texts[i] for i in range(len(texts))]
    else: in_texts = texts
    suffix_len = list(map(len, texts)) if prefix_cutoff else np.zeros(len(texts), dtype=int)
    inputs = tokenizer.prepare_for_model(in_texts, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    torch.cuda.empty_cache()

    n, d = input_ids.shape
    lls = []
    with torch.no_grad():
        for batch_idx in range(0, n, batch_size):
            batch_size_ = min(batch_size, n - batch_idx)
            input_ids_ = input_ids[batch_idx:batch_idx+batch_size_]
            attention_mask_ = attention_mask[batch_idx:batch_idx+batch_size_]
            logits = model.forward(input_ids=input_ids_, attention_mask=attention_mask_).logits[:,:-1,:]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs[torch.arange(0, batch_size_).unsqueeze(-1),
                                  torch.arange(0, d-1).unsqueeze(0), input_ids_[:,1:]]
            log_probs *= attention_mask_[:,1:]
            lls_ = torch.cat(tuple(torch.sum(log_probs[i][-suffix_len[i]:], dim=-1).reshape(1)
                                   for i in range(log_probs.shape[0])), dim=0)
            lls.append(lls_)

    lls = torch.cat(lls, dim=0)

    return lls.to("cpu")

@torch.no_grad()
def logits(model: AutoModelForCausalLM, I: list, prefix: int = [2], use_cache: bool = True) -> torch.Tensor:
    if prefix is None: X = torch.tensor([I], dtype=int).to(model.device)
    else: X = torch.tensor([prefix+I], dtype=int).to(model.device)
    return model(X, use_cache=use_cache).logits[0,:-1].log_softmax(dim=-1).gather(-1, X[:,1:].reshape(-1, 1)).flatten()

@torch.no_grad()
def logits_softmax(model: AutoModelForCausalLM, prefix: list, I: list, use_cache: bool = True) -> torch.Tensor:
    "Logits of the next token, softmax'ed."
    X = torch.tensor(prefix, dtype=int).to(model.device)
    return model(X, use_cache=use_cache).logits[0,-1,I].log_softmax(dim=-1).to("cpu")

@torch.no_grad()
def logits_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, Y: list,
                 use_cache: bool = True, prefix: int = [2]) -> torch.Tensor:
    x = Y if prefix is None else [prefix+x for x in Y]
    X = tokenizer.prepare_for_model(x, padding=True, return_tensors="pt").to(model.device)
    n = max(map(len, Y))
    O = [n-len(i) for i in Y]
    I = X["input_ids"].reshape(len(Y), -1, 1)[:,1:]
    L = model(**X, use_cache=use_cache).logits[:,:-1].log_softmax(dim=-1).gather(-1, I)
    return tuple(L[i][o:] for i, o in enumerate(O))

@torch.no_grad()
def perplexity(model: AutoModelForCausalLM, I: list) -> float:
    return math.exp(-loglikelihood(model, I)/len(I))

@torch.no_grad()
def perplexity_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, I: list,
                     use_cache: bool = True, prefix: int = [2]) -> torch.Tensor:
    X = tokenizer.prepare_for_model([prefix+x for x in I], padding=True,
                                    return_tensors="pt").to(model.device)
    n = max(map(len, I))
    O = [n-len(i) for i in I]
    J = X["input_ids"].reshape(len(I), -1, 1)[:,1:]
    P = model(**X, use_cache=use_cache).logits[:,:-1].log_softmax(dim=-1).gather(-1, J)
    return np.array([torch.sum(x[o:]).item()/(n-o) for x, o in zip(P, O)])

@torch.no_grad()
def generate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: torch.Tensor,
             max_new_tokens: int = 50) -> (str, float):
    out = model.generate(prompt, do_sample=True, top_k=0, top_p=1, output_scores=True,
                         return_dict_in_generate=True, renormalize_logits=True,
                         max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    scores = torch.cat(out.scores)
    n = scores.shape[0]
    return tokenizer.decode(out.sequences[0]), \
        torch.sum(torch.gather(scores, 1, out.sequences[:,-n:].reshape(-1, 1))).item()

def generate_toks(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, n: int, top_p: float = 1.0,
                  **kwargs):
    out = model.generate(do_sample=True, top_k=0, top_p=top_p, temperature=1.0,
                         num_return_sequences=n, pad_token_id=tokenizer.pad_token_id, **kwargs)
    return out

def generate_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, I: list, k: int = 50) -> (str, float):
    X = tokenizer.prepare_for_model(I, padding=True, return_tensors="pt").to("cuda")
    out = model.generate(**X, do_sample=True, top_k=0, top_p=1, output_scores=True,
                         temperature=1.0, return_dict_in_generate=True, renormalize_logits=True,
                         max_new_tokens=k)
    scores = torch.cat(out.scores).reshape(len(I), k, -1)
    idx = out.sequences[:,-k:]
    n = scores.shape[0]
    return tokenizer.batch_decode(out.sequences[0], skip_special_tokens=True), \
        torch.sum(torch.gather(scores, 1, out.sequences[:,-n:].reshape(-1, 1))).item()

def partial(func, *args, **kwargs): return functools.partial(func, *args, **kwargs)
def parallelize(func, X: list, use_tqdm: bool = True, cpu_count: int = multiprocessing.cpu_count(),
                **kwargs) -> list:
    with multiprocessing.Pool(cpu_count) as pool:
        c = max(1, math.ceil(len(X)//multiprocessing.cpu_count()))
        Y = list(tqdm.tqdm(pool.imap(func, X, chunksize=c), total=len(X), **kwargs)) if use_tqdm \
            else pool.map(func, X, chunksize=c)
    return Y
def iparallelize(func, X: iter, use_tqdm: bool = True, total: int = None,
                 cpu_count: int = multiprocessing.cpu_count(), **kwargs) -> list:
    with multiprocessing.Pool(cpu_count) as pool:
        c = 1 if total is None else max(1, math.ceil(total//multiprocessing.cpu_count()))
        Y = list(tqdm.tqdm(pool.imap(func, X, chunksize=c), total=total, **kwargs)) if use_tqdm \
            else pool.imap(func, X, chunksize=c)
    return Y
def parallelize_single(func, X: list, use_tqdm: bool = True,
                       cpu_count: int = multiprocessing.cpu_count(), **kwargs) -> list:
    with multiprocessing.Pool(cpu_count) as pool:
        P = [pool.apply_async(func, (x,)) for x in X]
        R = [p.get() for p in (tqdm.tqdm(P, **kwargs) if use_tqdm else P)]
    return R

def sample_logspace(L: np.ndarray, n: int = 1) -> np.ndarray:
    "Use the Gumbel-max trick to sample without going back to arithmetic space."
    return np.argmax(L + np.random.gumbel(loc=0, scale=1, size=(n, L.shape[0])), axis=1)

def remove_pad(T: torch.Tensor, pad_token_id: int) -> list:
    I = torch.argmax((T == pad_token_id)*torch.arange(T.shape[1], 0, -1), 1, keepdim=True)
    toks = [T[i,:I[i].item()-(I[i,0].item() == 0)] for i in range(len(I))]
    return toks

def batch_slices(n: int, m: int) -> list:
    k = m//n
    return [slice(i-k, i) for i in range(k, m+k, k)]

def _tok_name(tokenizer: AutoTokenizer) -> str:
    return ''.join(c if c.isalpha() or c.isdigit() else '_' for c in tokenizer.name_or_path).rstrip()
def is_llama_tokenizer(tokenizer: AutoTokenizer) -> bool:
    return "llama" in _tok_name(tokenizer).lower()
def is_gemma_tokenizer(tokenizer: AutoTokenizer) -> bool: return "gemma" in _tok_name(tokenizer).lower()
def is_mamba_tokenizer(tokenizer: AutoTokenizer) -> bool: return "mamba" in _tok_name(tokenizer).lower()

@torch.compile
def matmul_log(A, B):
    A_max, B_max = torch.amax(A, dim=-1, keepdim=True), torch.amax(B, dim=len(B.shape)-2, keepdim=True)
    A -= A_max; B -= B_max
    A.exp_(); B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max + B_max)
    return C

@torch.compile
def matmul_loga_b(A, B):
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A -= A_max
    A.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max)
    return C

@torch.compile
def matmul_a_logb(A, B):
    B_max = torch.amax(B, dim=len(B.shape)-2, keepdim=True)
    B -= B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)
    return C
