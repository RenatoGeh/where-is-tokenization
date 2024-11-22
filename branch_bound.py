import mdd, utils, models, torch, pickle, time, tqdm, gc

MODELS = {"Gemma2B": models.Gemma, "Mamba": models.Mamba, "Llama2": models.Llama2}
BUDGET = 60*60
all_top_k, all_elapsed_time = {m: None for m in MODELS}, {m: None for m in MODELS}
for mdl in MODELS:
    M = MODELS[mdl](device_map=0, dtype=torch.float32)
    S = "Language models typically tokenize text into subwords, utilizing a learned deterministic set of merge rules to aggregate tokens. A common assumption is that such a tokenization is unique: there is a one-to-one correspondence between a given text and its tokenization, and therefore, the probability of a tokenization suffices for the probability of the text it represents. In this paper, we address this misconception, showing that the task of computing the probability of an LLM generating a given text is computationally hard as the length of the text increases. This has significant implications for generation tasks, where we are interested in the probability of a completion to a context text. There the standard practice of using a single tokenization does not accurately reflect the LLM's answer. We leverage sampling-based approaches to approximate the LLM's text probability, thereby achieving a consistent improvement across a set of generation benchmarks and models including Llama2, Gemma and Mamba."
    W = S.split(sep=' ')
    #S = "The quick brown fox jumped over the lazy dog"
    top_k, elapsed_time = [], []
    for i in tqdm.tqdm(range(len(W))):
        s = ' '.join(W[:i+1])
        d = mdd.build_mdd(M.tokenizer, s)
        K, t = d.bb(M.tokenizer, M.model, bound_tok=M.tokenizer.encode(s, add_special_tokens=True),
                           time_budget=BUDGET)
        top_k.append(K)
        elapsed_time.append(BUDGET-t)
        if t <= 0: break
    all_top_k[mdl] = top_k
    all_elapsed_time[mdl] = elapsed_time
    with open(f"out/branch_bound/{mdl}_top_k.pkl", "wb") as f: pickle.dump(top_k, f)
    with open(f"out/branch_bound/{mdl}_time.pkl", "wb") as f: pickle.dump(elapsed_time, f)
    del M; gc.collect(); torch.cuda.empty_cache()
