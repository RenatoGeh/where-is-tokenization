import lm_eval, tqdm, scipy, numpy as np, datasets, matplotlib.pyplot as plt, matplotlib, torch,scipy
import argparse, os, pickle, random, multiprocessing, re, collections
import harness, mdd, utils
from transformers import AutoTokenizer

def MODEL_EQUIVALENCE(name: str) -> str:
    name = name.lower()
    return {"gemma2b": "Gemma", "gemma": "Gemma", "llama2": "Llama2", "mamba": "Mamba"}[name]

def TOK_MAP(path: str) -> AutoTokenizer:
    path = path.lower()
    return AutoTokenizer.from_pretrained("google/gemma-2b" if "gemma" in path else
                                         "/space/renatolg/llama-2-7b-hf" if "llama" in path else
                                         "state-spaces/mamba-130m-hf")

def load_data(task: str, subtask: str = None, mgr = None, include_path: str = None, limit = None):
    if mgr is None: mgr = lm_eval.tasks.TaskManager(include_path=include_path)
    G = mgr.load_task_or_group(task)
    g = G[next(iter(G)) if subtask is None else subtask]
    D = g[1] if isinstance(g, tuple) else g
    D.build_all_requests(limit=limit, rank=0, world_size=1, cache_requests=False,
                         rewrite_requests_cache=False)
    return D

def load_rets(path: str) -> list:
    with open(f"out/rets/{path}_rets.pkl", "rb") as f: return pickle.load(f)

def compute_acc(D: lm_eval.tasks.Task, limit: int, L: list) -> float:
    for i in range(len(L)): D.instances[i].resps = [(L[i], False)]
    D.apply_filters()
    instances_doc_id = collections.defaultdict(list)
    for i in D.instances: instances_doc_id[i.doc_id].append(i)
    for i in instances_doc_id.values(): i.sort(key=lambda x: x.idx)
    hits, _all = 0, 0
    for doc_id, doc in D.doc_iterator(rank=0, limit=limit, world_size=1):
        metrics = D.process_results(doc, [r.filtered_resps['none'] for r in instances_doc_id[doc_id]])
        hits += metrics["acc"]
        _all += 1
    return hits/_all

def get_path(t: str, m: str, n: str, s: str) -> str: return f"{t}_{m}_{n}_{s}_MarLM_llm"

def compute_canonicity(path: str, tokenizer: AutoTokenizer) -> np.ndarray:
    with open(f"out/rets/{path}_samples.pkl", "rb") as f: S = pickle.load(f)
    O = [None for _ in range(len(S))]
    for i, s in enumerate(S):
        o = tokenizer.batch_decode(s)
        assert all(o[0] == x for x in o), f"Invalid tokenizations at {i}!"
        O[i] = o[0]
    C = tokenizer(O, add_special_tokens=False)["input_ids"]
    W = np.empty((len(S), len(S[0])), dtype=np.bool_)
    for i in range(len(S)): W[i,:] = list(map(lambda s: s == C[i], S[i]))
    return W

BATCH_SIZE = 512
def _eval(path: str, task: str, limit: int, nruns: int, W: np.ndarray, i: int, **kwargs) -> np.ndarray:
    D = load_data(task, limit=limit, **kwargs)
    R = load_rets(path)
    LL_P, LL_Q = [r[1] for r in R], [r[2] for r in R]
    I = np.array(tuple(tuple(np.random.choice(LL_P[l].numel(), size=i, replace=False)
                             for _ in range(nruns)) for l in range(len(LL_P))))
    A = np.empty(nruns, dtype=np.float32)
    M = []
    for b in range(0, len(LL_P), BATCH_SIZE):
        k = min(b+BATCH_SIZE, len(LL_P))
        ll_p = np.stack(tuple(np.vstack(tuple(LL_P[l][I[l,j]] for j in range(nruns)))
                              for l in range(b, k)))
        ll_q = np.stack(tuple(np.vstack(tuple(LL_Q[l][I[l,j]] for j in range(nruns)))
                              for l in range(b, k)))
        mar = scipy.special.logsumexp(ll_p-ll_q, axis=-1)-np.log(i)
        M.append(mar)
    M = np.concatenate(M, axis=0)
    P = np.sum(tuple(np.sum(W[j,I[j]], axis=-1)/i for j in range(W.shape[0])), axis=0)/W.shape[0]
    q = np.array(tuple(np.any(W[j,I[j]], axis=-1) for j in range(W.shape[0])))
    Q = np.sum(q, axis=0)/W.shape[0]
    return np.array([compute_acc(D, limit, M[:,j]) for j in range(nruns)]), P, Q, \
        np.array(tuple(np.sum(W[j,I[j]], axis=-1)/i for j in range(W.shape[0])))

def eval_noncanon(path: str, task: str, limit: int, W: np.ndarray, **kwargs) -> np.ndarray:
    D = load_data(task, limit=limit, **kwargs)
    R = load_rets(path)
    M = np.empty(len(R), dtype=np.float32)
    LL_P, LL_Q = [r[1] for r in R], [r[2] for r in R]
    U = np.bitwise_not(W)
    for i in range(len(R)):
        if np.all(U[i]):
            M[i] = LL_C[i]
        else:
            ll_p, ll_q = LL_P[i][U[i]], LL_Q[i][U[i]]
            M[i] = scipy.special.logsumexp(ll_p-ll_q, axis=-1)-np.log(len(ll_p))
    return compute_acc(D, limit, M)

def evaluate_standalone_parallel(task: str, model: str, limit: str, samples: str, nsamples: int,
                                 nruns: int, output_path: str, **kwargs):
    path = get_path(task, model, limit, samples)
    W = compute_canonicity(path, TOK_MAP(path))
    fun = utils.partial(_eval, path, task, limit, nruns, W, **kwargs)
    if os.path.isfile(f"out/csv/{path}_checkpoint.pkl") and not OVERRIDE:
        with open(f"out/csv/{path}_checkpoint.pkl", "rb") as f: K = pickle.load(f)
    else:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            R = [pool.apply_async(fun, (i,)) for i in range(1, nsamples+1)]
        # R = [fun(i) for i in range(1, nsamples+1)]
            K = [r.get() for r in tqdm.tqdm(R, desc="Computing accuracy")]
        # K = utils.parallelize(fun, range(1, nsamples+1), desc="Computing accuracy")
        # K = list(map(fun, range(1, nsamples+1)))
        with open(f"out/csv/{path}_checkpoint.pkl", "wb") as f: pickle.dump(K, f)
    scores, canons = np.empty((nsamples, nruns), dtype=np.float32), np.empty((nsamples, nruns), dtype=np.float32)
    anyc = np.empty((nsamples, nruns), dtype=np.float32)
    C = [None for _ in range(nsamples)]
    for i in tqdm.tqdm(range(nsamples), desc="Recovering buffers"):
        scores[i,:], canons[i,:], anyc[i,:], C[i] = K[i]
    C = np.array(C)
    with open(f"{output_path}/{MODEL_EQUIVALENCE(model)}_{task}_canon_ll.pkl", "rb") as f: LL_C = pickle.load(f)
    none_canon = eval_noncanon(path, task, limit, W, **kwargs)
    with open(f"{output_path}/{path}_all.pkl", "wb") as f: pickle.dump((scores, canons, none_canon), f)
    return data2csv(scores, f"{output_path}/{path}.csv"), data2csv(canons, f"{output_path}/{path}_canons.csv")

def evaluate_standalone(task: str, model: str, limit: str, samples: str, nsamples: int, nruns: int,
                        output_path: str, **kwargs):
    D = load_data(task, limit=limit, **kwargs)
    path = get_path(task, model, limit, samples)
    with open(f"out/rets/{path}_rets.pkl", "rb") as f:
        R = pickle.load(f)
    LL_P, LL_Q = [r[1] for r in R], [r[2] for r in R]
    scores = np.empty((nsamples, nruns), dtype=np.float32)
    for i in tqdm.tqdm(range(1, nsamples+1), desc="Subset size", position=0):
        ll_p = np.stack(tuple(np.vstack(tuple(np.random.choice(LL_P[l], size=i, replace=False)
                                              for _ in range(nruns))) for l in range(len(R))))
        ll_q = np.stack(tuple(np.vstack(tuple(np.random.choice(LL_Q[l], size=i, replace=False)
                                              for _ in range(nruns))) for l in range(len(R))))
        mar = scipy.special.logsumexp(ll_p-ll_q, axis=-1)-np.log(i)
        for j in tqdm.tqdm(range(nruns), desc="Run", position=1, leave=False):
            scores[i-1,j] = compute_acc(D, mar[:,j])
    return data2csv(scores, f"{output_path}/{path}.csv")

def data2csv(scores: np.ndarray, output_path: str):
    import pandas as pd
    mu, sigma = np.mean(scores, axis=-1), np.std(scores, axis=-1)
    D = pd.DataFrame({"x": range(1, scores.shape[0]+1), "mean": mu, "stdev": sigma})
    D.to_csv(output_path, index=False)
    return D

if __name__ == "__main__":
    random.seed(101); np.random.seed(101)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-l", "--limit", type=int, required=True)
    parser.add_argument("-s", "--model-samples", type=int, required=True)
    parser.add_argument("-n", "--samples", type=int, required=False, default=None)
    parser.add_argument("-o", "--output-path", type=str, required=False, default=None)
    parser.add_argument("-st", "--subtask", type=str, required=False, default=None)
    parser.add_argument("-r", "--runs", type=int, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=False, default=512)
    parser.add_argument("-ov", "--override", action=argparse.BooleanOptionalAction)
    args = vars(parser.parse_args())
    if args["samples"] is None: args["samples"] = args["model_samples"]
    BATCH_SIZE = args["batch_size"]
    OVERRIDE = args["override"]
    retval = evaluate_standalone_parallel(args["task"], args["model"], args["limit"], args["model_samples"],
                                 args["samples"], args["runs"], args["output_path"],
                                 subtask=args["subtask"], include_path="/space/renatolg/tokens/")
