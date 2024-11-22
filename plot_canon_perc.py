import pickle, os, gc, argparse
import transformers, torch, tqdm, Levenshtein, numpy as np
import matplotlib.pyplot as plt, plotly.express, plotly.graph_objects, plotly.subplots, pandas as pd
import models, utils

MODELS = {
    "Llama_4": utils.partial(models.Llama, load_in_4bit=True, dtype=torch.float32),
    "Llama_8": utils.partial(models.Llama, load_in_8bit=True, dtype=torch.float32),
    "Llama_16": utils.partial(models.Llama, dtype=torch.float16),
    "Llama_32": utils.partial(models.Llama, dtype=torch.float32),
    "Llama2_4": utils.partial(models.Llama2, load_in_4bit=True, dtype=torch.float32),
    "Llama2_8": utils.partial(models.Llama2, load_in_8bit=True, dtype=torch.float32),
    "Llama2_16": utils.partial(models.Llama2, dtype=torch.float16),
    "Llama2_32": utils.partial(models.Llama2, dtype=torch.float32),
    "Llama2Chat_4": utils.partial(models.Llama2Chat, load_in_4bit=True, dtype=torch.float32),
    "Llama2Chat_8": utils.partial(models.Llama2Chat, load_in_8bit=True, dtype=torch.float32),
    "Llama2Chat_16": utils.partial(models.Llama2Chat, dtype=torch.float16),
    "Llama2Chat_32": utils.partial(models.Llama2Chat, dtype=torch.float32),
    "Gemma_2b_4": utils.partial(models.Gemma, "2b", load_in_4bit=True, dtype=torch.float32),
    "Gemma_2b_8": utils.partial(models.Gemma, "2b", load_in_8bit=True, dtype=torch.float32),
    "Gemma_2b_16": utils.partial(models.Gemma, "2b", dtype=torch.float16),
    "Gemma_2b_32": utils.partial(models.Gemma, "2b", dtype=torch.float32),
    "Gemma_7b_4": utils.partial(models.Gemma, "7b", load_in_4bit=True, dtype=torch.float32),
    "Gemma_7b_8": utils.partial(models.Gemma, "7b", load_in_8bit=True, dtype=torch.float32),
    "Gemma_7b_16": utils.partial(models.Gemma, "7b", dtype=torch.float16),
    "Gemma_7b_32": utils.partial(models.Gemma, "7b", dtype=torch.float32),
    "Gemma_2b_it_4": utils.partial(models.Gemma, "2b-it", load_in_4bit=True, dtype=torch.float32),
    "Gemma_2b_it_8": utils.partial(models.Gemma, "2b-it", load_in_8bit=True, dtype=torch.float32),
    "Gemma_2b_it_16": utils.partial(models.Gemma, "2b-it", dtype=torch.float16),
    "Gemma_2b_it_32": utils.partial(models.Gemma, "2b-it", dtype=torch.float32),
    "Gemma_7b_it_4": utils.partial(models.Gemma, "7b-it", load_in_4bit=True, dtype=torch.float32),
    "Gemma_7b_it_8": utils.partial(models.Gemma, "7b-it", load_in_8bit=True, dtype=torch.float32),
    "Gemma_7b_it_16": utils.partial(models.Gemma, "7b-it", dtype=torch.float16),
    "Gemma_7b_it_32": utils.partial(models.Gemma, "7b-it", dtype=torch.float32),
    "GPT2_small": utils.partial(models.GPT2, "small"),
    "GPT2_medium": utils.partial(models.GPT2, "medium"),
    "GPT2_large": utils.partial(models.GPT2, "large"),
    "GPT2_xl": utils.partial(models.GPT2, "xl"),
    "Mistral_4": utils.partial(models.Mistral, load_in_4bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mistral_8": utils.partial(models.Mistral, load_in_8bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mistral_16": utils.partial(models.Mistral, dtype=torch.float16, tok_args={"padding_side": "left"}),
    "Mistral_32": utils.partial(models.Mistral, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mistral_it_4": utils.partial(models.Mistral, "it", load_in_4bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mistral_it_8": utils.partial(models.Mistral, "it", load_in_8bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mistral_it_16": utils.partial(models.Mistral, "it", dtype=torch.float16, tok_args={"padding_side": "left"}),
    "Mistral_it_32": utils.partial(models.Mistral, "it", dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mamba_4": utils.partial(models.Mamba, load_in_4bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mamba_8": utils.partial(models.Mamba, load_in_8bit=True, dtype=torch.float32, tok_args={"padding_side": "left"}),
    "Mamba_16": utils.partial(models.Mamba, dtype=torch.float16, tok_args={"padding_side": "left"}),
    "Mamba_32": utils.partial(models.Mamba, dtype=torch.float32, tok_args={"padding_side": "left"}),
}

TOKENIZERS = {
    "Llama_4": models.Llama.get_tokenizer(),
    "Llama_8": models.Llama.get_tokenizer(),
    "Llama_16": models.Llama.get_tokenizer(),
    "Llama_32": models.Llama.get_tokenizer(),
    "Llama2_4": models.Llama2.get_tokenizer(),
    "Llama2_8": models.Llama2.get_tokenizer(),
    "Llama2_16": models.Llama2.get_tokenizer(),
    "Llama2_32": models.Llama2.get_tokenizer(),
    "Llama2Chat_4": models.Llama2Chat.get_tokenizer(),
    "Llama2Chat_8": models.Llama2Chat.get_tokenizer(),
    "Llama2Chat_16": models.Llama2Chat.get_tokenizer(),
    "Llama2Chat_32": models.Llama2Chat.get_tokenizer(),
    "Gemma_2b_4": models.Gemma.get_tokenizer("2b"),
    "Gemma_2b_8": models.Gemma.get_tokenizer("2b"),
    "Gemma_2b_16": models.Gemma.get_tokenizer("2b"),
    "Gemma_2b_32": models.Gemma.get_tokenizer("2b"),
    "Gemma_7b_4": models.Gemma.get_tokenizer("7b"),
    "Gemma_7b_8": models.Gemma.get_tokenizer("7b"),
    "Gemma_7b_16": models.Gemma.get_tokenizer("7b"),
    "Gemma_7b_32": models.Gemma.get_tokenizer("7b"),
    "Gemma_2b_it_4": models.Gemma.get_tokenizer("2b-it"),
    "Gemma_2b_it_8": models.Gemma.get_tokenizer("2b-it"),
    "Gemma_2b_it_16": models.Gemma.get_tokenizer("2b-it"),
    "Gemma_2b_it_32": models.Gemma.get_tokenizer("2b-it"),
    "Gemma_7b_it_4": models.Gemma.get_tokenizer("7b-it"),
    "Gemma_7b_it_8": models.Gemma.get_tokenizer("7b-it"),
    "Gemma_7b_it_16": models.Gemma.get_tokenizer("7b-it"),
    "Gemma_7b_it_32": models.Gemma.get_tokenizer("7b-it"),
    "GPT2_small": models.GPT2.get_tokenizer("small"),
    "GPT2_medium": models.GPT2.get_tokenizer("medium"),
    "GPT2_large": models.GPT2.get_tokenizer("large"),
    "GPT2_xl": models.GPT2.get_tokenizer("xl"),
    "Mistral_4": models.Mistral.get_tokenizer(padding_side="left"),
    "Mistral_8": models.Mistral.get_tokenizer(padding_side="left"),
    "Mistral_16": models.Mistral.get_tokenizer(padding_side="left"),
    "Mistral_32": models.Mistral.get_tokenizer(padding_side="left"),
    "Mistral_it_4": models.Mistral.get_tokenizer("it", padding_side="left"),
    "Mistral_it_8": models.Mistral.get_tokenizer("it", padding_side="left"),
    "Mistral_it_16": models.Mistral.get_tokenizer("it", padding_side="left"),
    "Mistral_it_32": models.Mistral.get_tokenizer("it", padding_side="left"),
    "Mamba_4": models.Mamba.get_tokenizer(padding_side="left"),
    "Mamba_8": models.Mamba.get_tokenizer(padding_side="left"),
    "Mamba_16": models.Mamba.get_tokenizer(padding_side="left"),
    "Mamba_32": models.Mamba.get_tokenizer(padding_side="left"),
}

def sample(model: models.Model, name: str, n: int, b: int, max_new_tokens: int, p_step) -> torch.Tensor:
    T = [[] for _ in np.arange(0, 1.0+p_step, p_step)]
    pbar = tqdm.tqdm(total=int(1.0/p_step)*n, desc="Sampling", leave=False)
    for j, p in enumerate(np.arange(0, 1.0+p_step, p_step)):
        pbar.set_description(f"Sampling (p={p:.2f})")
        for i in range(0, n, b):
            X = utils.generate_toks(model.model, model.tokenizer, b, top_p=p, max_new_tokens=max_new_tokens).to("cpu")
            X = utils.remove_pad(X, model.tokenizer.pad_token_id)
            T[j].extend(X)
            pbar.update(min(b, n-i))
    pbar.close()
    with open(f"tmp/plot_canon_perc/{name}{APPEND}_samples.pkl", "wb") as f: pickle.dump(T, f)
    return T

def statistics(T: list, name: str, tokenizer: transformers.AutoTokenizer) -> float:
    n, m = len(T), len(T[0])
    num_canons, dists = np.zeros(n, dtype=np.int32), np.empty((n, m), dtype=np.float32)
    invalids = np.zeros(n, dtype=np.int32)
    for i in tqdm.tqdm(range(n), desc="Distances", leave=False):
        texts = tokenizer.batch_decode([x[1:] if x[0] == tokenizer.bos_token_id else x for x in T[i]])
        K = tokenizer(texts, add_special_tokens=True)["input_ids"]
        W = [x == y.tolist() for x, y in zip(K, T[i])]
        num_canons[i] = sum(W)
        invalids[i] = sum([x != y for x, y in zip(texts, tokenizer.batch_decode([x[1:] for x in K]))])
        dists[i,:] = [Levenshtein.distance(x, y) for x, y in zip(K, T[i])]
    perc = (num_canons/m)*100
    invalids = (invalids/m)*100
    with open(f"tmp/plot_canon_perc/{name}{APPEND}_statistics.pkl", "wb") as f: pickle.dump((perc, dists, invalids), f)

def run(mdls: list, names: list, n: int, b: int, max_new_tokens: int, p_step: float):
    pbar, m = tqdm.tqdm(mdls, leave=False), len(mdls)
    T = [None for _ in range(m)]
    stats = [None for _ in range(m)]
    tokenizers = [None for _ in range(m)]
    for i, M in enumerate(pbar):
        path = f"tmp/plot_canon_perc/{names[i]}{APPEND}_samples.pkl"
        if os.path.isfile(path):
            with open(path, "rb") as f: T[i] = pickle.load(f)
            tokenizers[i] = TOKENIZERS[names[i]]
        else:
            model = M(device_map=0)
            tokenizers[i] = model.tokenizer
            pbar.set_description(f"Sampling for {names[i]}")
            T[i] = sample(model, names[i], n, b, max_new_tokens, p_step)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    for i, _ in enumerate(pbar):
        pbar.set_description(f"Computing statistics for {names[i]}")
        stats[i] = statistics(T[i], names[i], tokenizers[i])

    return T, stats, names, tokenizers

def by_length():
    M = ["Llama2", "Gemma2B", "Mamba"]
    paths = {"Llama2": "Llama2", "Gemma2B": "Gemma_2b", "Mamba": "Mamba"}
    TOKS = {m: transformers.AutoTokenizer.from_pretrained(k) for m, k in zip(["Llama2", "Gemma2B", "Mamba"],
                                                                             ["/space/renatolg/llama-2-7b-hf", "google/gemma-2b", "state-spaces/mamba-130m-hf"])}
    n = 128
    P = {m: np.empty(n, dtype=np.float32) for m in M}
    bar = tqdm.tqdm(total=len(M)*n, desc="Computing")
    invalids = {m: np.empty(n, dtype=int) for m in M}
    for m in M:
        for i in range(n):
            with open(f"tmp/plot_canon_perc/{paths[m]}_32_0_01_samples.pkl", "rb") as f: S = pickle.load(f)
            T = [x[:i+1] for x in S[-1] if len(x) > i]
            texts = TOKS[m].batch_decode(list(map(lambda x: x[1:], T)))
            K = TOKS[m](texts, add_special_tokens=True)["input_ids"]
            invalids[m][i] = np.sum(np.array([x != y for x, y in zip(TOKS[m].batch_decode(K if m == "Mamba" else [x[1:] for x in K]), texts)]))
            W = [x == y.tolist() for x, y in zip(K, ([u[1:] for u in T] if m == "Mamba" else T))]
            P[m][i] = sum(W)/len(W)
            bar.update()
    bar.close()
    np.savetxt("tmp/plot_canon_perc/canon_perc_len_data.txt",
               np.stack((list(range(1, len(P["Llama2"])+1)), *list(100*x for x in P.values())), axis=1),
               fmt=("%d", *["%.10f" for _ in P]), header="i " + ' '.join(list(P.keys())), comments='')
    return P, invalids

def by_top_p():
    M = ["Llama2", "Gemma2B", "Mamba"]
    paths = {"Llama2": "Llama2", "Gemma2B": "Gemma_2b", "Mamba": "Mamba"}
    TOKS = {m: transformers.AutoTokenizer.from_pretrained(k) for m, k in zip(["Llama2", "Gemma2B", "Mamba"],
                                                                             ["/space/renatolg/llama-2-7b-hf", "google/gemma-2b", "state-spaces/mamba-130m-hf"])}
    n = len(np.arange(0, 1.01, 0.01))
    P = {m: np.empty(n, dtype=np.float32) for m in M}
    bar = tqdm.tqdm(total=len(M)*n, desc="Computing")
    invalids = {m: np.empty(n, dtype=int) for m in M}
    for m in M:
        for i in range(n):
            with open(f"tmp/plot_canon_perc/{paths[m]}_32_0_01_samples.pkl", "rb") as f: S = pickle.load(f)
            T = S[i]
            texts = TOKS[m].batch_decode(list(map(lambda x: x[1:], T)))
            K = TOKS[m](texts, add_special_tokens=True)["input_ids"]
            invalids[m][i] = np.sum(np.array([x != y for x, y in zip(TOKS[m].batch_decode(K if m == "Mamba" else [x[1:] for x in K]), texts)]))
            W = [x == y.tolist() for x, y in zip(K, ([u[1:] for u in T] if m == "Mamba" else T))]
            P[m][i] = sum(W)/len(W)
            bar.update()
    bar.close()
    np.savetxt("tmp/plot_canon_perc/canon_perc_data.txt",
               np.stack((np.arange(0, 1.01, 0.01), *list(100*x for x in P.values())), axis=1),
               fmt=("%.10f", *["%.10f" for _ in P]), header="i " + ' '.join(list(P.keys())), comments='')
    return P, invalids

def plot_interactive():
    global APPEND
    APPEND="_0_01"
    # Load statistics.
    S = {}
    for m in tqdm.tqdm(MODELS.keys(), desc="Loading statistics"):
        path = f"tmp/plot_canon_perc/{m}{APPEND}_statistics.pkl"
        if os.path.isfile(path):
            with open(path, "rb") as f: S[m] = pickle.load(f)

    # Gather statistics.
    mu, sigma = {}, {}
    for m in tqdm.tqdm(S, desc="Computing mean and stdev of distances", total=len(S)):
        mu[m], sigma[m] = np.mean(S[m][1], axis=1), np.std(S[m][1], axis=1)

    # Plot.
    fig = plotly.subplots.make_subplots(
        rows=3, cols=1,
        column_widths=[1.0],
        row_heights=[1.0, 1.0, 1.0],
    )

    rng = np.linspace(0., 1., len(S[m][0]))
    dtick = 10*(rng[1]-rng[0])
    #which = ["Llama2Chat_32", "Gemma_2b_32", "GPT2_small", "GPT2_medium", "GPT2_large", "GPT2_xl"]
    f = plotly.express.line({k: S[k][0]/100 for k in S}, title="Percentage of canonical generations")
    f.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(len(rng))),
                               ticktext=[f"{x:.2f}" for x in rng]), yaxis_tickformat='.0%',
                    xaxis_title="p in top-p", yaxis_title="Percentage of canonical generations",
                    legend_title="Models")
    # f.write_image("tmp/a.pdf")
    f.write_html("/home/renatolg/www/tokens/canons.html")

    rng_d = np.concatenate((rng, rng[::-1]))
    dist_plot = plotly.graph_objects.Figure()
    for m in S:
        dist_plot.add_trace(plotly.graph_objects.Scatter(
            x=rng_d,
            y=np.concatenate((mu[m]+sigma[m], (mu[m]-sigma[m])[::-1])),
            fill="toself",
            name=m,
            showlegend=False))
    for m in S:
        dist_plot.add_trace(plotly.graph_objects.Scatter(
            x=rng, y=mu[m], name=m
        ))
    dist_plot.write_html("/home/renatolg/www/tokens/distances.html")

    f = plotly.express.line({k: S[k][2] for k in S}, title="Percentage of invalids")
    f.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(len(rng))), ticktext=[str(x) for x in rng]))
    f.write_html("/home/renatolg/www/tokens/invalids.html")

APPEND = ''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-M", "--models", nargs='+', choices=list(MODELS.keys()) + ["all"], required=True)
    parser.add_argument("-l", "--length", type=int, default=100)
    parser.add_argument("-p", "--p-step", type=float, default=0.05)
    parser.add_argument("-a", "--append", type=str, default='')
    args = vars(parser.parse_args())

    APPEND = ('_' + args["append"]) if len(args["append"]) != 0 else ''
    if "all" in args["models"]: mdls, names = list(MODELS.values()), list(MODELS.keys())
    else: mdls, names = [MODELS[k] for k in args["models"] if k != "all"], [k for k in args["models"] if k != "all"]

    run(mdls, names, args["num_samples"], args["batch_size"], args["length"], args["p_step"])
