import tokens, models, mdd, config, utils, pmdd
import lm_eval, torch, tqdm, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse, pickle, math, multiprocessing, sys, os, random

class MarLM(lm_eval.models.huggingface.HFLM):
    def __init__(self, model: str, nsamples: int, use_cache: bool, device: str, batch_size: int,
                 device_map_option: int, proposal: str, prepend_path: str, bigram_path: list,
                 include_canonical: bool, **kwargs):
        if isinstance(model, str):
            super().__init__(pretrained=model, max_length=None, device=device, batch_size=batch_size,
                             parallelize=False, device_map_option=device_map_option, **kwargs)
        else:
            super().__init__(model, max_length=None, device=device, batch_size=batch_size,
                             parallelize=False, device_map_option=device_map_option, **kwargs)
        self.nsamples = nsamples
        self.use_cache = use_cache
        self.prepend_path = prepend_path
        self.proposal = proposal
        self.bigram_path = bigram_path if isinstance(bigram_path, list) else [bigram_path]
        self.include_canonical = include_canonical

    def loglikelihood(self, requests: list) -> list:
        # random.seed(14141414)
        # np.random.seed(1414141414)
        # torch.manual_seed(1414141414)
        ret = [None for _ in range(len(requests))]
        samples = []
        append = "_1" if APPEND_LAST else ''
        if len(EVAL_APPEND) == 0:
            if os.path.isfile(f"tmp/{self.prepend_path}_batch_checkpoint{append}.pkl"):
                with open(f"tmp/{self.prepend_path}_batch_checkpoint{append}.pkl", "rb") as f:
                    mdd.MAR_BATCH_LIST = pickle.load(f)
            print(mdd.MAR_BATCH_LIST)
            print("Which", append)
        _rng = list(enumerate([req.args for req in requests]))
        rng = list(reversed(_rng)) if APPEND_LAST or REVERSE else _rng
        prefixes = [[self.eot_token_id] if r.args[0] == "" else self.tok_encode(r.args[0], add_special_tokens=True) for r in requests]
        if os.path.isfile(f"tmp/{self.prepend_path}_all_mdds.pkl"):
            with open(f"tmp/{self.prepend_path}_all_mdds.pkl", "rb") as f: all_mdds = pickle.load(f)
        else:
            f_build = pmdd.build_pmdd_parallel_pargs if "bigram" in self.proposal else mdd.build_mdd_parallel_pargs
            all_mdds = f_build(self.tokenizer, [r.args[1] for r in requests],
                               prefix_space=[r.args[1][0] != ' ' for r in requests])
            if "bigram" in self.proposal:
                B = pmdd.bigram_from_files(self.bigram_path, 10, tokenizer=self.tokenizer)
                pmdd.learn_weights_pargs(B, all_mdds, start_token=[p[-1] for p in prefixes],
                                         parallel=False, uncond=True)
                breakpoint()
                if "reweight" in self.proposal:
                    T_can = [self.tok_encode(r.args[1], add_special_tokens=False) for r in requests]
                    pmdd.reweight_paths(B, T_can, self.model, self.batch_size, prefix=prefixes,
                                        tokenizer=self.tokenizer)
            with open(f"tmp/{self.prepend_path}_all_mdds.pkl", "wb") as f: pickle.dump(all_mdds, f)

        for i, (context, continuation) in tqdm.tqdm(rng, position=0, desc="Requests"):
            if len(EVAL_APPEND) > 0:
                if os.path.isfile(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}.pkl"):
                    with open(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}.pkl", "rb") as f:
                        ll, S, LL_P, LL_Q, n_LL = pickle.load(f)
                elif os.path.isfile(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}_1.pkl"):
                    with open(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}_1.pkl", "rb") as f:
                        ll, S, LL_P, LL_Q, n_LL = pickle.load(f)
            elif os.path.isfile(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}{append}.pkl"):
                with open(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}{append}.pkl", "rb") as f:
                    ll, S, LL_P, LL_Q, n_LL = pickle.load(f)
            else:
                prefix = prefixes[i]
                k = len(prefix)+len(continuation)
                M = all_mdds[i]
                try:
                    can = self.tok_encode(continuation[1:] if continuation[0] == ' ' and not utils.is_mamba_tokenizer(self.tokenizer) else continuation,
                                          add_special_tokens=False) if self.include_canonical else None
                    ll, S, LL_P, LL_Q, n_LL = M.mar_by_sample(self.model, self.tokenizer, self.nsamples,
                                            prefix=prefix, batch_size=self.batch_size if
                                            self.batch_size != "auto" else self.max_length,
                                            size_estimate=k, canonical=can, return_extra=True)
                except Exception as exc:
                    with open(f"tmp/{self.prepend_path}_batch_checkpoint{append}.pkl", "wb") as f:
                        for j in range(len(mdd.MAR_BATCH_LIST)):
                            b, l = mdd.MAR_BATCH_LIST[j]
                            if k < l:
                                mdd.MAR_BATCH_LIST[j] = (b, k)
                                break
                        print("Saving", mdd.MAR_BATCH_LIST)
                        pickle.dump(mdd.MAR_BATCH_LIST, f)
                    raise exc
                with open(f"out/mar_samples/{self.prepend_path}_{self.nsamples}_{i}{append}.pkl", "wb") as f: pickle.dump((ll, S, LL_P, LL_Q, n_LL), f)
            ret[i] = (ll, LL_P, LL_Q, n_LL)
            samples.append(S)
        with open(f"out/rets/{self.prepend_path}_samples.pkl", "wb") as f: pickle.dump(samples, f)
        with open(f"out/rets/{self.prepend_path}_rets.pkl", "wb") as f: pickle.dump(ret, f)
        return [(ll, False) for ll, _, _, _ in ret]

APPEND_LAST = False
REVERSE = False

class OneSampleLM(lm_eval.models.huggingface.HFLM):
    def __init__(self, model: str, use_cache: bool, device: str, batch_size: int,
                 device_map_option: int, proposal: str, **kwargs):
        if model == "Gemma7B": path = "google/gemma-7b"
        else:
            path = getattr(models, model).LOCAL_PATH
            if model == "Gemma": path += "2b"
        super().__init__(pretrained=path, dtype=torch.float32, max_length=None, device=device,
                         batch_size=batch_size, parallelize=False,
                         device_map_option=device_map_option, **kwargs)
        self.use_cache = use_cache
        self.prepend_path = prepend_path

        def llm_proposal(R: list, T: list, prefix: list) -> torch.Tensor:
            M = mdd.build_mdd_parallel(self.tokenizer, [c for _, c in R])
            LL_Q = torch.tensor([M[i].pr_from_lm(self.model, T[i], prefix=prefix[i], use_cache=self.use_cache)
                                 for i in tqdm.tqdm(range(len(T)), desc="Computing Q(T*|S)")])
            return LL_Q
        def bigram_proposal(R: list, T: list, prefix: list) -> torch.Tensor:
            # if os.path.isfile(f"tmp/{self.prepend_path}_qr.pkl"):
                # with open(f"tmp/{self.prepend_path}_qr.pkl", "rb") as f: LL_Q = pickle.load(f)
            # else:
            LL_Q = torch.tensor(utils.parallelize(utils.partial(self._bigram_f, self.tokenizer, self.B, R, prefix, T),
                                                  list(range(len(T))), desc="Computing Q(T*|S)", position=1))
                # with open(f"tmp/{self.prepend_path}_qr.pkl", "wb") as f: pickle.dump(LL_Q, f)
            return LL_Q
        def partition_proposal(R: list, T: list, prefix: list) -> torch.Tensor:
            return torch.tensor(utils.parallelize(utils.partial(self._bigram_p, self.tokenizer,
                                                                self.B, R, prefix, T),
                                                  list(range(len(T))), desc="Computing Q(E)", position=1))
        def bigram_reweight_proposal(R: list, T: list, prefix: list) -> torch.Tensor:
            P = pmdd.build_pmdd_parallel(self.tokenizer, [r[1] for r in R])
            pmdd.learn_weights_pargs(self.B, P, start_token=[p[-1] for p in prefix], uncond=True)
            pmdd.reweight_paths(P, T, self.model, self.batch_size, prefix, tokenizer=self.tokenizer)
            return torch.tensor([P[i].logpr(T[i])-P[i].logpr() for i in tqdm.tqdm(range(len(T)), desc="Computing probs")])

        def partition_reweight_proposal(R: list, T: list, prefix: list) -> torch.Tensor:
            P = pmdd.build_pmdd_parallel(self.tokenizer, [r[1] for r in R])
            pmdd.learn_weights_pargs(self.B, P, start_token=[p[-1] for p in prefix], uncond=True)
            pmdd.reweight_paths(P, T, self.model, self.batch_size, prefix, tokenizer=self.tokenizer)
            return torch.tensor([P[i].logpr() for i in tqdm.tqdm(range(len(T)), desc="Computing probs")])

        PROPOSAL_MAP = {"bigram": bigram_proposal, "partition": partition_proposal,
                        "bigram-reweight": bigram_reweight_proposal,
                        "partition-reweight": partition_reweight_proposal}

        if proposal != "llm":
            p = f"compiled/{self.prepend_path}_bigram.pkl"
            if os.path.isfile(p):
                with open(p, "rb") as f: self.B = pickle.load(f)
            else:
                self.B = pmdd.bigram_from_files([f"out/samples/sample_orval{u}_0.pkl" for u in ["", "-1", "-2"]],
                                                 1.0, parallel=True)
                with open(p, "wb") as f: pickle.dump(self.B, f)
            self.f_prop= PROPOSAL_MAP[proposal]
        else: self.f_prop = llm_proposal
        self.proposal = proposal

    @staticmethod
    def _bigram_f(tokenizer: AutoTokenizer, B: pmdd.Bigram, R: list, prefix: list, T: list, i: int):
        P = pmdd.build_pmdd(tokenizer, R[i][1])
        pmdd.learn_weights(B, P, start_token=prefix[i][-1], normalize=True)
        return P.logpr(T[i])-P.logpr()

    @staticmethod
    def _bigram_p(tokenizer: AutoTokenizer, B: pmdd.Bigram, R: list, prefix: list, T: list, i: int):
        P = pmdd.build_pmdd(tokenizer, R[i][1])
        pmdd.learn_weights(B, P, start_token=prefix[i][-1], normalize=True)
        return P.logpr()

    def loglikelihood(self, requests: list) -> list:
        prefix, T = [None for _ in range(len(requests))], [None for _ in range(len(requests))]
        R = [req.args for req in requests]
        for i, (context, continuation) in enumerate(R):
            prefix[i] = self.tok_encode(context, add_special_tokens=True)
            T[i] = self.tok_encode(continuation[1:] if continuation[0] == ' ' and not utils.is_mamba_tokenizer(self.tokenizer)
                                   else continuation, add_special_tokens=False)
        b = self.batch_size if self.batch_size != "auto" else self.max_length
        if self.proposal == "partition":
            LL_t = self.f_prop(R, T, prefix)
        else:
            LL_P = utils.rhloglikelihood(self.model, self.tokenizer, T, batch_size=b, prefix=prefix,
                                       prefix_cutoff=True, online_memory=True, use_tqdm=True,
                                       desc="Computing P(T*, S)")
            LL_Q = self.f_prop(R, T, prefix)
            LL_t = LL_P-LL_Q
            with open(f"tmp/{self.prepend_path}_can_pr_qr_ll_checkpoint.pkl", "wb") as f: pickle.dump((LL_P, LL_Q.to("cpu")), f)
        return [(ll.item(), False) for ll in LL_t]

class CanonicalLM(lm_eval.models.huggingface.HFLM):
    def __init__(self, model: str, use_cache: bool, device: str, batch_size: int,
                 device_map_option: int, **kwargs):
        super().__init__(pretrained=model, max_length=None, device=device,
                         batch_size=batch_size, parallelize=False,
                         device_map_option=device_map_option, **kwargs)
        self.use_cache = use_cache

    def loglikelihood(self, requests: list) -> list:
        prefix, T = [None for _ in range(len(requests))], [None for _ in range(len(requests))]
        for i, (context, continuation) in enumerate([req.args for req in requests]):
            prefix[i] = self.tok_encode(context, add_special_tokens=True)
            T[i] = self.tok_encode(' ' + continuation if continuation[0] != ' ' else continuation, add_special_tokens=False)
        b = self.batch_size if self.batch_size != "auto" else self.max_length
        LL = utils.rhloglikelihood(self.model, self.tokenizer, T, batch_size=b, prefix=prefix,
                                  prefix_cutoff=True, online_memory=True, use_tqdm=True)
        return [(ll.item(), False) for ll in LL]

class TokenLM(lm_eval.models.huggingface.HFLM):
    def __init__(self, model: str, which: str, device: str, batch_size: int, tok_batch_size: int,
                 parallelize: bool, device_map_option: str, use_cache: bool, path_prepend: str,
                 **kwargs):
        path = getattr(models, model).LOCAL_PATH
        super().__init__(pretrained=path, max_length=None, device=device,
                         batch_size=batch_size, parallelize=parallelize,
                         device_map_option=device_map_option)
        print("TokenLM")
        self.which = which
        self._mdl = models.Model.from_existing(self.model, self.tokenizer)
        self.use_cache = use_cache
        self.tok_batch_size = tok_batch_size
        self.status = tqdm.tqdm(desc="Tokenizing", colour="green", position=1)
        self.path_prepend = path_prepend

    MDD_MAPPING = {"Greedy": "greedy", "Wh-Sh": "shortest"}
    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=False, **kwargs) -> list:
        if len(string) == 0: return [self.eot_token_id]
        if self.which in TokenLM.MDD_MAPPING:
            if len(string) > 5000:
                M = mdd.build_blob(self._mdl.tokenizer, string, **kwargs)
                t = mdd.blob_f(M, TokenLM.MDD_MAPPING[self.which], add_eos=False)
                if left_truncate_len: t = t[-left_truncate_len:]
                return t
            else: m = mdd.build_mdd(self.tokenizer, string)
        else: m = None
        t = tokens.TOKENIZATION_MAPPING[self.which](string, m, self._mdl, self.device,
                                                    self._detect_batch_size(), self.use_cache,
                                                    **kwargs)
        if left_truncate_len: t = t[-left_truncate_len:]
        self.status.update()
        return t

    def tok_batch_encode(self, strings: list, padding_side: str = "left",
                         left_truncate_len: int = None, truncation: bool = False) -> tuple:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        I = [self.tok_encode(s) for s in strings]
        X = self.tokenizer.prepare_for_model(I, add_special_tokens=False, padding=True,
                                             truncation=truncation, return_tensors="pt")
        if left_truncate_len:
            X["input_ids"] = X["input_ids"][:,-left_truncate_len:]
            X["attention_mask"] = X["attention_mask"][:,-left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return X["input_ids"], X["attention_mask"]

    @staticmethod
    def _decode_parallel(tokenizer, X: list) -> tuple:
        return tokenizer.decode(X[0], skip_special_tokens=True), tokenizer.decode(X[1], skip_special_tokens=True), \
               tokens.has_pre_impl_ws(X[0][0], tokenizer), tokens.has_pre_impl_ws(X[1][0], tokenizer), X[0]
    @staticmethod
    def _encode_mdd_f(tokenizer: AutoTokenizer, f_str: str, eos_token_id: int, X: list) -> list:
        a = [eos_token_id] if len(X[0]) == 0 else \
            getattr(mdd.build_mdd(tokenizer, X[0], prefix_space=X[2]),
                    TokenLM.MDD_MAPPING[f_str])(add_eos=False)
        b = [eos_token_id] if len(X[1]) == 0 else \
            getattr(mdd.build_mdd(tokenizer, X[1], prefix_space=X[3]),
                    TokenLM.MDD_MAPPING[f_str])(add_eos=False)
        return a, b
    def _encode_parallel(self, W: list) -> list:
        if self.which in TokenLM.MDD_MAPPING:
            return utils.parallelize(utils.partial(TokenLM._encode_mdd_f, self._mdl.tokenizer,
                                                   self.which, self.eot_token_id), W,
                                     desc="Re-encoding", position=4)
        return [(self.tok_encode(a, prefix_space=ws_a), self.tok_encode(b, prefix_space=ws_b, prefix=last_token))
                for a, b, ws_a, ws_b, last_token in tqdm.tqdm(W, desc="Re-encoding", position=4)]

    def loglikelihood_rolling(self, requests: list) -> list:
        loglikelihoods = []
        tokenizations = []
        canonical = []

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        i = 0
        for (string,) in tqdm.tqdm([req.args for req in requests], disable=(self.rank != 0),
                                   position=2):
            rolling_token_windows = list(
                map(
                    lm_eval.utils.make_disjoint_window,
                    lm_eval.utils.get_rolling_token_windows(
                        token_list=self._mdl.tokenizer.encode(string, add_special_tokens=False),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            canonical.append(rolling_token_windows)
            if self.which != "0-Ca-LL":
                rolling_str_windows = utils.parallelize(utils.partial(TokenLM._decode_parallel,
                                                                      self._mdl.tokenizer),
                                                        rolling_token_windows, desc="Decoding",
                                                        position=3)
                rolling_token_windows = self._encode_parallel(rolling_str_windows)
            tokenizations.append(rolling_token_windows)
            # with open(f"{self.path_prepend}_tokens_part_{i}.pkl", "wb") as f: pickle.dump(tokenizations, f)
            # with open(f"{self.path_prepend}_tokens_canon_part_{i}.pkl", "wb") as f: pickle.dump(canonical, f)
            i += 1
            m = -1
            for a, b in rolling_token_windows:
                if len(a) > m: m = len(a)
                if len(b) > m: m = len(b)
            old_max_length = self.max_length
            self._max_length = m

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=False,
                override_bs=adaptive_batch_size,
            )
            self._max_length = old_max_length

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        with open(f"{self.path_prepend}_tokens.pkl", "wb") as f: pickle.dump(tokenizations, f)
        with open(f"{self.path_prepend}_ll.pkl", "wb") as f: pickle.dump(loglikelihoods, f)
        with open(f"{self.path_prepend}_canon_tokens.pkl", "wb") as f: pickle.dump(canonical, f)

        return loglikelihoods

GEN_KWARGS = "do_sample=True,top_k=0,top_p=1,temperature=1.0"

EVAL_APPEND = ''

if __name__ == "__main__":
    random.seed(101); np.random.seed(101)
    old_lm_eval = hasattr(lm_eval.tasks, "include_path")
    if old_lm_eval:
        lm_eval.tasks.include_path(os.path.dirname(os.path.abspath(__file__)) + '/')
        lm_eval.tasks.initialize_tasks()
    else: mgr = lm_eval.tasks.TaskManager(include_path=os.path.dirname(os.path.abspath(__file__)) + '/')
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", action="store", required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("-D", "--device-map", type=str, default="0")
    parser.add_argument("-c", "--use-cache", type=bool, default=True)
    parser.add_argument("-l", "--limit", type=int, default=500)
    parser.add_argument("-n", "--num-samples", type=int, default=1000)
    parser.add_argument("-s", "--num-fewshot", type=int, default=None)
    parser.add_argument("-t", "--tasks", nargs='+', choices=lm_eval.tasks.ALL_TASKS if old_lm_eval else mgr.all_tasks, required=True)
    parser.add_argument("-q", "--quantization", type=int, choices=[4, 8, 16, 32], default=32)
    parser.add_argument("-T", "--tokenization", choices=list(tokens.TOKENIZATION_MAPPING.keys()))
    parser.add_argument("-B", "--tok-batch-size", default=64)
    parser.add_argument("-m", "--mode", choices=["CanonicalLM", "MarLM", "TokenLM", "OneSampleLM"], default="MarLM")
    parser.add_argument("-M", "--model", default="Llama2")
    parser.add_argument("-p", "--proposal", choices=["llm", "bigram", "bigram-reweight", "partition",
                                                     "partition-reweight"], default="llm")
    parser.add_argument("-a", "--append", type=str, default='')
    parser.add_argument("-r", "--reverse", action=argparse.BooleanOptionalAction)
    parser.add_argument("-e", "--evaluate-as", type=str, default='')
    parser.add_argument("-mp", "--model-path", type=str, default='')
    parser.add_argument("-bp", "--bigram-path", nargs='+', default=None)
    parser.add_argument("-ic", "--include-canonical", action=argparse.BooleanOptionalAction)
    args = vars(parser.parse_args())
    print(args)
    mode = getattr(sys.modules[__name__], args["mode"])
    EVAL_APPEND = args["evaluate_as"]
    dtype = torch.float32 if args["quantization"] == 32 else torch.float16
    mdl_path = args["model_path"] if len(args["model_path"]) > 0 else args["model"]
    if len(args["append"]) > 0: APPEND_LAST = True
    if args["reverse"]: REVERSE = True
    prepend_path = f"{'-'.join(args['tasks'])}_{args['model']}_{args['limit']}_{args['tokenization'] if args['mode'] == 'TokenLM' else args['num_samples']}_{args['mode']}" + \
        (f"_{args['proposal']}" if args["mode"] in ["OneSampleLM", "MarLM"] else "") + args["append"] + EVAL_APPEND
    print("Path", prepend_path)
    tk_args = [mdl_path, args["tokenization"], args["device"], int(args["batch_size"]) if
               args["batch_size"].isdecimal() else args["batch_size"], args["tok_batch_size"],
               False, args["device_map"], args["use_cache"], prepend_path]
    mar_args = [mdl_path, args["num_samples"], args["use_cache"], args["device"],
                int(args["batch_size"]) if args["batch_size"].isdecimal() else args["batch_size"],
                args["device_map"], args["proposal"], prepend_path, args["bigram_path"],
                args["include_canonical"]]
    can_args = [mdl_path, args["use_cache"], args["device"],
                int(args["batch_size"]) if args["batch_size"].isdecimal() else args["batch_size"],
                args["device_map"]]
    os_args = [mdl_path, args["use_cache"], args["device"],
               int(args["batch_size"]) if args["batch_size"].isdecimal() else args["batch_size"],
               args["device_map"], args["proposal"]]
    model_args = {MarLM: mar_args, TokenLM: tk_args, CanonicalLM: can_args, OneSampleLM: os_args}
    eval_args = {"model": mode(*model_args[mode], dtype=dtype),
                 "tasks": args["tasks"], "limit": args["limit"],
                 "num_fewshot": args["num_fewshot"], "batch_size": args["batch_size"],
                 "device": args["device"], "gen_kwargs": GEN_KWARGS}
    if not old_lm_eval: eval_args["task_manager"] = mgr
    res = lm_eval.simple_evaluate(**eval_args)
    with open(f"out/{prepend_path}.pkl", "wb") as f: pickle.dump(res, f)
