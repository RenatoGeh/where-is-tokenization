import mdd, transformers, models, argparse, sys, pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="most_likely",
                                     description="Attempts at finding the most likely tokenization." \
"""
String given by --input is the string to be tokenized. The branch-and-bound algorithm will attempt
to find the top-k best candidates (according to likelihood) that are at least as good as the lower
bound (i.e. the canonical tokenization here) within the time budget (in seconds). You may
optionally give a device (e.g. cuda:0 for the first GPU). The output of the branch-and-bound
algorithm is a pickle file containing a tuple (L, r), where L is a list of pairs (log-likelihood,
tokenization) for each candidate and r is the remaining time left from the time budget.
""")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input string to be tokenized")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Top-k candidates")
    parser.add_argument("-b", "--budget", type=int, default=60*60*60, help="Time budget in seconds")
    parser.add_argument("-m", "--model", type=str, choices=["Llama2", "Gemma", "Mamba"], help="LLM to evaluate on")
    parser.add_argument("-d", "--device", type=str, required=True, help="Device to run on (e.g. cuda:0)")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="Output file path")
    args = vars(parser.parse_args())

    print("Loading model...")
    M = getattr(sys.modules["models"], args["model"])(device_map=args["device"],
                                                      attn_implementation="eager")
    S = args["input"]
    print("Compiling MDD...")
    dd = mdd.build_mdd(M.tokenizer, S)
    print("Running branch-and-bound...")
    t = dd.bb(M.tokenizer, M.model, bound_tok=M.tokenizer.encode(S, add_special_tokens=False),
              time_budget=args["budget"], k=args["top_k"])
    print(f"Writing to {args['output_file']}...")
    with open(args["output_file"], "wb") as f: pickle.dump(t, f)
