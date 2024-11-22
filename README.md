# Experiments and code for the paper "Where is the signal in tokenization space?"

See paper for more information: https://aclanthology.org/2024.emnlp-main.230/

To install dependencies, run (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

To replicate the experiments for the branch-and-bound most likely tokenization, run
`branch_bound.py`.

```bash
python branch_bound.py
```

To replicate the experiments for the marginal, run `harness.py` with the desired task, model, etc.
For example:

```bash
python harness.py --device=cuda:0 --num-samples=64 --tasks=shellaswag --mode=MarLM --model=Gemma
```
