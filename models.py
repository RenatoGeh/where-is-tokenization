from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from typing import Union
import torch
import collections
import utils

class Model:
    def __init__(self, path: Union[str, AutoModelForCausalLM],
                 device_map: Union[str, AutoTokenizer] = "auto", from_existing: bool = False,
                 dtype = torch.float16, tok_args: dict = {}, **kwargs):
        if from_existing: self.model, self.tokenizer = path, device_map
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, **tok_args)
            self.model = AutoModelForCausalLM.from_pretrained(path, device_map=device_map,
                                                              torch_dtype=dtype, **kwargs)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.name = f"{type(self).__name__}_{str(dtype)[-2:]}b"

    def loglikelihood(self, X: list, **kwargs):
        if isinstance(X[0], collections.abc.Iterable):
            return utils.loglikelihood_batch(self.model, self.tokenizer, X, **kwargs)
        return utils.loglikelihood(self.model, X)

    def perplexity(self, X: list, **kwargs):
        if isinstance(X[0], collections.abc.Iterable):
            return utils.perplexity_batch(self.model, self.tokenizer, X, **kwargs)
        return utils.perplexity(self.model, X)

    def generate(self, prompt: collections.abc.Iterable, **kwargs):
        if isinstance(prompt, torch.Tensor): prompt = torch.tensor(prompt)
        return utils.generate(self.model, self.tokenizer, prompt, **kwargs)

    def __call__(self, X, **kwargs): return self.model(X, **kwargs)

    def encode(self, X, **kwargs): return self.tokenizer.encode(X, **kwargs)
    def decode(self, X, **kwargs): return self.tokenizer.decode(X, **kwargs)
    def convert_ids_to_tokens(self, X, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(X, **kwargs)
    def convert_tokens_to_ids(self, X, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(X, **kwargs)
    def prepare_for_model(self, I, **kwargs): return self.tokenizer.prepare_for_model(I, **kwargs)

    @staticmethod
    def from_existing(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        return Model(model, tokenizer, from_existing=True)

    @classmethod
    def get_tokenizer(cls, which: str = '', **kwargs):
        tokenizer = AutoTokenizer.from_pretrained((cls.LOCAL_PATH + which) if isinstance(cls.LOCAL_PATH, str) else
                                                  cls.LOCAL_PATH[which], **kwargs)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

class Llama(Model):
    LOCAL_PATH = "/space/renatolg/llama-7b"
    def __init__(self, **kwargs): super().__init__(Llama.LOCAL_PATH, **kwargs)

class Llama2(Model):
    LOCAL_PATH = "/space/renatolg/llama-2-7b-hf"
    def __init__(self, **kwargs): super().__init__(Llama2.LOCAL_PATH, **kwargs)

class Llama2Chat(Model):
    LOCAL_PATH = "/space/renatolg/llama-2-7b-chat-hf"
    def __init__(self, **kwargs): super().__init__(Llama2Chat.LOCAL_PATH, **kwargs)

class Gemma(Model):
    LOCAL_PATH = "google/gemma-"
    def __init__(self, which: str = "2b", **kwargs): super().__init__(Gemma.LOCAL_PATH + which, **kwargs)

class GPT2(Model):
    LOCAL_PATH = {
        "small": "gpt2",
        "medium": "gpt2-medium",
        "large": "gpt2-large",
        "xl": "gpt2-xl",
    }
    def __init__(self, which: str = "small", **kwargs):
        super().__init__(GPT2LMHeadModel.from_pretrained(GPT2.LOCAL_PATH[which], **kwargs),
                         GPT2Tokenizer.from_pretrained(GPT2.LOCAL_PATH[which], padding_side="left",
                                                       pad_token="<|endoftext|>"),
                         from_existing=True)

class Mistral(Model):
    LOCAL_PATH = {
        "": "mistralai/Mistral-7B-v0.1",
        "it": "mistralai/Mistral-7B-Instruct-v0.1",
    }
    def __init__(self, which: str = '', **kwargs): super().__init__(Mistral.LOCAL_PATH[which], **kwargs)

class Mamba(Model):
    LOCAL_PATH = "state-spaces/mamba-130m-hf"
    def __init__(self, **kwargs): super().__init__(Mamba.LOCAL_PATH, **kwargs)
