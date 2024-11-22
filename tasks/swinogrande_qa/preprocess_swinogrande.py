import datasets

def doc_to_text(doc):
    idx = doc["sentence"].index("_")
    return doc["sentence"][:idx-1]

def doc_to_target(doc): return int(doc["answer"])-1

def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    return [doc["sentence"][idx:].replace('_', x) for x in [doc["option1"], doc["option2"]]]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset: return dataset.shuffle(seed=101)
