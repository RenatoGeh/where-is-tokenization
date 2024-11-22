import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.shuffle(seed=101)
