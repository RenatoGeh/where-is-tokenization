import datasets

def doc_to_choice(doc: dict) -> list:
  return [doc["answerA"], doc["answerB"], doc["answerC"]]

def preprocess_docs(dataset: datasets.Dataset) -> datasets.Dataset:
  return dataset.shuffle(seed=101)
