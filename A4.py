from simpletransformers.classification import ClassificationModel
import pandas as pd

train = pd.read_csv("data/Assignment 2_IntrotoNLP2022_data/olid-train.csv")
test = pd.read_csv("data/Assignment 2_IntrotoNLP2022_data/olid-test.csv")
model_args = {
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
}

model = ClassificationModel("bert", "bert-base-cased", args=model_args, use_cuda=False)

n_tokens = 0
n_tokens_split = 0
for index, row in train.iterrows():
    tokens = model.tokenizer.tokenize(row['text'])
    n_tokens = len(tokens) + n_tokens
    for token in tokens:
        if "##" in token:
            n_tokens_split = n_tokens_split + 1

print(n_tokens) #478955
print(n_tokens_split) #91024

word = ''
lenght = 0
for vocab in model.tokenizer.vocab:
    if len(vocab) > lenght:
        lenght = len(vocab)
        word = vocab
print(lenght) #18
print(word) #Telecommunications