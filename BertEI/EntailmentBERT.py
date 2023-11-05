#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install torch')


# In[2]:


import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
import pandas as pd
import math
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import TrainingArguments, Trainer
import warnings

warnings.filterwarnings("ignore")


# In[3]:


with open("train.json") as f:
    document_data = json.load(f)


# In[4]:


total_span_indices = 0
total_spans = 0
total_num_spans = 0
least_num_spans = math.inf
total_documents = 0

for document in document_data["documents"]:
    total_num_spans += len(document["spans"])
    least_num_spans = min(least_num_spans, len(document["spans"]))
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            choice = annotation["choice"]
            if choice == "Entailment":
                spans_indices = annotation["spans"]
                total_span_indices += len(spans_indices)
                total_spans += 1
    total_documents += 1

average_span_indices = total_span_indices / total_spans
print(f"Average number of elements in span_indices: {average_span_indices}")

average_num_spans = total_num_spans / total_documents
print(f"Average number of spans per document: {average_num_spans}")
print(f"Least number of spans across all documents: {least_num_spans}")


# In[7]:


total_sliced_texts = 0
total_random_sliced_texts = 0

for document in document_data["documents"]:
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            choice = annotation["choice"]
            if choice == "Entailment":
                spans_indices = annotation["spans"]
                document_spans = document["spans"]
                spans = [document_spans[i] for i in spans_indices]
                sliced_texts = [document["text"][start:end] for start, end in spans]

                total_sliced_texts += len(sliced_texts)

                random_indices = random.sample(range(len(document_spans)), min(len(document_spans) - len(spans_indices), 6))
                random_indices = [i for i in random_indices if i not in spans_indices]
                random_spans = [document_spans[i] for i in random_indices]
                random_sliced_texts = [document["text"][start:end] for start, end in random_spans]

                total_random_sliced_texts += len(random_sliced_texts)

                hypothesis = document_data["labels"][nda]["hypothesis"]

                # print(f"Entailment: {sliced_texts} -> {hypothesis}")
                # print(f"Not Entailed: {random_sliced_texts} -> {hypothesis}")
print(f"Number of Entailment : Non-Entalments = {total_sliced_texts} : {total_random_sliced_texts}")


# In[8]:


class TrainDataset(Dataset):
    def __init__(self, document_data, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        for document in document_data["documents"]:
            for annotation_set in document["annotation_sets"]:
                annotations = annotation_set["annotations"]
                for nda, annotation in annotations.items():
                    choice = annotation["choice"]
                    if choice == "Entailment":
                        spans_indices = annotation["spans"]
                        document_spans = document["spans"]
                        spans = [document_spans[i] for i in spans_indices]
                        sliced_texts = [document["text"][start:end].lower() for start, end in spans]

                        random_indices = random.sample(range(len(document_spans)), 2)
                        random_indices = [i for i in random_indices if i not in spans_indices]
                        random_spans = [document_spans[i] for i in random_indices]
                        random_sliced_texts = [document["text"][start:end].lower() for start, end in random_spans]

                        hypothesis = document_data["labels"][nda]["hypothesis"].lower()

                        self.data.extend([(text, hypothesis, 1) for text in sliced_texts])
                        self.data.extend([(text, hypothesis, 0) for text in random_sliced_texts])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, hypothesis, label = self.data[idx]
        encoding = self.tokenizer(text, hypothesis, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# In[9]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TrainDataset(document_data, tokenizer, 512)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {average_loss}")


# In[10]:


model.save_pretrained("fine-tuned-bert-entailment")

