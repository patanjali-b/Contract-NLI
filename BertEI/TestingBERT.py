#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install torch transformers')


# In[4]:


get_ipython().system(' pip install numpy prettytable')


# In[6]:


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
from prettytable import PrettyTable
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import TrainingArguments, Trainer
import warnings

warnings.filterwarnings("ignore")


# In[3]:


with open("ContractNLI-Dataset/test.json") as f:
    document_data = json.load(f)


# In[56]:


model = BertForSequenceClassification.from_pretrained("fine-tuned-bert-entailment")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

total_correct = 0
total_predicted = 0

for document in document_data["documents"]:
    correct = 0
    predicted = 0
    
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            choice = annotation["choice"]
            if choice == "Entailment":
                spans_indices = annotation["spans"]
                document_spans = document["spans"]
                spans = [document_spans[i] for i in spans_indices]
                sliced_texts = [document["text"][start:end] for start, end in spans]

                random_indices = random.sample(range(len(document_spans)), min(len(document_spans) - len(spans_indices), 6))
                random_indices = [i for i in random_indices if i not in spans_indices]
                random_spans = [document_spans[i] for i in random_indices]
                random_sliced_texts = [document["text"][start:end] for start, end in random_spans]

                hypothesis = document_data["labels"][nda]["hypothesis"]
                
                for text in sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] < probabilities[0][1]:
                        correct += 1
                        total_correct += 1
                    predicted += 1
                    total_predicted += 1
                    
                for text in random_sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] > probabilities[0][1]:
                        correct += 1
                        total_correct += 1
                    predicted += 1
                    total_predicted += 1

    print(f"Accuracy on Document {document['id']}: {correct / predicted}")
    
print(f"Total Accuracy: {total_correct / total_predicted}")


# In[66]:


model = BertForSequenceClassification.from_pretrained("fine-tuned-bert-contradiction")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

total_correct = 0
total_predicted = 0

for document in document_data["documents"]:
    correct = 0
    predicted = 0
    
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            choice = annotation["choice"]
            if choice == "Contradiction":
                spans_indices = annotation["spans"]
                document_spans = document["spans"]
                spans = [document_spans[i] for i in spans_indices]
                sliced_texts = [document["text"][start:end] for start, end in spans]

                random_indices = random.sample(range(len(document_spans)), min(len(document_spans) - len(spans_indices), 10))
                random_indices = [i for i in random_indices if i not in spans_indices]
                random_spans = [document_spans[i] for i in random_indices]
                random_sliced_texts = [document["text"][start:end] for start, end in random_spans]

                hypothesis = document_data["labels"][nda]["hypothesis"]
                
                for text in sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] < probabilities[0][1]:
                        correct += 1
                        total_correct += 1
                    predicted += 1
                    total_predicted += 1
                    
                for text in random_sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] > probabilities[0][1]:
                        correct += 1
                        total_correct += 1
                    predicted += 1
                    total_predicted += 1
    if predicted == 0:
        continue
    print(f"Accuracy on Document {document['id']}: {correct / predicted}")
    
print(f"Total Accuracy: {total_correct / total_predicted}")    


# In[63]:


import re
import matplotlib.pyplot as plt

file_path = "entailment.txt"
with open(file_path, "r") as file:
    text = file.read()

accuracy_values = re.findall(r"Accuracy on Document \d+: (\d+\.\d+)", text)
accuracy_values = [float(accuracy) * 100 for accuracy in accuracy_values]

plt.figure(figsize=(10, 6))
plt.plot(accuracy_values, marker='.')
plt.title("Entailment Accuracy on Documents")
plt.xlabel("Document ID")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# In[67]:


import re
import matplotlib.pyplot as plt

file_path = "contradiction.txt"
with open(file_path, "r") as file:
    text = file.read()

accuracy_values = re.findall(r"Accuracy on Document \d+: (\d+\.\d+)", text)
accuracy_values = [float(accuracy) * 100 for accuracy in accuracy_values]

plt.figure(figsize=(10, 6))
plt.plot(accuracy_values, marker='.')
plt.title("Contradiction Accuracy on Documents")
plt.xlabel("Document ID")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


# In[7]:


model = BertForSequenceClassification.from_pretrained("fine-tuned-bert-entailment")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

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

                random_indices = random.sample(range(len(document_spans)), min(len(document_spans) - len(spans_indices), 6))
                random_indices = [i for i in random_indices if i not in spans_indices]
                random_spans = [document_spans[i] for i in random_indices]
                random_sliced_texts = [document["text"][start:end] for start, end in random_spans]

                hypothesis = document_data["labels"][nda]["hypothesis"]
                
                for text in sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] < probabilities[0][1]:
                        true_positive += 1
                    else:
                        false_negative += 1
                    
                for text in random_sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] > probabilities[0][1]:
                        true_negative += 1
                    else:
                        false_positive += 1
                        
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * precision * recall / (precision + recall)
confusion_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("Confusion Matrix:")
table = PrettyTable()
table.field_names = ["", "Actual Positive", "Actual Negative"]
table.add_row(["Predicted Positive", true_positive, false_positive])
table.add_row(["Predicted Negative", false_negative, true_negative])
print(table)


# In[8]:


model = BertForSequenceClassification.from_pretrained("fine-tuned-bert-contradiction")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for document in document_data["documents"]:
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            choice = annotation["choice"]
            if choice == "Contradiction":
                spans_indices = annotation["spans"]
                document_spans = document["spans"]
                spans = [document_spans[i] for i in spans_indices]
                sliced_texts = [document["text"][start:end] for start, end in spans]

                random_indices = random.sample(range(len(document_spans)), min(len(document_spans) - len(spans_indices), 10))
                random_indices = [i for i in random_indices if i not in spans_indices]
                random_spans = [document_spans[i] for i in random_indices]
                random_sliced_texts = [document["text"][start:end] for start, end in random_spans]

                hypothesis = document_data["labels"][nda]["hypothesis"]
                
                for text in sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] < probabilities[0][1]:
                        true_positive += 1
                    else:
                        false_negative += 1
                    
                for text in random_sliced_texts:
                    encoding = tokenizer(text, hypothesis, max_length=512, padding="max_length", truncation=True, return_tensors="pt", return_attention_mask=True)
                    input_ids = encoding["input_ids"]
                    attention_mask = encoding["attention_mask"]

                    with torch.no_grad():
                        logits = model(input_ids, attention_mask=attention_mask)
                        probabilities = logits.logits.softmax(dim=1)

                    if probabilities[0][0] > probabilities[0][1]:
                        true_negative += 1
                    else:
                        false_positive += 1
    if true_positive + true_negative + false_positive + false_negative == 0:
        continue
    
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * precision * recall / (precision + recall)
confusion_matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print("Confusion Matrix:")
table = PrettyTable()
table.field_names = ["", "Actual Positive", "Actual Negative"]
table.add_row(["Predicted Positive", true_positive, false_positive])
table.add_row(["Predicted Negative", false_negative, true_negative])
print(table)


# In[10]:


import matplotlib.pyplot as plt
import numpy as np

TP = 1761
TN = 5406
FP = 252
FN = 219

confusion_matrix = np.array([[TP, FN], [FP, TN]])

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Entailment BERT')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Positive", "Negative"])
plt.yticks(tick_marks, ["Positive", "Negative"])

plt.xlabel('Actual')
plt.ylabel('Predicted')

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

plt.show()


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

TP = 365
TN = 1996
FP = 136
FN = 59

confusion_matrix = np.array([[TP, FN], [FP, TN]])

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Contradiction BERT')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Positive", "Negative"])
plt.yticks(tick_marks, ["Positive", "Negative"])

plt.xlabel('Actual')
plt.ylabel('Predicted')

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

plt.show()

