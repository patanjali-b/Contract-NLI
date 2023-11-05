import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, BertForSequenceClassification, BertTokenizer
import warnings

warnings.filterwarnings('ignore', message=".*overflowing tokens.*")

with open('test.json', 'r') as f:
    document_data = json.load(f)
            
# model = BertForSequenceClassification.from_pretrained("distilbert_model.pt")

# load the model distilbert_model.pt from my directory
# model = torch.load('distilbert_model.pt')

model = DistilBertForSequenceClassification.from_pretrained("distilbert_model")   
tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")

model_contradiction = BertForSequenceClassification.from_pretrained("fine-tuned-bert-contradiction")
model_entailment = BertForSequenceClassification.from_pretrained("fine-tuned-bert-entailment")
tokenizer_evidence = BertTokenizer.from_pretrained("bert-base-uncased")

print("Model loaded.")  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

total_correct = 0
total_predicted = 0

choices = {"NotMentioned": 0, "Entailment": 1, "Contradiction": 2}

for document in document_data["documents"]:
    correct = 0
    predicted_counter = 0
    
    for annotation_set in document["annotation_sets"]:
        annotations = annotation_set["annotations"]
        for nda, annotation in annotations.items():
            prob_entailments = []
            prob_contradictions = []
            text = document["text"]
            hypothesis = document_data["labels"][nda]["hypothesis"]

            # Tokenize input text and hypothesis
            encoding = tokenizer(text, hypothesis, return_tensors='pt', truncation=True, padding='max_length', max_length=512, return_attention_mask=True)

            # Move tensors to the appropriate device
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Perform inference with torch.no_grad()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
           # print(f"Predictions: {predictions.item()}, Labels: {annotation['choice']}")

            # Calculate correct and predicted counts
            correct += int(predictions.item() == choices[annotation['choice']])
            predicted_counter += 1

            total_correct += correct
            total_predicted += predicted_counter

            if predictions.item() == 1:
                document_spans = document['spans']
                sliced_texts = [document['text'][span[0]:span[1]] for span in document_spans]

                for idx, sliced_text in enumerate(sliced_texts):
                    encoding = tokenizer_evidence(sliced_text, hypothesis, return_tensors='pt', truncation=True, padding='max_length', max_length=512, return_attention_mask=True)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    with torch.no_grad():
                        outputs = model_entailment(input_ids, attention_mask=attention_mask)
                        prob_entailment = outputs.logits.softmax(dim=1)

                    if prob_entailment[0][0] < prob_entailment[0][1]:
                        prob_entailments.append(idx)

            elif predictions.item() == 2:
                document_spans = document['spans']
                sliced_texts = [document['text'][span[0]:span[1]] for span in document_spans]

                for idx, sliced_text in enumerate(sliced_texts):
                    encoding = tokenizer_evidence(sliced_text, hypothesis, return_tensors='pt', truncation=True, padding='max_length', max_length=512, return_attention_mask=True)
                    input_ids = encoding['input_ids']
                    attention_mask = encoding['attention_mask']

                    with torch.no_grad():
                        outputs = model_contradiction(input_ids, attention_mask=attention_mask)
                        prob_contradiction = outputs.logits.softmax(dim=1)

                    if prob_contradiction[0][0] < prob_contradiction[0][1]:
                        prob_contradictions.append(idx)

# Calculate accuracy
accuracy = total_correct / total_predicted
print(f"Accuracy: {accuracy * 100:.2f}%")
