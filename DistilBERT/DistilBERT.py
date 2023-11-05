import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import wandb 
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def json_to_csv(json_file, csv_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    documents = data['documents']  # Get the 'documents' array
    labels = data['labels']        # Get the 'labels' object

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['document', 'hypothesis', 'label', 'spans'])   # Write the header row

        # Iterate over each document
        for document in documents:
            doc_id = document['id']                                         # Get the document ID
            spans = document['spans']                                       # Get the 'spans' array
            annotation_set = document['annotation_sets'][0]['annotations']  # Get the 'annotations' object

            # Iterate over each annotation in the annotation set
            for annotation_id, annotation in annotation_set.items():
                label = annotation['choice']                                # Get the 'choice' label
                hypothesis = labels[annotation_id]['hypothesis']            # Get the hypothesis from 'labels' using the annotation ID
                span_indices = annotation['spans']                          # Get the span indices

                span_text = []
                for index in span_indices:
                    if isinstance(index, int):
                        span_text.append(spans[index])
                    else:
                        span_text.append(document['text'][index[0]:index[1]])

                writer.writerow([document['text'], hypothesis, label, span_text])  # Write the row to the CSV file

    print(f"Conversion complete. CSV file '{csv_file}' created.")


# convert train, test and validation json into csv files
json_to_csv('train.json', 'train.csv')
json_to_csv('test.json', 'test.csv')
json_to_csv('dev.json', 'validation.csv')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('validation.csv')

model_name2 = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name2)

def tokenize_data(data):
    return tokenizer(text=data['document'].tolist(), text_pair=data['hypothesis'].tolist(), truncation=True, padding="max_length", max_length=512)

label_mapping = {"NotMentioned": 0, "Entailment": 1, "Contradiction": 2}

train_labels = [label_mapping[label] for label in train_df['label'].tolist()]
valid_labels = [label_mapping[label] for label in val_df['label'].tolist()]
test_labels = [label_mapping[label] for label in val_df['label'].tolist()]

class ContractNLIDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}

        for key, value in self.embeddings.items():
            element_at_idx = value[idx]
            tensor_at_idx = torch.tensor(element_at_idx)
            item[key] = tensor_at_idx

        label_at_idx = self.labels[idx]
        label_tensor = torch.tensor(int(label_at_idx))
        item['labels'] = label_tensor

        return item

train_encodings = tokenize_data(train_df)
valid_encodings = tokenize_data(val_df)
test_encodings = tokenize_data(test_df)


train_dataset = ContractNLIDataset(train_encodings, train_labels)
valid_dataset = ContractNLIDataset(valid_encodings, valid_labels)


def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}



model2 = AutoModelForSequenceClassification.from_pretrained(model_name2, num_labels=3)




# wandb.login()

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="contract-nli-mobilebert",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.01,
#     "architecture": "MobileBERT",
#     "dataset": "Contract-NLI",
#     "epochs": 5,
#     }
# )

training_args = TrainingArguments(
    output_dir='./distilbert_model',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    # report_to="wandb",
)


trainer = Trainer(
    model=model2,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=metrics,

)

# for epoch in range(training_args.num_train_epochs):
# Training logic here

# Train the model
train_results = trainer.train()

# Evaluate the model on the validation dataset
eval_results = trainer.evaluate()

    # Log custom metrics to WandB
    # wandb.log({"epoch": epoch, "train_loss": train_results.loss, "eval_loss": eval_results.loss})
    
# Finish the WandB run
# wandb.finish()

# torch.save(model2.state_dict(), 'distilbert_model.pt')  


print("Training complete.")

class ContractNLIDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

test_encodings = tokenize_data(test_df)
test_labels = [label_mapping[label] for label in test_df['label'].tolist()]
test_dataset = ContractNLIDatasetTest(test_encodings)


# Use the Trainer.predict() method to get predictions
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=1)


confusion_mat = confusion_matrix(test_labels, pred_labels)
print("Confusion Matrix:")
print(confusion_mat)


# Compute evaluation metrics
accuracy = accuracy_score(test_labels, pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

class_names = list(label_mapping.keys())
print("\nClassification Report:")
print(classification_report(test_labels, pred_labels, target_names=class_names))

# Error Analysis

incorrect_predictions = []
for i in range(len(test_labels)):
    if test_labels[i] != pred_labels[i]:
        incorrect_predictions.append(i)

print("Number of incorrect predictions:", len(incorrect_predictions), " out of ", len(test_labels))

def error_analysis(predictions, true_labels, test_df):
    incorrect_classifications = []

    for i in range(len(predictions)):
        predicted_label = predictions[i]
        true_label = true_labels[i]

        text = test_df.iloc[i]['document']
        hypothesis = test_df.iloc[i]['hypothesis']

        if predicted_label != true_label:
            incorrect_classifications.append(
                {
                    "text": text,
                    "hypothesis": hypothesis,
                    "true_label": true_label,
                    "predicted_label": predicted_label
                }
            )
        
    return incorrect_classifications

incorrect_classifications = error_analysis(pred_labels, test_labels, test_df)

# Plot confusion matrix
plt.figure(figsize=(6,6))
plt.imshow(confusion_mat, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()
plt.savefig('distilbert_confusion_matrix.png')


