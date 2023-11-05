import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model = torch.load('mobilebert_model.pt')

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
