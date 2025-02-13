import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# Load and split data
tweets_file_path = '/home/datasci03/input_files/data_pretrain.txt'
labels_file_path = '/home/datasci03/input_files/data_pretrain_labelled.csv'

# Read tweets (comma-separated) and labels
with open(tweets_file_path, 'r', encoding='utf-8') as f:
     tweets = f.read().split(',')

labels_df = pd.read_csv(labels_file_path)
labels = labels_df['Label'].values

# Ensure labels are integers (if they're not already)
unique_labels = list(set(labels))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
encoded_labels = [label_to_id[label] for label in labels]

# Create a DataFrame
data = pd.DataFrame({'tweet_text': tweets, 'Label': encoded_labels})

#data = '/home/datasci03/input_files/data_pretrain_labelled.csv'
# Split into train and test sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['tweet_text'].tolist(), data['Label'].tolist(), test_size=0.2, random_state=42
)

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels=len(unique_labels))

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=514)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=514)

# Convert to Dataset objects
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})

# Define evaluation metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.tensor(np.argmax(logits, axis=-1))  # Convert to tensor
    labels = torch.tensor(labels)  # Ensure labels are tensor
    precision, recall, f1, _ = precision_recall_fscore_support(labels.numpy(), predictions.numpy(), average='weighted')
    acc = accuracy_score(labels.numpy(), predictions.numpy())
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

