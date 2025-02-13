import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your ABSA dataset
df = pd.read_csv("total_1365_labled.csv")

# Display the first few rows to understand the structure
print(df.head())

# Check for NaN values in labels and remove rows with missing labels
df = df.dropna(subset=['Label'])

# Map sentiment labels to integers (if they are strings)
label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['Label'] = df['Label'].map(label_mapping)

# Check if there are any invalid label values
if df['Label'].isna().any():
    print("Found NaN values in 'Label' column after mapping. Cleaning up...")
    df = df.dropna(subset=['Label'])

# Check for any negative or out-of-bound label values (although this should be handled by the mapping)
invalid_labels = df[~df['Label'].isin([0, 1, 2])]
if not invalid_labels.empty:
    print(f"Found invalid labels:\n{invalid_labels}")

# Function to remove http links
def remove_links(text):
    return re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

# Create combined input for aspect-based sentiment analysis
df['tweet_text'] = df['tweet_text'].apply(remove_links)  # Remove links from tweet text
df['combined_input'] = df.apply(lambda x: f"{x['tweet_text']} [ASPECT] {x['handle_or_hash']}", axis=1)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", num_labels=3)  # 3 labels for positive, neutral, negative

# Preprocess the dataset (train-test split)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['combined_input'].tolist(), df['Label'].tolist(), test_size=0.2)

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert labels to tensor with the correct type (LongTensor)
train_labels = torch.tensor(train_labels).long()  # Ensure labels are of type LongTensor
test_labels = torch.tensor(test_labels).long()

# Create PyTorch dataset
class ABSA_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = ABSA_Dataset(train_encodings, train_labels)
test_dataset = ABSA_Dataset(test_encodings, test_labels)

# Define compute_metrics to evaluate accuracy, precision, recall, and F1-score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Convert logits to class predictions
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",  # Evaluate after every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Include metrics for evaluation
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_roberta_absa_new")
tokenizer.save_pretrained("fine_tuned_roberta_absa_new")

# Evaluate the model on the test dataset
eval_results = trainer.evaluate()

# Print the evaluation results
print(f"Test Set Results: {eval_results}")

