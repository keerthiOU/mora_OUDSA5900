import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the tokenizer and model
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Read the labeled data
input_file_path = '/home/datasci03/input_files/data_pretrain_labelled.csv'
data = pd.read_csv(input_file_path)

# Map string labels to integers
label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
data['Label'] = data['Label'].map(label_map)

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['tweet_text'].tolist(),
    data['Label'].tolist(),
    test_size=0.25,
    random_state=42
)

# Tokenize the input texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create a dataset class
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = TweetDataset(train_encodings, train_labels)
test_dataset = TweetDataset(test_encodings, test_labels)

# Define compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics  # Include the metrics function
)

# Start training
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Metrics:", eval_results)

# Save the model after training
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model training complete and saved!")

# Load the fine-tuned model for labeling
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# Load the preprocessed tweet data
preprocessed_input_file_path = '/home/datasci03/input_files/preprocessed_300_file.csv'
preprocessed_data = pd.read_csv(preprocessed_input_file_path)

# Process each tweet for sentiment analysis
preprocessed_tweet_texts = preprocessed_data['tweet_text'].tolist()
results = []
for tweet in preprocessed_tweet_texts:
    if len(tweet) > 512:
        tweet = tweet[:512]  # Truncate to the first 512 characters
    try:
        inputs = fine_tuned_tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = fine_tuned_model(**inputs)
        predicted_class = outputs.logits.argmax(dim=1).item()
        sentiment_label = {0: "Positive", 1: "Negative", 2: "Neutral"}
        results.append(sentiment_label[predicted_class])
    except Exception as e:
        print(f"Error processing tweet: {tweet}\nException: {e}")
        results.append("Error")

# Save labeled data to CSV
output_data = [{'tweet_text': tweet, 'sentiment': result} for tweet, result in zip(preprocessed_tweet_texts, results)]
df_output = pd.DataFrame(output_data)
output_file_path = '/home/datasci03/output_files/labeled_preprocessed_300_file.csv'
df_output.to_csv(output_file_path, index=False)

print(f"Labeled tweets saved to {output_file_path}")
