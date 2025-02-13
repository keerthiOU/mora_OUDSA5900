import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer from the base model path
base_model_path = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load the fine-tuned model from the checkpoint directory
model_path = '/home/datasci03/test/results/checkpoint-411'
model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=False, from_flax=False)

# Load new data
new_data_file_path = '/home/datasci03/input_files/data_pretrain.txt'
with open(new_data_file_path, 'r', encoding='utf-8') as f:
    new_tweets = f.read().split(',')

# Define label-to-id mapping based on training labels
labels_file_path = '/home/datasci03/input_files/data_pretrain_labelled.csv'
labels_df = pd.read_csv(labels_file_path)
unique_labels = labels_df['Label'].unique()
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}  # Reverse the dictionary

# Tokenize new data
encodings = tokenizer(new_tweets, truncation=True, padding=True, max_length=514, return_tensors="pt")

# Predict
model.eval()  # Set to evaluation mode
with torch.no_grad():
    outputs = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# Convert predictions to labels
predicted_labels = [id_to_label[pred.item()] for pred in predictions]

# Save the results to a DataFrame
output_df = pd.DataFrame({
    'tweet_text': new_tweets,
    'predicted_sentiment': predicted_labels
})

# Save to CSV
output_file_path = '/home/datasci03/output_files/predictions_roberta.csv'
output_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")

