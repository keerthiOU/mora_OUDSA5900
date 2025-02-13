import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Load the input file
input_file_path = "/home/datasci03/input_files/preprocessed_300_file.txt"
output_file_path = "/home/datasci03/output_files/roberta_new.csv"

# Read tweets from the .txt file and split by commas
with open(input_file_path, 'r', encoding='utf-8') as f:
    tweets = f.read().split(',')

# Create a DataFrame from the list of tweets
df = pd.DataFrame(tweets, columns=['tweet_text'])

# Define function to predict sentiment for each chunk
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=514, truncation=True)

    # Debugging: Print token IDs and their lengths
    print(f"Input text: {text[:50]}...")  # Print first 50 characters for context
    print(f"Input IDs: {inputs['input_ids']}")
    print(f"Input length: {inputs['input_ids'].size(1)}")
    
    try:
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]
        labels = ["Negative", "Neutral", "Positive"]
        return labels[np.argmax(probabilities)]
    except Exception as e:
        print(f"Error processing text: {text}\n{e}")
        return "Error"

# Define function to handle long text by chunking
def predict_sentiment_long_text(text):
    max_length = 514
    tokens = tokenizer.tokenize(text)

    if len(tokens) > max_length:
        # Split tokens into chunks of 514
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        sentiments = []
        
        # Predict sentiment for each chunk
        for chunk in chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            sentiment = predict_sentiment(chunk_text)
            sentiments.append(sentiment)
        
        # Rule-based aggregation
        if "Negative" in sentiments:
            return "Negative"
        elif "Positive" in sentiments:
            return "Positive"
        else:
            return "Neutral"
    else:
        return predict_sentiment(text)

# Apply sentiment prediction on each tweet
df["sentiment"] = df["tweet_text"].apply(predict_sentiment_long_text)

# Save the results to a new CSV file
df[["tweet_text", "sentiment"]].to_csv(output_file_path, index=False)

print(f"Sentiment analysis complete. Results saved to {output_file_path}")

