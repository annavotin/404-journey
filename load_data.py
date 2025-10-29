from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

# Create output directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load training data
newsgroups_train = fetch_20newsgroups(subset='train')
train_df = pd.DataFrame({
    'text': newsgroups_train.data,
    'category': newsgroups_train.target,
    'category_name': [newsgroups_train.target_names[i] for i in newsgroups_train.target]
})

# Load test data
newsgroups_test = fetch_20newsgroups(subset='test')
test_df = pd.DataFrame({
    'text': newsgroups_test.data,
    'category': newsgroups_test.target,
    'category_name': [newsgroups_test.target_names[i] for i in newsgroups_test.target]
})

# Save as parquet files
train_df.to_parquet('data/20newsgroups_train.parquet')
test_df.to_parquet('data/20newsgroups_test.parquet')

print(f"Training set saved: {len(train_df)} documents")
print(f"Test set saved: {len(test_df)} documents")
print("Files saved to data/20newsgroups_train.parquet and data/20newsgroups_test.parquet")
