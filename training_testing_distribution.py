import pandas as pd
from collections import Counter

# Load both CSV files
training_df = pd.read_csv('/Users/bkishor/Desktop/kg_new/training2/reports.csv')
testing_df = pd.read_csv('/Users/bkishor/Desktop/kg_new/testing2/reports.csv')

# Function to extract and count classes
def get_class_distribution(df, name):
    """Extract classes from 'labels' column and count distribution"""
    # Parse the labels column (format: ['class1', 'class2', ...])
    all_classes = []
    for labels_str in df['labels'].dropna():
        # Remove brackets and quotes, split by comma
        labels = str(labels_str).strip("[]'\"").split("', '")
        labels = [l.strip().strip("'\"") for l in labels]
        all_classes.extend(labels)
    
    class_counts = Counter(all_classes)
    return pd.Series(class_counts).sort_values(ascending=False)

# Get distributions
train_dist = get_class_distribution(training_df, "Training")
test_dist = get_class_distribution(testing_df, "Testing")

# Create comparison dataframe
comparison = pd.DataFrame({
    'Training Count': train_dist,
    'Testing Count': test_dist,
    'Training %': (train_dist / train_dist.sum() * 100).round(2),
    'Testing %': (test_dist / test_dist.sum() * 100).round(2),
}).fillna(0)

comparison['Count Diff'] = comparison['Training Count'] - comparison['Testing Count']
comparison['% Diff'] = comparison['Training %'] - comparison['Testing %']

print("="*100)
print("CLASS DISTRIBUTION COMPARISON: Training vs Testing")
print("="*100)
print(comparison.to_string())

print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"Training - Total samples: {train_dist.sum()}, Unique classes: {len(train_dist)}")
print(f"Testing  - Total samples: {test_dist.sum()}, Unique classes: {len(test_dist)}")
print(f"\nClasses only in Training: {set(train_dist.index) - set(test_dist.index)}")
print(f"Classes only in Testing: {set(test_dist.index) - set(train_dist.index)}")

# Check if distributions are similar (chi-square or visual comparison)
import numpy as np
common_classes = set(train_dist.index) & set(test_dist.index)
if common_classes:
    max_diff = comparison.loc[comparison.index.isin(common_classes), '% Diff'].abs().max()
    print(f"\nMax percentage point difference (common classes): {max_diff:.2f}%")