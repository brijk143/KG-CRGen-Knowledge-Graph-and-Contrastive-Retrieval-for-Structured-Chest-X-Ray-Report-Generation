import pandas as pd
import numpy as np

def calculate_jaccard_accuracy(y_true, y_pred):
    """
    Calculate Jaccard accuracy for multi-label classification
    Jaccard = (Intersection) / (Union)
    y_true uses ';' as separator (from label column)
    y_pred uses '|' as separator (from predicted_classes column)
    """
    if isinstance(y_true, str):
        y_true = set(y_true.split(';'))
    else:
        y_true = set()
    
    if isinstance(y_pred, str):
        y_pred = set(y_pred.split('|'))
    else:
        y_pred = set()
    
    # Remove empty strings and strip whitespace
    y_true = {x.strip() for x in y_true if x.strip()}
    y_pred = {x.strip() for x in y_pred if x.strip()}
    
    # Calculate Jaccard similarity
    if len(y_true) == 0 and len(y_pred) == 0:
        return 1.0
    
    intersection = len(y_true & y_pred)
    union = len(y_true | y_pred)
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    return jaccard

# Open file for writing results
output_file = open('BiomedCLIP/Output/jaccard_accuracy_report.txt', 'w')

def log_output(text):
    """Write to both file and terminal"""
    output_file.write(text + '\n')
    print(text)

# Load both CSV files
log_output("Loading CSV files...")
predictions_df = pd.read_csv('BiomedCLIP/Output/predictions.csv')
indiana_df = pd.read_csv('testing/indiana_reports.csv')

log_output(f"\nPredictions CSV shape: {predictions_df.shape}")
log_output(f"Indiana Reports CSV shape: {indiana_df.shape}")

log_output("\n" + "="*80)
log_output("PREDICTIONS CSV COLUMNS")
log_output("="*80)
log_output(str(predictions_df.columns.tolist()))
log_output("\nFirst few rows of predictions:")
log_output(str(predictions_df.head(3)))

log_output("\n" + "="*80)
log_output("INDIANA REPORTS CSV COLUMNS")
log_output("="*80)
log_output(str(indiana_df.columns.tolist()))
log_output("\nFirst few rows of indiana_reports (uid, labels, label):")
log_output(str(indiana_df[['uid', 'labels', 'label']].head(3)))

# Merge dataframes on uid
log_output("\n" + "="*80)
log_output("MERGING DATAFRAMES")
log_output("="*80)
merged_df = pd.merge(predictions_df, indiana_df[['uid', 'label']], on='uid', how='inner')

log_output(f"Total UIDs in predictions: {len(predictions_df)}")
log_output(f"Total UIDs in indiana_reports: {len(indiana_df)}")
log_output(f"Total UIDs matched: {len(merged_df)}")

# Calculate Jaccard accuracy for each UID
jaccard_scores = []
results = []

log_output("\n" + "="*80)
log_output("CALCULATING JACCARD ACCURACY FOR EACH UID")
log_output("="*80)

for idx, row in merged_df.iterrows():
    uid = row['uid']
    predicted = row['predicted_classes']
    actual = row['label']
    
    jaccard = calculate_jaccard_accuracy(actual, predicted)
    jaccard_scores.append(jaccard)
    
    results.append({
        'uid': uid,
        'predicted_classes': predicted,
        'actual_labels': actual,
        'jaccard_accuracy': jaccard
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Calculate overall accuracy metrics
overall_jaccard = np.mean(jaccard_scores)
median_jaccard = np.median(jaccard_scores)
min_jaccard = np.min(jaccard_scores)
max_jaccard = np.max(jaccard_scores)
std_jaccard = np.std(jaccard_scores)

log_output("\n" + "="*80)
log_output("OVERALL JACCARD ACCURACY RESULTS")
log_output("="*80)
log_output(f"\n✓ Overall Jaccard Accuracy: {overall_jaccard:.6f} ({overall_jaccard*100:.4f}%)")
log_output(f"✓ Median Jaccard Accuracy: {median_jaccard:.6f} ({median_jaccard*100:.4f}%)")
log_output(f"✓ Min Jaccard Accuracy: {min_jaccard:.6f} ({min_jaccard*100:.4f}%)")
log_output(f"✓ Max Jaccard Accuracy: {max_jaccard:.6f} ({max_jaccard*100:.4f}%)")
log_output(f"✓ Standard Deviation: {std_jaccard:.6f}")
log_output(f"✓ Total UIDs evaluated: {len(jaccard_scores)}")

log_output("\n" + "="*80)
log_output("ACCURACY DISTRIBUTION")
log_output("="*80)
perfect = len([x for x in jaccard_scores if x == 1.0])
high = len([x for x in jaccard_scores if x >= 0.8])
medium = len([x for x in jaccard_scores if x >= 0.5])
low = len([x for x in jaccard_scores if x < 0.5])

log_output(f"\nPerfect matches (100% accuracy): {perfect} UIDs ({perfect/len(jaccard_scores)*100:.2f}%)")
log_output(f"High accuracy (≥80%): {high} UIDs ({high/len(jaccard_scores)*100:.2f}%)")
log_output(f"Medium accuracy (50-80%): {medium} UIDs ({medium/len(jaccard_scores)*100:.2f}%)")
log_output(f"Low accuracy (<50%): {low} UIDs ({low/len(jaccard_scores)*100:.2f}%)")

log_output("\n" + "="*80)
log_output("DETAILED CALCULATION EXAMPLES (First 5 UIDs)")
log_output("="*80)
for idx, row in results_df.head(5).iterrows():
    uid = row['uid']
    predicted_str = row['predicted_classes']
    actual_str = row['actual_labels']
    accuracy = row['jaccard_accuracy']
    
    # Parse to sets
    predicted_set = set(p.strip() for p in str(predicted_str).split(';') if p.strip())
    actual_set = set(a.strip() for a in str(actual_str).split(';') if a.strip())
    
    intersection = predicted_set & actual_set
    union = predicted_set | actual_set
    
    log_output(f"\n{'─'*80}")
    log_output(f"UID: {uid}")
    log_output(f"Predicted classes: {predicted_set}")
    log_output(f"Actual labels: {actual_set}")
    log_output(f"\nCalculation:")
    log_output(f"  Intersection (both predicted AND actual): {intersection}")
    log_output(f"  Union (either predicted OR actual): {union}")
    log_output(f"  Jaccard = |Intersection| / |Union|")
    log_output(f"  Jaccard = {len(intersection)} / {len(union)}")
    log_output(f"  Jaccard = {accuracy:.6f} ({accuracy*100:.4f}%)")

# Show worst performing UIDs
log_output("\n" + "="*80)
log_output("WORST PERFORMING UIDs (Lowest Accuracy)")
log_output("="*80)
worst_df = results_df.nsmallest(5, 'jaccard_accuracy')
for idx, row in worst_df.iterrows():
    uid = row['uid']
    predicted_str = row['predicted_classes']
    actual_str = row['actual_labels']
    accuracy = row['jaccard_accuracy']
    
    predicted_set = set(p.strip() for p in str(predicted_str).split(';') if p.strip())
    actual_set = set(a.strip() for a in str(actual_str).split(';') if a.strip())
    
    intersection = predicted_set & actual_set
    union = predicted_set | actual_set
    
    log_output(f"\n{'─'*80}")
    log_output(f"UID: {uid}")
    log_output(f"Predicted: {predicted_set}")
    log_output(f"Actual: {actual_set}")
    log_output(f"Jaccard = {len(intersection)} / {len(union)} = {accuracy:.6f} ({accuracy*100:.4f}%)")

# Show best performing UIDs
log_output("\n" + "="*80)
log_output("BEST PERFORMING UIDs (Highest Accuracy)")
log_output("="*80)
best_df = results_df.nlargest(5, 'jaccard_accuracy')
for idx, row in best_df.iterrows():
    uid = row['uid']
    predicted_str = row['predicted_classes']
    actual_str = row['actual_labels']
    accuracy = row['jaccard_accuracy']
    
    predicted_set = set(p.strip() for p in str(predicted_str).split(';') if p.strip())
    actual_set = set(a.strip() for a in str(actual_str).split(';') if a.strip())
    
    log_output(f"\nUID: {uid} | Accuracy: {accuracy:.6f} ({accuracy*100:.4f}%)")

# Save detailed results
results_df.to_csv('BiomedCLIP/Output/jaccard_accuracy_results.csv', index=False)
log_output("\n" + "="*80)
log_output(f"✓ Detailed results saved to 'BiomedCLIP/Output/jaccard_accuracy_results.csv'")
log_output(f"✓ Report saved to 'BiomedCLIP/Output/jaccard_accuracy_report.txt'")
log_output("="*80)

# Close file
output_file.close()
