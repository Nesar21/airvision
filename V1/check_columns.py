import pandas as pd

print("="*80)
print("CHECKING CSV COLUMN NAMES")
print("="*80)

fold1 = pd.read_csv('stage2_results/fold1_predictions.csv')
print("\nFold 1 columns:")
print(fold1.columns.tolist())
print("\nFirst 3 rows:")
print(fold1.head(3))
print("="*80)
