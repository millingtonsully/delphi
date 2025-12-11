"""Quick script to analyze prediction scales."""
import pandas as pd
import numpy as np

df = pd.read_csv('evaluation_results/predictions.csv')

print('Prediction Scale Analysis:')
print('=' * 50)
print(f'Mean predicted: {df["predicted"].mean():.6f}')
print(f'Mean actual: {df["actual"].mean():.6f}')
print(f'Scale ratio (pred/actual): {df["predicted"].mean() / df["actual"].mean():.4f}')
print(f'Std predicted: {df["predicted"].std():.6f}')
print(f'Std actual: {df["actual"].std():.6f}')

print('\nPer-series scale ratios (first 10):')
for series_id in df['series_id'].unique()[:10]:
    sdf = df[df['series_id'] == series_id]
    ratio = sdf['predicted'].mean() / sdf['actual'].mean()
    print(f'  {series_id}: {ratio:.4f}')

print('\nScale ratio distribution:')
ratios = []
for series_id in df['series_id'].unique():
    sdf = df[df['series_id'] == series_id]
    ratio = sdf['predicted'].mean() / sdf['actual'].mean()
    ratios.append(ratio)

ratios = np.array(ratios)
print(f'  Mean ratio: {ratios.mean():.4f}')
print(f'  Std ratio: {ratios.std():.4f}')
print(f'  Min ratio: {ratios.min():.4f}')
print(f'  Max ratio: {ratios.max():.4f}')
print(f'  Series with ratio > 2.0: {(ratios > 2.0).sum()}')
print(f'  Series with ratio < 0.5: {(ratios < 0.5).sum()}')

