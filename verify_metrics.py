"""Quick verification script for MSE and MAE calculations."""
import pandas as pd
import numpy as np

# Read the summary
summary = pd.read_csv('evaluation_results/series_summary.csv')

# Calculate overall metrics
overall_mse = summary['mse'].mean()
overall_mae = summary['mae'].mean()

print("=" * 60)
print("VERIFICATION OF METRICS CALCULATIONS")
print("=" * 60)
print(f"\nNumber of series: {len(summary)}")
print(f"\nOverall MSE (averaged across series): {overall_mse:.10f}")
print(f"Overall MAE (averaged across series): {overall_mae:.10f}")
print(f"\nExpected from evaluation: MSE: 0.0004, MAE: 0.0119")
print(f"\nDifference:")
print(f"  MSE: {abs(overall_mse - 0.0004):.10f}")
print(f"  MAE: {abs(overall_mae - 0.0119):.10f}")

# Verify a few individual series manually
print("\n" + "=" * 60)
print("VERIFYING INDIVIDUAL SERIES CALCULATIONS")
print("=" * 60)

predictions_df = pd.read_csv('evaluation_results/predictions.csv')

# Check first series
series_0 = predictions_df[predictions_df['series_id'] == 'br_female_outerwear_0']
mse_0_manual = np.mean((series_0['predicted'] - series_0['actual'])**2)
mae_0_manual = np.mean(np.abs(series_0['predicted'] - series_0['actual']))
mse_0_summary = summary[summary['series_id'] == 'br_female_outerwear_0']['mse'].values[0]
mae_0_summary = summary[summary['series_id'] == 'br_female_outerwear_0']['mae'].values[0]

print(f"\nSeries 0 (br_female_outerwear_0):")
print(f"  MSE (manual): {mse_0_manual:.10f}")
print(f"  MSE (summary): {mse_0_summary:.10f}")
print(f"  Match: {abs(mse_0_manual - mse_0_summary) < 1e-10}")
print(f"  MAE (manual): {mae_0_manual:.10f}")
print(f"  MAE (summary): {mae_0_summary:.10f}")
print(f"  Match: {abs(mae_0_manual - mae_0_summary) < 1e-10}")

# Check a few more series
for i in [1, 5, 10]:
    series_id = f'br_female_outerwear_{i}'
    series_data = predictions_df[predictions_df['series_id'] == series_id]
    mse_manual = np.mean((series_data['predicted'] - series_data['actual'])**2)
    mae_manual = np.mean(np.abs(series_data['predicted'] - series_data['actual']))
    mse_summary = summary[summary['series_id'] == series_id]['mse'].values[0]
    mae_summary = summary[summary['series_id'] == series_id]['mae'].values[0]
    
    print(f"\nSeries {i} ({series_id}):")
    print(f"  MSE match: {abs(mse_manual - mse_summary) < 1e-10}")
    print(f"  MAE match: {abs(mae_manual - mae_summary) < 1e-10}")

# Check data ranges
print("\n" + "=" * 60)
print("DATA RANGES (to verify scale)")
print("=" * 60)
print(f"\nPredictions range: [{predictions_df['predicted'].min():.6f}, {predictions_df['predicted'].max():.6f}]")
print(f"Actuals range: [{predictions_df['actual'].min():.6f}, {predictions_df['actual'].max():.6f}]")
print(f"Errors range: [{predictions_df['error'].min():.6f}, {predictions_df['error'].max():.6f}]")
print(f"Absolute errors range: [{predictions_df['abs_error'].min():.6f}, {predictions_df['abs_error'].max():.6f}]")

print(f"\nMSE range across series: [{summary['mse'].min():.10f}, {summary['mse'].max():.10f}]")
print(f"MAE range across series: [{summary['mae'].min():.10f}, {summary['mae'].max():.10f}]")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

