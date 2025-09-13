import numpy as np
import pandas as pd

# Load inputs
df_energy = pd.read_parquet('cache/df_energy.parquet')
df_result = pd.read_parquet('cache/df_result.parquet')

df_check = df_energy.copy()

# Calculate direct export/import without battery
interval_hours = 15 / 60  # 0.25 hours

df_check['GridExport'] = np.maximum(
    df_check['HomeGeneration'] - df_check['HomeConsumption'], 0
)

df_check['GridImport'] = np.maximum(
    df_check['HomeConsumption'] - df_check['HomeGeneration'], 0
)

# Calculate profit (same formula as battery case)
df_check['Profit'] = (
    df_check['GridExport'] * df_check['PriceSignal'] * interval_hours / 1000 -
    df_check['GridImport'] * df_check['PriceSignal'] * interval_hours / 1000
)
cumulative_profit_check = df_check['Profit'].cumsum()

# Comparison with battery optimization results
df_compare = pd.DataFrame({
    'With Battery': df_result['Profit'].cumsum(),
    'No Battery': cumulative_profit_check
}, index=df_check.index)

