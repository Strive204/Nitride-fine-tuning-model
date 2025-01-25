import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import math

csv_file_path = 'validation_data.csv'

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"File not found: {csv_file_path}")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"File is empty: {csv_file_path}")
    exit(1)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

required_columns = [
    'Existing_Band_Gap_eV',
    'New_Band_Gap_eV',
    'Existing_Predicted_Formation_Energy_eV_per_atom',
    'New_Predicted_Formation_Energy_eV_per_atom'
]

for col in required_columns:
    if col not in df.columns:
        print(f"CSV file is missing necessary column: {col}")
        exit(1)

df_clean = df.dropna(subset=required_columns)

existing_band_gap = df_clean['Existing_Band_Gap_eV']
new_band_gap = df_clean['New_Band_Gap_eV']

existing_pfe = df_clean['Existing_Predicted_Formation_Energy_eV_per_atom']
new_pfe = df_clean['New_Predicted_Formation_Energy_eV_per_atom']

def plot_comparison(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6, edgecolors='w', s=100)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)

    r2 = r2_score(x, y)
    mse = mean_squared_error(x, y)
    rmse = math.sqrt(mse)

    plt.text(0.05, 0.95, f'$R^2$ = {r2:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

plot_comparison(
    x=existing_band_gap,
    y=new_band_gap,
    xlabel='Existing Band Gap (eV)',
    ylabel='New Band Gap (eV)',
    title='Existing vs New Band Gap',
    save_path='band_gap_comparison.png'
)

plot_comparison(
    x=existing_pfe,
    y=new_pfe,
    xlabel='Existing Predicted Formation Energy (eV/atom)',
    ylabel='New Predicted Formation Energy (eV/atom)',
    title='Existing vs New Predicted Formation Energy',
    save_path='predicted_formation_energy_comparison.png'
)
