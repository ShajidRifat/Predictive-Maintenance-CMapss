import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# 1. Setup column layouts
columns = [
    'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA',
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
    'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
    'PCNfR_dmd', 'W31', 'W32'
]

datasets = ['FD001', 'FD002', 'FD003', 'FD004']
all_train, all_test = [], []
offset = 0

print("Reading and stitching training/test data from CMaps...")
for fd in datasets:
    df_tr = pd.read_csv(f'CMaps/train_{fd}.txt', sep=r'\s+', header=None, names=columns)
    df_te = pd.read_csv(f'CMaps/test_{fd}.txt', sep=r'\s+', header=None, names=columns)
    rul = pd.read_csv(f'CMaps/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    
    # Apply unique offset to unit numbers so that units are unique across subsets
    df_tr['unit_number'] += offset
    df_te['unit_number'] += offset
    
    # Calculate EOL & RUL - Train Set
    eol_tr = df_tr.groupby('unit_number')['time_in_cycles'].transform('max')
    df_tr['EOL'] = eol_tr
    df_tr['RUL'] = eol_tr - df_tr['time_in_cycles']
    
    # Calculate EOL & RUL - Test Set
    eol_te = []
    for eng in df_te['unit_number']:
        cycles_run = len(df_te[df_te['unit_number'] == eng])
        actual_rul = rul[(eng - offset) - 1]
        eol_te.append(cycles_run + actual_rul)
    df_te['EOL'] = eol_te
    df_te['RUL'] = df_te['EOL'] - df_te['time_in_cycles']
    
    all_train.append(df_tr)
    all_test.append(df_te)
    offset += 1000

mega_train_df = pd.concat(all_train, ignore_index=True)
mega_test_df = pd.concat(all_test, ignore_index=True)
print(f"Stitched Training Set: {mega_train_df.shape[0]} rows")
print(f"Stitched Test Set    : {mega_test_df.shape[0]} rows")

# Drop unit number and EOL metadata
mega_train = mega_train_df.drop(columns=['unit_number', 'EOL'])
mega_test = mega_test_df.drop(columns=['unit_number', 'EOL'])

# Identify and drop constant features (std < 0.01)
flatline_cols = mega_train.columns[mega_train.std() < 0.01]
print("Dropped flatline features (zero-variance):", list(flatline_cols))

mega_train = mega_train.drop(columns=flatline_cols)
mega_test = mega_test.drop(columns=flatline_cols, errors='ignore')

# 2. Extract targets and features
X_train_raw = mega_train.drop(columns=['RUL'])
X_test_raw = mega_test.drop(columns=['RUL'])

# Target RUL with piecewise linear capping at 125 cycles
y_train_reg = mega_train['RUL'].clip(upper=125)
y_test_reg = mega_test['RUL'].clip(upper=125)

# Calculate Life Ratio (LR) and map to 3 discrete classification classes:
# Class 0: LR <= 0.6 (Healthy)
# Class 1: 0.6 < LR <= 0.8 (Warning)
# Class 2: LR > 0.8 (Critical / EOL Urgent)
# To match the notebooks, we use train_df/test_df's EOL to get LR:
LR_train = mega_train_df['time_in_cycles'] / mega_train_df['EOL']
LR_test = mega_test_df['time_in_cycles'] / mega_test_df['EOL']

y_train_class = []
for lr in LR_train:
    if lr <= 0.6: y_train_class.append(0)
    elif lr <= 0.8: y_train_class.append(1)
    else: y_train_class.append(2)

y_test_class = []
for lr in LR_test:
    if lr <= 0.6: y_test_class.append(0)
    elif lr <= 0.8: y_test_class.append(1)
    else: y_test_class.append(2)

y_train_class = np.array(y_train_class)
y_test_class = np.array(y_test_class)

# 3. Fit standard scaler
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

# 4. Train RandomForest Classifier & Regressor
os.makedirs('models', exist_ok=True)

print("🌲 Training RandomForest Regressor on Mega-Dataset... (approx. 10-15s)")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_scaled, y_train_reg)

print("🌲 Training RandomForest Classifier on Mega-Dataset... (approx. 5-10s)")
rf_class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_class.fit(X_train_scaled, y_train_class)

# Calculate test metrics to verify models are working perfectly
test_pred_reg = rf_reg.predict(X_test_scaled)
test_pred_class = rf_class.predict(X_test_scaled)

mae = np.mean(np.abs(y_test_reg - test_pred_reg))
acc = np.mean(y_test_class == test_pred_class)
print(f"\nModel training finished successfully!")
print(f"  Test RUL Regressor MAE  : {mae:.4f} cycles")
print(f"  Test Health Classifier Acc: {acc:.4%}")

# 5. Serialize and Save Model Assets
print("\nSerializing and saving model assets to models/ folder...")
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(rf_reg, 'models/rf_regressor.joblib')
joblib.dump(rf_class, 'models/rf_classifier.joblib')
print("All assets saved successfully! Directory listing of models/:")
print(os.listdir('models'))
