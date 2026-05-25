# train_models.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# ==========================================
# 🛠️ PROCESS DATASET 1: LAYOUT & ORIENTATION
# ==========================================
print("⚡ Loading Dataset 1 (Layout & Orientation)...")
df1 = pd.read_csv('Pareto.csv', header=[0, 1], encoding='latin1')

clean_cols_1 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct',
    'unnamed'
]
df1.columns = clean_cols_1[:len(df1.columns)]
feature_cols_1 = ['window_to_wall', 'orientation', 'facade_type', 'shading_type', 'window_open_pct']

X1 = df1[feature_cols_1].copy()
encoder_1 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X1[['facade_type', 'shading_type']] = encoder_1.fit_transform(X1[['facade_type', 'shading_type']])

with open('encoder_layout.pkl', 'wb') as f:
    pickle.dump(encoder_1, f)

targets_1 = {'total_energy': 'total_energy', 'cooling_energy': 'cooling_energy', 'discomfort_hours': 'discomfort_hours'}
for name, col in targets_1.items():
    print(f"🤖 Training Layout Model for: {name}...")
    y1 = df1[col].copy()
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    with open(f'model_layout_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# ==========================================
# 🛠️ PROCESS DATASET 2: MATERIAL PROPERTIES
# ==========================================
print("\n⚡ Loading Dataset 2 (Material & Thermal Envelope)...")
df2 = pd.read_csv('2nd Optimization Results.csv', header=[0, 1], encoding='latin1')

clean_cols_2 = [
    'iteration', 'generation', 'category',
    'total_energy', 'discomfort_hours', 'cooling_energy',
    'external_wall', 'flat_roof', 'glazing_type', 'partition_wall',
    'unnamed'
]
df2.columns = clean_cols_2[:len(df2.columns)]
feature_cols_2 = ['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']

X2 = df2[feature_cols_2].copy()
encoder_2 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X2[feature_cols_2] = encoder_2.fit_transform(X2[feature_cols_2])

with open('encoder_materials.pkl', 'wb') as f:
    pickle.dump(encoder_2, f)

for name, col in targets_1.items():
    print(f"🤖 Training Materials Model for: {name}...")
    y2 = df2[col].copy()
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    with open(f'model_mat_{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

print("\n✅ Success! All models and encoders have been successfully compiled.")