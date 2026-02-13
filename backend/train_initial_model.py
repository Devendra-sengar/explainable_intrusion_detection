import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os

# Load the data
data_path = "../CICIDS2017_small_balanced.csv"
model_dir = "../data"
model_path = os.path.join(model_dir, "Hybrid_Adaptive_XGBoost_Model.pkl")

# Create data directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Read and prepare data
print("Loading data...")
df = pd.read_csv(data_path)

# Select relevant features (matching our frontend form)
selected_features = [
    'Flow Duration',  # duration
    'Total Length of Fwd Packets',  # src_bytes
    'Total Length of Bwd Packets',  # dst_bytes
    'Protocol',  # protocol_type (we'll add a dummy one)
    'Service',  # service (we'll add a dummy one)
    'FIN Flag Count',  # flag (we'll convert this to a flag type)
]

# Add dummy protocol and service columns for demonstration
df['Protocol'] = 'tcp'
df['Service'] = df['Destination Port'].apply(lambda x: 'http' if x == 80 else 'other')

# Convert flag counts to flag types
df['Flag'] = df.apply(lambda row: 'SF' if row['ACK Flag Count'] > 0 else 'REJ', axis=1)

# Prepare features
X = pd.DataFrame({
    'duration': df['Flow Duration'],
    'protocol_type': df['Protocol'],
    'service': df['Service'],
    'flag': df['Flag'],
    'src_bytes': df['Total Length of Fwd Packets'],
    'dst_bytes': df['Total Length of Bwd Packets'],
})

# Prepare target (binary classification: attack vs benign)
y = (df['_label_binary'] == 1).astype(int)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le_dict = {}
categorical_columns = ['protocol_type', 'service', 'flag']
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    le_dict[column] = le

print("Training model...")
# Train initial model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X, y)

print("Saving model...")
# Save model and label encoders
with open(model_path, 'wb') as f:
    pickle.dump({'model': model, 'label_encoders': le_dict}, f)

print("Model trained and saved successfully!")
print(f"Model saved at: {model_path}")

# Create a sample batch for testing
print("\nCreating sample batch for testing...")
sample_batch = pd.DataFrame({
    'duration': [1000, 2000],
    'protocol_type': ['tcp', 'tcp'],
    'service': ['http', 'other'],
    'flag': ['SF', 'REJ'],
    'src_bytes': [1000, 0],
    'dst_bytes': [2000, 0]
})

sample_batch.to_csv("../data/sample_batch.csv", index=False)
print("Sample batch saved at: ../data/sample_batch.csv")