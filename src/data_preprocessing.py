import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

file_path = "Mobile Reviews Sentiment null.csv"

df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully")

# =========================
# DISPLAY BASIC INFO
# =========================

print(df.head())

print(df.info())

# =========================
# HANDLE MISSING VALUES
# =========================

df.fillna(method='ffill', inplace=True)

# =========================
# REMOVE DUPLICATES
# =========================

df.drop_duplicates(inplace=True)

# =========================
# SELECT IMPORTANT FEATURES
# =========================

selected_columns = []

for col in df.columns:
    selected_columns.append(col)

df = df[selected_columns]

# =========================
# ENCODE CATEGORICAL DATA
# =========================

label_encoders = {}

for column in df.select_dtypes(include=['object']).columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column].astype(str))

    label_encoders[column] = le

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

# =========================
# SAVE SCALER
# =========================

with open("models/scaler.pkl", "wb") as file:

    pickle.dump(scaler, file)

# =========================
# SAVE CLEANED DATA
# =========================

scaled_df.to_csv(
    "data/processed/cleaned_mobile_data.csv",
    index=False
)

print("✅ Data Preprocessing Completed")