import pandas as pd
from sklearn.cluster import KMeans
import pickle

# =========================
# LOAD CLEANED DATA
# =========================

df = pd.read_csv(
    "data/processed/cleaned_mobile_data.csv"
)

print("✅ Cleaned Data Loaded")

# =========================
# K-MEANS CLUSTERING
# =========================

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

clusters = kmeans.fit_predict(df)

# =========================
# ADD CLUSTER COLUMN
# =========================

df['Cluster'] = clusters

# =========================
# SAVE MODEL
# =========================

with open(
    "models/kmeans_model.pkl",
    "wb"
) as file:

    pickle.dump(kmeans, file)

# =========================
# SAVE CLUSTERED DATA
# =========================

df.to_csv(
    "data/processed/clustered_mobile_data.csv",
    index=False
)

print("✅ Clustering Completed")