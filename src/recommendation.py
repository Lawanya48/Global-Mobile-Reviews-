import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# =========================
# LOAD CLUSTERED DATA
# =========================

df = pd.read_csv(
    "data/processed/clustered_mobile_data.csv"
)

print("✅ Clustered Data Loaded")

# =========================
# REMOVE CLUSTER COLUMN
# =========================

feature_matrix = df.drop(
    columns=['Cluster']
)

# =========================
# CALCULATE SIMILARITY
# =========================

similarity_matrix = cosine_similarity(
    feature_matrix
)

# =========================
# SAVE SIMILARITY MATRIX
# =========================

with open(
    "models/similarity_matrix.pkl",
    "wb"
) as file:

    pickle.dump(similarity_matrix, file)

print("✅ Recommendation System Ready")

# =========================
# RECOMMEND FUNCTION
# =========================

def recommend_products(product_index):

    similarity_scores = list(
        enumerate(similarity_matrix[product_index])
    )

    sorted_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = sorted_scores[1:6]

    print("\n📱 Recommended Products:\n")

    for item in recommendations:

        print(df.iloc[item[0]])