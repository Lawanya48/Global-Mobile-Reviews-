import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/clustered_mobile_data.csv"
)

# =========================
# PRICE DISTRIBUTION
# =========================

plt.figure(figsize=(10, 5))

sns.histplot(
    df.iloc[:, 0],
    kde=True
)

plt.title("Distribution Plot")

plt.show()

# =========================
# CLUSTER VISUALIZATION
# =========================

plt.figure(figsize=(10, 5))

sns.scatterplot(
    x=df.iloc[:, 0],
    y=df.iloc[:, 1],
    hue=df['Cluster'],
    palette='viridis'
)

plt.title("Cluster Visualization")

plt.show()

# =========================
# HEATMAP
# =========================

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(),
    annot=False,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()