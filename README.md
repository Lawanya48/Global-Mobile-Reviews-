# 📱 Mobile Product Segmentation & Recommendation System

## 📌 Project Overview

The **Mobile Product Segmentation & Recommendation System** is a complete end-to-end Machine Learning and Data Analytics project developed using Python, Scikit-Learn, and Streamlit.

The project focuses on:

* Mobile product segmentation using **K-Means Clustering**
* Product recommendation using **Cosine Similarity**
* Exploratory Data Analysis (EDA)
* Interactive dashboard creation using **Streamlit**

The system analyzes mobile product data, groups similar smartphones into meaningful market segments, and recommends similar products based on specifications and product similarity.

---

# 🎯 Objectives

The main objectives of this project are:

* Perform data cleaning and preprocessing
* Handle missing values and duplicate records
* Conduct Exploratory Data Analysis (EDA)
* Apply feature engineering and scaling
* Perform product segmentation using K-Means clustering
* Identify meaningful mobile market segments
* Build a similarity-based recommendation system
* Develop an interactive Streamlit dashboard
* Generate business insights for decision making

---

# 🧠 Machine Learning Workflow

## 1️⃣ Data Collection

* Imported mobile product dataset in CSV format
* Loaded dataset using Pandas DataFrame

## 2️⃣ Data Preprocessing

* Handled missing values
* Removed duplicate records
* Encoded categorical features
* Selected relevant features
* Standardized dataset using StandardScaler

## 3️⃣ Exploratory Data Analysis (EDA)

* Product distribution analysis
* Statistical summaries
* Correlation analysis
* Feature distribution analysis
* Outlier analysis

## 4️⃣ Clustering (Segmentation)

* Applied K-Means Clustering
* Used Elbow Method for optimal cluster selection
* Created 3 mobile product segments:

  * Budget Phones
  * Mid-Range Phones
  * Premium Phones

## 5️⃣ Recommendation System

* Implemented content-based recommendation system
* Used Cosine Similarity algorithm
* Recommended similar mobile products based on specifications

## 6️⃣ Streamlit Application

* Built interactive dashboard
* Added EDA visualizations
* Added clustering analysis
* Added recommendation interface

---

# 📊 Technologies Used

| Technology        | Purpose                   |
| ----------------- | ------------------------- |
| Python            | Programming Language      |
| Pandas            | Data Handling             |
| NumPy             | Numerical Operations      |
| Matplotlib        | Visualization             |
| Seaborn           | Statistical Visualization |
| Plotly            | Interactive Visualization |
| Scikit-Learn      | Machine Learning          |
| Streamlit         | Web Application           |
| K-Means           | Clustering                |
| Cosine Similarity | Recommendation System     |

---

# 📂 Project Structure

```text
Global Mobile Reviews/
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       └── clustered_mobile_data.csv
│
├── notebooks/
│   └── Mobile_Product_Segmentation.ipynb
│
├── Mobile Reviews Sentiment null.csv
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <repository-url>
```

## Navigate to Project Folder

```bash
cd "Global Mobile Reviews"
```

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Mobile_Product_Segmentation.ipynb
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app/app.py
```

---

# 📈 Elbow Method

The Elbow Method was used to determine the optimal number of clusters.

The elbow point was observed at:

```text
K = 3
```

Therefore, the dataset was segmented into:

1. Budget Phones
2. Mid-Range Phones
3. Premium Phones

---

# 🤖 Recommendation System

The recommendation system works using:

* Product specifications
* Feature similarity
* Cosine Similarity algorithm

When a user selects a smartphone model, the system recommends similar mobile phones based on:

* Specifications
* Ratings
* Product similarity
* Feature closeness

---

# 📊 Streamlit Dashboard Features

✅ Dataset Overview
✅ EDA Dashboard
✅ Correlation Heatmaps
✅ Cluster Visualization
✅ Elbow Method Analysis
✅ Product Recommendation System
✅ Interactive Graphs
✅ Business Insights

---

# 📌 Business Insights

* Budget phones dominate the lower pricing segment.
* Mid-range phones provide balanced performance and pricing.
* Premium phones contain flagship specifications.
* Clustering helps identify market segments effectively.
* Recommendation system improves customer product discovery.
* Similarity-based recommendations enhance user engagement.

---

# 📉 Evaluation Metrics

* Data preprocessing quality
* EDA effectiveness
* Clustering performance
* Elbow Method validation
* Silhouette Score analysis
* Recommendation relevance
* Visualization quality
* Streamlit application functionality

---

# 📷 Output Screens

* Dataset Dashboard
* EDA Analysis
* Elbow Method Visualization
* Cluster Analysis Dashboard
* Recommendation Dashboard

---

# 🚀 Future Improvements

* Deep Learning Recommendation System
* Hybrid Recommendation Engine
* Real-Time Product Recommendation
* Customer Review Sentiment Analysis
* Mobile Price Prediction
* Cloud Deployment

---

# 👨‍💻 Author

**Lawanya D**

Machine Learning & Data Analytics Project

---

# ✅ Conclusion

This project successfully implemented:

* Mobile product segmentation using K-Means Clustering
* Product recommendation using Cosine Similarity
* Interactive data visualization using Streamlit
* Complete end-to-end machine learning workflow

The project demonstrates the practical implementation of clustering and recommendation systems in the smartphone industry.
