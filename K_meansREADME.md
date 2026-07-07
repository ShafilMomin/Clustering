# 🛍️ Mall Customer Segmentation using K-Means Clustering

An unsupervised machine learning project that segments mall customers into distinct groups based on **Annual Income** and **Spending Score** using the K-Means Clustering algorithm.

---

## 📌 Problem Statement

A mall wants to understand its customers better. Instead of treating all customers the same, we can group them by **spending behaviour** — allowing the marketing team to target each group with the right strategy.

This is an **unsupervised learning** problem — there are no pre-defined labels. The algorithm discovers the groups on its own!

---

## 📂 Repository Structure

```
├── k_means_clustering.ipynb    # Main notebook with full pipeline
├── Mall_Customers.csv          # Dataset (200 customers)
└── README.md
```

---

## 📊 Dataset

**File:** `Mall_Customers.csv`
**Rows:** 200 customers
**Columns:** 5 features

| Column | Description |
|---|---|
| CustomerID | Unique customer ID |
| Genre | Male / Female |
| Age | Customer's age |
| **Annual Income (k$)** | Annual income in thousands ← Used for clustering |
| **Spending Score (1-100)** | Mall-assigned spending score ← Used for clustering |

Only columns **Annual Income** and **Spending Score** were used for clustering (index 3 and 4).

---

## ⚙️ Project Pipeline

### Part 1 — Load & Select Features

```python
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# Selecting only: Annual Income (k$) and Spending Score (1-100)
```

---

### Part 2 — Finding Optimal Number of Clusters (Elbow Method)

Before clustering, we need to know **how many clusters (k) to use**. The Elbow Method helps us find it:

```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

**WCSS** = Within Cluster Sum of Squares — measures how tight each cluster is.

```
WCSS
 |
 | *
 |   *
 |     *
 |       * ← Elbow here! k=5
 |          * * * * *
 |_________________________ k
   1  2  3  4  5  6  7

The "elbow" bend at k=5 = optimal number of clusters!
```

---

### Part 3 — Training K-Means with k=5

```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
```

**Why k-means++?**
Standard K-Means picks initial centroids randomly — can give bad results. `k-means++` picks smart initial centroids that are spread out, giving faster and more accurate convergence!

---

### Part 4 — Visualising the Clusters

```python
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red',     label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue',    label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green',   label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan',    label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

---

## 📈 Results — 5 Customer Segments Discovered

| Cluster | Annual Income | Spending Score | Profile |
|---|---|---|---|
| **Cluster 1** 🔴 | Low | High | Spend a lot despite low income |
| **Cluster 2** 🔵 | Low | Low | Low income, careful spenders |
| **Cluster 3** 🟢 | Medium | Medium | Average customers |
| **Cluster 4** 🩵 | High | High | ⭐ High value — target these! |
| **Cluster 5** 🟣 | High | Low | Rich but don't spend much |

> ⭐ **Key Insight:** Cluster 4 (High Income + High Spending) = Premium customers — best ROI for marketing campaigns!

---

## 🧠 How K-Means Works

```
Step 1: Randomly place k centroids (k-means++ does this smartly)

Step 2: Assign each customer to nearest centroid

Step 3: Recalculate centroid = mean of all points in cluster

Step 4: Repeat Steps 2-3 until centroids stop moving

Done! Clusters are stable ✅
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Loading dataset |
| `numpy` | Array operations |
| `matplotlib` | Elbow plot + Cluster scatter plot |
| `scikit-learn` | KMeans algorithm |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**3. Run the notebook**
Open `k_means_clustering.ipynb` in Jupyter Notebook or Google Colab and run all cells.

---

## 🧠 Key Concepts Used

- **Unsupervised Learning** — No labels, algorithm finds groups on its own
- **K-Means Clustering** — Partition-based clustering algorithm
- **Elbow Method** — Finding optimal number of clusters via WCSS
- **WCSS** — Within Cluster Sum of Squares (measures cluster tightness)
- **k-means++** — Smart centroid initialization for better results
- **Customer Segmentation** — Real-world business application of clustering

---

## 👤 Author

**Shafil** — AI & ML Engineering Student
B.Tech Final Year | Specialization: Computer Vision & Deep Learning

---
