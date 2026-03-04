# K-means Clustering Results

---

## **Files Overview**

### **Visualizations**
- **`optimal_k_analysis.png`** - Analysis for selecting optimal number of clusters (K=2 to K=10)
  - Elbow method plot
  - Silhouette score analysis
  - Calinski-Harabasz score
  - Optimal K selection (K=3)

- **`kmeans_visualizations.png`** - Four-panel visualization of clustering results
  - PCA scatter plot showing cluster separation
  - Cluster centers in 2D space
  - Cluster size distribution
  - Income vs Expenditure by cluster

- **`feature_importance.png`** - Top 20 most important features from Random Forest
- **`correlation_heatmap.png`** - Correlation matrix of selected features

### **Data Files**
- **`clustered_households.csv`** - Original dataset with cluster assignments (13,886 households)
- **`cluster_statistics.csv`** - Detailed statistics for each cluster
- **`cluster_profiles.csv`** - Business-friendly cluster descriptions

---

## **Cluster Summary**

### **Cluster 0: High Income Savers (32.6%)**
- Income: $199,221 | Expenditure: $22,720 | Savings: 86.4%
- **Recommended Products**: Investment accounts, wealth management, tax optimization

### **Cluster 1: Zero Income Households (12.2%)**
- Income: $3,260 | Expenditure: $12,937 | Savings: -49.9%
- **Recommended Products**: Emergency assistance, debt consolidation, social programs

### **Cluster 2: Middle Income Families (55.2%)**
- Income: $49,963 | Expenditure: $8,074 | Savings: 79.7%
- **Recommended Products**: Savings accounts, mortgage products, insurance, retirement planning

---

## **Clustering Metrics**
- **Optimal K**: 3 clusters
- **Silhouette Score**: 0.184 (moderate clustering quality)
- **Calinski-Harabasz Score**: 2,117
- **Features Used**: 65 standardized variables
- **Total Households**: 13,886

---

## **Technical Details**
- **Algorithm**: K-means with random_state=42
- **Preprocessing**: StandardScaler (mean=0, std=1)
- **Dimensionality Reduction**: PCA for visualization
- **Feature Selection**: 65 numeric features (excluded IDs and categorical)

---

## **Usage Notes**
- Use `clustered_households.csv` for downstream analysis
- Refer to `cluster_profiles.csv` for business interpretations
- Visualizations are presentation-ready for stakeholders
- All files are ready for integration with recommendation systems
