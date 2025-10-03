# Image Clustering with Deep Learning Features

## 🎯 Project Overview

This project implements unsupervised machine learning algorithms to cluster flower images based on their visual similarities. Using deep learning feature extraction and advanced clustering techniques, the system automatically groups images without prior knowledge of their categories.

## 📊 Dataset

- **Size:** 210 flower images (PNG format)
- **Format:** Color images with varying dimensions
- **Ground Truth:** CSV file containing true labels for evaluation
- **Structure:** Images numbered from 0001.png to 0210.png

## 🔧 Technical Approach

### Feature Extraction
- **Pre-trained Model:** VGG16 (ImageNet weights)
- **Input Processing:** Images resized to 256x256 pixels
- **Feature Dimensionality:** High-dimensional feature vectors extracted from VGG16's penultimate layer
- **Preprocessing:** BGR conversion and zero-centering using ImageNet statistics

### Clustering Algorithms

#### 1. K-Means Clustering
- **Algorithm:** Centroid-based partitioning
- **K Selection:** Elbow method with homogeneity score optimization
- **Optimal K:** 10 clusters
- **Initialization:** K-means++ for robust centroid initialization

#### 2. DBSCAN (Density-Based Spatial Clustering)
- **Algorithm:** Density-based clustering with noise detection
- **Parameters:** 
  - ε (epsilon): 0.04 (maximum distance for neighborhood)
  - MinPts: 2 (minimum points for core point definition)
- **Advantages:** Automatic cluster detection, outlier identification

### Dimensionality Reduction
- **Method:** Principal Component Analysis (PCA)
- **Purpose:** Visualization and computational efficiency
- **Components:** 2D projection for visualization, higher dimensions for clustering

## 🛠️ Implementation Details

### Dependencies
```python
# Core Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Deep Learning & Image Processing
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from PIL import Image

# Machine Learning
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import silhouette_score, homogeneity_score
from sklearn.neighbors import NearestNeighbors
```

### Key Hyperparameters
- `CLUSTERS_NUM = 10` - Number of clusters for K-means
- `N_INIT = 10` - Number of K-means initializations
- `EPSILON = 0.04` - DBSCAN neighborhood radius
- `MINPTS = 2` - DBSCAN minimum points for core point
- `PCA_DIMENSION = 2` - Dimensionality for visualization

### Core Functions

#### Feature Extraction Pipeline
```python
def pre_extract_vgg(files):
    """Extract VGG16 features from image files"""
    # Load and preprocess images
    # Extract features using pre-trained VGG16
    # Return flattened feature vectors

def extract_for_all(directory):
    """Process entire image directory"""
    # Extract features for all images
    # Load ground truth labels
    # Return features and labels
```

#### Clustering Analysis
```python
def calc_scores_kmeans(dimension):
    """Calculate clustering performance metrics"""
    # Apply PCA for dimensionality reduction
    # Perform K-means clustering
    # Calculate silhouette and homogeneity scores
    # Return results for visualization
```

## 📈 Results & Performance

### K-Means Clustering
- **Silhouette Score:** Optimized through multiple iterations
- **Homogeneity Score:** Evaluated against ground truth labels
- **Cluster Quality:** Well-separated spherical clusters
- **Visualization:** Clear cluster boundaries in PCA space

### DBSCAN Clustering
- **Parameter Optimization:** Grid search over ε and MinPts
- **Noise Detection:** Automatic outlier identification
- **Cluster Shapes:** Handles arbitrary cluster geometries
- **Performance:** Compared using composite scoring function

### Evaluation Metrics

#### Silhouette Score
- **Range:** [-1, 1]
- **Interpretation:** 
  - +1: Perfect clustering
  - 0: Overlapping clusters
  - -1: Incorrect assignments
- **Formula:** $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$

#### Homogeneity Score
- **Range:** [0, 1]
- **Purpose:** Measures cluster purity relative to true labels
- **Formula:** $H(C,K) = 1 - \frac{H(C|K)}{H(C)}$

## 🔍 Key Findings

### Algorithm Comparison
1. **K-Means Advantages:**
   - Better performance on spherical clusters
   - Computationally efficient
   - Consistent results with proper initialization

2. **DBSCAN Advantages:**
   - Automatic cluster number detection
   - Robust to outliers
   - Handles arbitrary cluster shapes

### Feature Extraction Benefits
- **Semantic Understanding:** VGG16 captures high-level visual features
- **Dimensionality Reduction:** More efficient than raw pixel processing
- **Transfer Learning:** Leverages ImageNet knowledge for flower classification

## 🚀 Usage Instructions

### Setup
```bash
# Install required packages
pip install tensorflow keras scikit-learn matplotlib pandas pillow numpy

# Ensure proper directory structure
./flower_images/
├── 0001.png
├── 0002.png
├── ...
└── labels.csv
```

### Running the Analysis
```python
# 1. Load and extract features
dir = "./flower_images/"
features, df = extract_for_all(dir)
true_groups = df["label"].tolist()

# 2. Apply clustering algorithms
# K-Means
kmeans = KMeans(n_clusters=CLUSTERS_NUM, n_init=N_INIT)
groups_kmeans = kmeans.fit_predict(features_pca)

# DBSCAN
dbscan = DBSCAN(eps=EPSILON, min_samples=MINPTS)
groups_dbscan = dbscan.fit_predict(features_pca)

# 3. Visualize results
# Plot clusters in PCA space
# Display sample images from each cluster
```

## 📋 Project Structure

```
AI_CA3_810101381/
├── CA3.ipynb                 # Main notebook with analysis
├── CA3.html                  # HTML export of notebook
├── README.md                 # This file
├── Description/
│   └── AI-A3.pdf            # Project requirements
└── flower_images/
    ├── 0001.png - 0210.png  # Image dataset
    └── labels.csv           # Ground truth labels
```

## 🔧 Optimization Strategies

### For K-Means
1. **Feature Scaling:** Normalize features for equal contribution
2. **Optimal K Selection:** Use elbow method with domain-specific metrics
3. **Initialization:** Multiple runs with k-means++ initialization
4. **Dimensionality Reduction:** PCA for computational efficiency

### For DBSCAN
1. **Parameter Tuning:** K-distance plots for ε selection
2. **Feature Preprocessing:** Scaling for distance metric consistency
3. **Distance Metrics:** Experiment with different distance functions
4. **Noise Analysis:** Investigate outlier patterns

### General Improvements
1. **Feature Engineering:** Advanced CNN architectures (ResNet, EfficientNet)
2. **Ensemble Methods:** Combine multiple clustering results
3. **Validation Techniques:** Cross-validation for robust evaluation
4. **Visualization:** t-SNE for non-linear dimensionality reduction

## 📚 Theoretical Background

### Why Feature Extraction?
Raw pixel processing suffers from:
- **High Dimensionality:** Computational complexity and curse of dimensionality
- **Lack of Abstraction:** Missing semantic information
- **Noise Sensitivity:** Vulnerable to lighting and scale variations

VGG16 features provide:
- **Semantic Representation:** High-level visual concepts
- **Dimensionality Reduction:** Compressed meaningful features
- **Transfer Learning:** Pre-trained knowledge from ImageNet

### Clustering Algorithm Selection
- **Spherical Clusters:** K-means excels with uniform, spherical distributions
- **Arbitrary Shapes:** DBSCAN handles complex geometries and noise
- **Dataset Characteristics:** Flower images exhibit relatively uniform cluster properties

## 🎯 Conclusion

This project successfully demonstrates the application of deep learning feature extraction combined with classical clustering algorithms for image grouping. The VGG16-based feature extraction provides meaningful representations that enable both K-means and DBSCAN to identify distinct flower groups. While K-means showed superior performance on this dataset due to the spherical nature of the clusters, DBSCAN provided valuable insights into data density and outlier detection.

The comprehensive evaluation using both silhouette and homogeneity scores provides a robust assessment of clustering quality, balancing internal cluster cohesion with external validation against ground truth labels.
