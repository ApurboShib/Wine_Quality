# Wine Quality Prediction

## Introduction

The **Wine Quality Dataset** is a well-structured collection of Portuguese "Vinho Verde" wines, including both **red** and **white** variants. It's widely used in machine learning applications for **predictive modeling**, especially for **regression** and **classification** tasks. The dataset features 11 **physicochemical attributes**, such as acidity, sugar content, and alcohol percentage, all of which influence the overall **quality** of the wine. The target variable is a **quality score** (ranging from 0 to 10), determined by expert sensory evaluations.

With:
- **1,599 red wine samples**  
- **4,898 white wine samples**

This dataset is highly valuable for **real-world applications** in the wine industry. Machine learning models trained on this data can help winemakers:
- Optimize production
- Improve quality control
- Enhance customer satisfaction

The dataset is particularly suited for both:
- **Regression** (predicting the exact quality score)
- **Classification** (categorizing wines into "good" or "bad" quality groups)

Its well-defined features make it easily interpretable and applicable to both **business** and **research scenarios**.

## Key Features of the Dataset

### Input Variables (11 Physicochemical Attributes):
- **Fixed acidity**
- **Volatile acidity**
- **Citric acid**
- **Residual sugar**
- **Chlorides**
- **Free sulfur dioxide**
- **Total sulfur dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**

### Size of the Dataset:
- **Red wine**: 1,599 samples  
- **White wine**: 4,898 samples

## Dataset Source & Preprocessing

### Source:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine)

### Preprocessing Steps:
- **Handling Missing Values**: Drop or impute missing data (rare in this dataset).
- **Feature Scaling**: 
  - **StandardScaler** for SVM
  - **MinMaxScaler** for neural networks
- **Class Balancing (Classification)**: Oversample minority classes ("high-quality" wines).
- **Feature Engineering**:
  - Create interaction terms (e.g., `alcohol × acidity`).
  - Derive sulfur ratio (`free_SO2 / total_SO2`).

## Objectives & Problem Formulation

### 1. Regression
**Goal**: Predict the exact quality score (0–10).

**Evaluation Metrics**:
- ✔ Mean Squared Error (MSE)
- ✔ R² Score (Explained variance)

### 2. Classification:
**Binary**: "Good" (≥7) vs. "Bad" (<7)  
**Multi-class**: Low (3–4), Medium (5–6), High (7–9)

**Evaluation Metrics**:
- ✔ Accuracy, Precision, Recall, F1-Score
- ✔ Confusion Matrix

**Details**: The goal is to build a machine learning model that predicts wine quality based on its chemical attributes. This can help wineries improve production and quality control.

## Suitable Machine Learning Algorithms

- **Logistic Regression**: Simple and interpretable.
- **Decision Tree Classifier**: Captures non-linear relationships.
- **Random Forest Classifier**: More accurate and robust against overfitting.
- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
- **Gradient Boosting (XGBoost, LightGBM, etc.)**: Advanced boosting techniques for higher accuracy.

### Discussion on the Algorithms
- **Decision Trees**: Useful for capturing interactions between variables but may overfit.
- **Random Forest**: Combats overfitting by combining multiple trees.
- **SVM**: Works well for complex decision boundaries.
- **Gradient Boosting**: Provides high accuracy and performs well on tabular data like this.

## Why Are These Algorithms Suitable?

The Wine Quality dataset consists of **numerical features** that describe the chemical composition of wines. The selected algorithms are capable of handling these features efficiently and capturing the intricate patterns within the data.

### 1. The Dataset Has Continuous Numerical Features
- The dataset contains 11 **numerical input features** (e.g., fixed acidity, volatile acidity, alcohol content, pH, sulphates) and the target variable (**quality**) is an integer score.
- **Tree-based models** (Decision Trees, Random Forest, Gradient Boosting) handle numerical features naturally.
- **Linear models** (Logistic Regression, SVM) work best with **scaled data**, but can still be used effectively with proper preprocessing.

### 2. Random Forest and Gradient Boosting Handle Feature Importance Well
- Wine quality is influenced by multiple factors like acidity, alcohol content, and sulphates. 
- **Random Forest** and **Gradient Boosting** models automatically evaluate feature importance, allowing wineries to focus on the most critical factors affecting wine quality.

### 3. SVM Works Well with Feature Scaling
- **SVM** works well when the data is **scaled**, especially when the decision boundary is complex.
- Kernelized SVM (e.g., RBF kernel) can capture **non-linear relationships** in the data, which is crucial for predicting wine quality.

## Research Papers on the Same Dataset

1. **Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009)**. “Modeling Wine Preferences by Data Mining from Physicochemical Properties”  
   *Summary*: Used decision trees, neural networks, and SVM to predict wine quality. Random Forest and SVM performed best.

2. **Rodrigues, L., & Francisco, M. (2016)**. “Wine Quality Prediction Using Machine Learning Techniques”  
   *Summary*: Applied Logistic Regression, Random Forest, and Gradient Boosting for wine quality classification. Ensemble methods outperformed single models.

3. **Singh, K., & Parthasarathy, R. (2020)**. “A Comparative Study on Wine Quality Prediction Using Machine Learning Algorithms”  
   *Summary*: Compared SVM, KNN, and Neural Networks for predicting wine quality. Found that Neural Networks performed best, followed by SVM.

These studies demonstrate the effectiveness of **Random Forest**, **Gradient Boosting**, and **Neural Networks** for predicting wine quality.

---

