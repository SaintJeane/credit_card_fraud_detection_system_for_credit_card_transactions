
---

# 💳🚨 Credit Card Fraud Detection System Project

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of all transactions, yet carry significant financial and reputational risk. This project presents an **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions, with a strong emphasis on **model performance, threshold optimization, and interpretability**.

The goal is not only to build an accurate fraud detection model, but also to ensure that predictions are **transparent, explainable, and aligned with real-world decision-making**.

---

## 🎯 Objectives

* Detect fraudulent transactions with high recall while maintaining acceptable precision
* Address severe class imbalance using appropriate modeling strategies
* Select decision thresholds based on business-aligned trade-offs
* Provide global and local explanations of model predictions using SHAP
* Produce a reproducible, portfolio-grade machine learning workflow

---

## 📂 Dataset

* **Source:** [Kaggle Dataset Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description:** Anonymized transaction features derived from PCA transformations, along with transaction amount and class label
* **Target Variable:**

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction
* **Key Challenge:** Extreme class imbalance (~0.17% fraud cases)

---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing

* Train–test split with stratification
* Feature scaling where appropriate
* Careful handling of class imbalance

### 2️⃣ Modeling Approaches

* **Random Forest Classifier**

  * Baseline model
  * Explored class weighting and threshold adjustment

* **LightGBM Classifier (Final Model)**

  * Gradient-boosted decision trees
  * Optimized for imbalanced classification
  * Superior recall–precision trade-off

### 3️⃣ Model Evaluation

Models were evaluated using metrics suitable for imbalanced data:

* Precision
* Recall
* F1-score
* ROC-AUC
* Precision–Recall curves

Decision thresholds were selected based on:

* Precision-constrained recall
* F1-score maximization
* Business cost considerations

---

## 🧠 Model Interpretability (SHAP)

To ensure transparency and trust, SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

### Global Interpretability

* Feature importance analysis to identify key fraud indicators
* SHAP summary plots to understand direction and magnitude of feature effects

### Local Interpretability

* Transaction-level explanations for individual fraud cases
* Clear visualization of features pushing predictions toward fraud or legitimacy

All explanations focus on the **fraud class (class 1)** and are aligned with the final decision threshold.

---

## 🏆 Results Summary

* LightGBM achieved strong recall on fraudulent transactions while maintaining reasonable precision
* SHAP analysis confirmed that the model relies on a consistent and interpretable set of fraud indicators
* The final system balances performance, explainability, and reproducibility

---

## 📁 Project Structure

```text
├── dataset/
│   └── archive.zip
├── notebooks/
│   └── Credit_Card_Fraud_Detection_System.ipynb
├── models/
│   ├── lgbm/
|   |     ├── lgbm_baseline.pkl
|   |     ├── lgbm_tuned.pkl
|   |     └── lgbm_metadata.json
│   └── random_forest/
|          ├── rf_metadata.json
|          └── rf_model.pkl 
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/SaintJeane/credit_card_fraud_detection_system_for_credit_card_transactions.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebooks/Credit_Card_Fraud_Detection_System.ipynb
```

---

## 📦 Tools & Technologies

* Python
* scikit-learn
* LightGBM
* SHAP
* NumPy
* pandas
* matplotlib / seaborn
* joblib

---

## 📈 Key Takeaways

* Imbalanced classification requires **careful metric selection**, not accuracy alone
* Decision thresholds are as important as the model itself
* Interpretability is essential for deploying fraud detection systems in practice
* Combining performance with explainability leads to more trustworthy models

---

> *This project was developed as a portfolio-grade demonstration of applied machine learning for fraud detection, with a focus on real-world constraints and interpretability.*

---
