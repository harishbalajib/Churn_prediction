# **XGChurn - Customer Churn Prediction**
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## **Overview**
XGChurn is a robust machine learning project aimed at predicting customer churn in the banking sector. Using the **XGBoost** algorithm and explainability tools like **SHAP**, the model achieves **87% accuracy** and **85% ROC-AUC**. An interactive **Streamlit dashboard** enables visualization of predictions and insights, helping stakeholders understand churn drivers.

---

## **Key Features**
- Predicts churn with **87% accuracy** and **85% ROC-AUC**.
- Implements advanced feature engineering (e.g., balance-to-salary ratio).
- Visualizes model explainability using **SHAP** values.
- Offers a user-friendly **Streamlit dashboard** for predictions and insights.

---

## **Mathematical Foundations**

### **Churn Prediction**
The model predicts a binary outcome ($ Y $) where:
$$
Y = \begin{cases} 
1, & \text{if customer churns} \\
0, & \text{otherwise}
\end{cases}
$$

Using **XGBoost**, the objective is to minimize the log-loss function:
$$
\text{Log-Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$
Where:
- $ y_i $ = True label
- $ \hat{y}_i $ = Predicted probability

### **Feature Importance**
Feature importance is derived from **SHAP** values, quantifying each feature's contribution to predictions:
$$
\text{SHAP Value} = \phi_i = f(x) - \mathbb{E}[f(x')]
$$
Where:
- $ f(x) $ = Model prediction
- $ \mathbb{E}[f(x')] $ = Average prediction over all features.

---

## **Tech Stack**
- **Programming Language:** Python
- **Machine Learning:** XGBoost, scikit-learn
- **Dashboard:** Streamlit
- **Libraries:** pandas, numpy, matplotlib, SHAP
- **Data Format:** CSV (Bank customer data)

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/XGChurn.git

