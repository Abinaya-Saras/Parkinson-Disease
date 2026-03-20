# 🧠 Parkinson's Disease Detection

A machine learning project to detect Parkinson's disease from biomedical voice measurements using multiple classification algorithms.

---

## 📁 Dataset

- **Source:** `parkinsons.data`
- **Rows:** 195 | **Features:** 24
- **Target:** `status` (1 = Parkinson's, 0 = Healthy)
- **Missing Values:** None

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🔍 Project Workflow

1. Load and explore the dataset
2. Visualize feature distributions and target balance
3. Drop irrelevant columns (`name`)
4. Split data into train/test sets (80/20)
5. Train and evaluate multiple ML models
6. Optimize best model using StandardScaler + GridSearchCV

---

## 📊 Model Comparison

| Model               | Test Accuracy | Kappa Score | Wrong Predictions |
|---------------------|:-------------:|:-----------:|:-----------------:|
| Logistic Regression | 87.18%        | —           | —                 |
| Random Forest       | 87.18%        | 0.5877      | 5 / 39            |
| Decision Tree       | 100%*         | 1.0*        | 0 / 39            |
| Naive Bayes         | 69.23%        | 0.3938      | 12 / 39           |
| KNN                 | 84.62%        | 0.4800      | 6 / 39            |
| SVM (Linear)        | 89.74%        | 0.6533      | 4 / 39            |
| **RF (Optimized)**  | **92.31%**    | —           | —                 |

> ⚠️ *Decision Tree shows 100% due to data leakage (trained on full dataset).

---

## ✅ Best Model

**Optimized Random Forest** with `StandardScaler` + `GridSearchCV`
- **Accuracy:** 92.31%
- **Recall (Parkinson's):** 0.97
- **Precision:** 0.93

---

## 📂 Project Structure
```
├── parkinsons.data
├── Parkinson.ipynb
└── README.md
```

---

## 🚀 How to Run
```bash
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection
pip install -r requirements.txt
jupyter notebook Parkinson.ipynb
```

---

## 📌 Key Findings

- The dataset is **imbalanced** (75% Parkinson's, 25% Healthy)
- **SVM** and **Optimized Random Forest** are the best-performing models
- Feature scaling significantly improves model performance
