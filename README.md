# Parkinson's Disease Detection

A machine learning project to detect Parkinson's disease from biomedical voice measurements.

---

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

## 2. Load Dataset

```python
df = pd.read_csv('/content/drive/MyDrive/parkinsons.data')
# Reading dataset
```

---

## 3. Exploratory Data Analysis (EDA)

### View Dataset

```python
df
```

> **195 rows × 24 columns**

---

### Shape

```python
df.shape
# Output: (195, 24)
```

---

### Check for Missing Values

```python
# There are no null values in the dataframe.
df.isnull().sum()
```

> All columns return `0` — no missing values.

---

### Dataset Info

```python
# Missing values / data types
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 195 entries, 0 to 194
Data columns (total 24 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   name              195 non-null    object
 1   MDVP:Fo(Hz)       195 non-null    float64
 ...
 17  status            195 non-null    int64
 ...
dtypes: float64(22), int64(1), object(1)
memory usage: 36.7+ KB
```

---

### Statistical Summary

```python
# describe() shows percentile, mean, std of numerical values
df.describe()
```

---

### Column Names

```python
# column names
df.columns
```

---

### Target Column

```python
# target column
df['status']
```

> `status`: `1` = Parkinson's, `0` = Healthy

---

## 4. Visualizations

### Target Distribution

```python
plt.figure(figsize=(10, 6))
df.status.hist()
plt.xlabel('status')
plt.ylabel('Frequencies')
plt.plot()
```

---

### NHR by Status (Noise-to-Harmonics Ratio)

```python
# Ratio of noise to tonal components in the voice (NHR)
plt.figure(figsize=(10, 6))
sns.barplot(x="status", y="NHR", data=df)
```

---

### RPDE by Status

```python
# RPDE
plt.figure(figsize=(10, 6))
sns.barplot(x="status", y="RPDE", data=df)
```

---

### Distribution Plots

```python
rows = 3
cols = 7
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 4))
col = df.columns
index = 1
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax=ax[i][j])
        index = index + 1

plt.tight_layout()
```

> ⚠️ Note: `distplot` is deprecated in newer versions of Seaborn. Use `histplot` or `displot` instead.

---

## 5. Feature Engineering

### Drop Irrelevant Column & Define Features/Target

```python
df.drop(['name'], axis=1, inplace=True)

X = df.drop(labels=['status'], axis=1)
Y = df['status']
X.head()
```

---

## 6. Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=40
)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# Output: (156, 22) (39, 22) (156,) (39,)
```

---

## 7. Machine Learning Models

Models evaluated:
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Naive Bayes
5. K-Nearest Neighbours (KNN)
6. Support Vector Machine (SVM)

---

### 7.1 Logistic Regression

```python
log_reg = LogisticRegression().fit(X_train, Y_train)

train_preds = log_reg.predict(X_train)
print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds))

test_preds = log_reg.predict(X_test)
print("Model accuracy on test is: ", accuracy_score(Y_test, test_preds))

print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds))
print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds))
```

```
Model accuracy on train is:  0.8717948717948718
Model accuracy on test is:   0.8717948717948718
```

---

### 7.2 Random Forest

```python
RF = RandomForestClassifier().fit(X_train, Y_train)

train_preds2 = RF.predict(X_train)
print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds2))

test_preds2 = RF.predict(X_test)
print("Model accuracy on test is: ", accuracy_score(Y_test, test_preds2))

print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds2))
print("confusion_matrix test is: ", confusion_matrix(Y_test, test_preds2))
```

```
Model accuracy on train is:  1.0
Model accuracy on test is:   0.8717948717948718
```

#### Wrong Predictions

```python
print((Y_test != test_preds2).sum(), '/', ((Y_test == test_preds2).sum() + (Y_test != test_preds2).sum()))
# Output: 5 / 39
```

#### Kappa Score

```python
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test, test_preds2))
# Output: KappaScore is: 0.587737843551797
```

---

### 7.3 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier().fit(X, Y)

train_preds3 = DT.predict(X_train)
test_preds3  = DT.predict(X_test)

print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds3))
print("Model accuracy on test is: ",  accuracy_score(Y_test, test_preds3))
```

```
Model accuracy on train is:  1.0
Model accuracy on test is:   1.0
```

#### Confusion Matrix & Wrong Predictions

```python
print("confusion_matrix train is: ", confusion_matrix(Y_train, train_preds3))
print("confusion_matrix test is: ",  confusion_matrix(Y_test, test_preds3))
print((Y_test != test_preds3).sum(), '/', ...)
# Output: 0 / 39
```

#### Kappa Score

```python
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test, test_preds3))
# Output: KappaScore is: 1.0
```

---

### 7.4 Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train, Y_train)

train_preds4 = NB.predict(X_train)
test_preds4  = NB.predict(X_test)

print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds4))
print("Model accuracy on test is: ",  accuracy_score(Y_test, test_preds4))
```

```
Model accuracy on train is:  0.7307692307692307
Model accuracy on test is:   0.6923076923076923
```

#### Kappa Score

```python
print('KappaScore is: ', metrics.cohen_kappa_score(Y_test, test_preds4))
# Output: KappaScore is: 0.3937823834196892
```

---

### 7.5 K-Nearest Neighbours (KNN)

```python
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier().fit(X_train, Y_train)

train_preds5 = KNN.predict(X_train)
test_preds5  = KNN.predict(X_test)

print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds5))
print("Model accuracy on test is: ",  accuracy_score(Y_test, test_preds5))
```

```
Model accuracy on train is:  0.9102564102564102
Model accuracy on test is:   0.8461538461538461
```

#### Wrong Predictions & Kappa Score

```python
print((Y_test != test_preds5).sum(), '/', ...)
# Output: 6 / 39

print('KappaScore is: ', metrics.cohen_kappa_score(Y_test, test_preds5))
# Output: KappaScore is: 0.48
```

---

### 7.6 Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

SVM = SVC(kernel='linear')
SVM.fit(X_train, Y_train)

train_preds6 = SVM.predict(X_train)
test_preds6  = SVM.predict(X_test)

print("Model accuracy on train is: ", accuracy_score(Y_train, train_preds6))
print("Model accuracy on test is: ",  accuracy_score(Y_test, test_preds6))
```

```
Model accuracy on train is:  0.8782051282051282
Model accuracy on test is:   0.8974358974358975
```

#### Recall, Wrong Predictions & Kappa Score

```python
print("recall", metrics.recall_score(Y_test, test_preds6))
# recall 0.967741935483871

print((Y_test != test_preds6).sum(), '/', ...)
# Output: 4 / 39

print('KappaScore is: ', metrics.cohen_kappa_score(Y_test, test_preds6))
# Output: KappaScore is: 0.6533333333333333
```

---

## 8. Optimized Random Forest (with Preprocessing + Grid Search)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('/content/drive/MyDrive/parkinsons.data')
data.drop(columns=['name'], inplace=True)

X = data.drop(columns=['status'])
y = data['status']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Optimized Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
```

```
Optimized Accuracy: 0.9231

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.80      0.84        10
           1       0.93      0.97      0.95        29

    accuracy                           0.92        39
   macro avg       0.91      0.88      0.90        39
weighted avg       0.92      0.92      0.92        39
```

### Confusion Matrix

```python
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Parkinson'],
            yticklabels=['Healthy', 'Parkinson'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Confusion Matrix')
plt.show()
```

---

## 9. Model Comparison Summary

| Model               | Test Accuracy | Kappa Score | Wrong Predictions |
|---------------------|---------------|-------------|-------------------|
| Logistic Regression | 87.18%        | —           | —                 |
| Random Forest       | 87.18%        | 0.5877      | 5 / 39            |
| Decision Tree       | **100%**      | **1.0**     | 0 / 39            |
| Naive Bayes         | 69.23%        | 0.3938      | 12 / 39           |
| KNN                 | 84.62%        | 0.48        | 6 / 39            |
| SVM (Linear)        | 89.74%        | 0.6533      | 4 / 39            |
| RF (Optimized)      | **92.31%**    | —           | —                 |

> ⚠️ **Note:** Decision Tree achieving 100% test accuracy is likely due to overfitting (it was trained on the full dataset `X, Y` instead of only `X_train, Y_train`).
