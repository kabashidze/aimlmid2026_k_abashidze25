
# AI & Machine Learning for Cybersecurity – Midterm  
## Task 2: Email Spam Classification using Logistic Regression

---

## 1. Task Overview

The goal of this task is to develop a **Python console application** that classifies emails into **Spam** and **Legitimate** classes using **Logistic Regression**.

The classification is based on numerical features extracted from email content, such as the number of words, links, capitalized words, and spam-related keywords.  
The solution follows all requirements defined in the assignment description and demonstrates the full machine learning pipeline: data loading, training, validation, prediction, and visualization.

---

## 2. Dataset

The dataset was provided at:

```
max.ge/aiml_midterm/859458492_csv
```

The dataset is uploaded to this repository under the name:

```
k_abashidze25_859458492.csv
```

### Dataset Format

The CSV file contains the following columns:

| Column Name        | Description                                      |
|--------------------|--------------------------------------------------|
| words              | Total number of words in the email               |
| links              | Number of hyperlinks in the email                |
| capital_words      | Number of fully capitalized words                |
| spam_word_count    | Count of spam-related keywords                   |
| is_spam            | Target label (1 = Spam, 0 = Legitimate)          |

Example row:
```
142,5,4,10,1
```

---

## 3. Source Code

The full implementation is provided in a **single Python file**:

```
task2_email_classifier.py
```

This file contains:
- Data loading and preprocessing
- Model training and evaluation
- Feature extraction from raw email text
- Manual email classification examples
- Visualization generation

---

## 4. Data Loading and Processing

The dataset is loaded using **pandas**.  
Numerical validation is applied to ensure that all feature columns and labels are valid.

The dataset is split into:
- **70% training data**
- **30% testing data**

```python
df = pd.read_csv("k_abashidze25_859458492.csv")
X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

---

## 5. Logistic Regression Model

A **Logistic Regression** classifier from `scikit-learn` is used.

### Model Configuration
- Feature scaling: `StandardScaler`
- Solver: `LBFGS`
- Maximum iterations: `1000`

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="lbfgs", max_iter=1000))
])
model.fit(X_train, y_train)
```

---

## 6. Model Coefficients

After training, the model coefficients are transformed back to the **original feature scale**.

Each coefficient represents how strongly a feature influences the probability of an email being classified as spam.

---

## 7. Model Validation

The trained model is evaluated on the **test dataset (30%)**.

### Metrics Used
- **Confusion Matrix**
- **Accuracy**

```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```

---

## 8. Email Text Classification

The application can classify **raw email text** by extracting the same features used in the dataset and evaluating them using the trained model.

---

## 9. Manually Composed Emails

### 9.1 Spam Email Example

```
CONGRATULATIONS!!! YOU ARE A WINNER!
You have been selected to receive a FREE CASH BONUS.
Click the link below to claim your prize now:
https://free-prize-now.example
LIMITED TIME OFFER!!!
```

**Explanation:**  
This email contains multiple spam keywords, capitalized words, urgency, and a hyperlink, which are typical indicators of spam.

---

### 9.2 Legitimate Email Example

```
Hi team,

Thanks for the meeting today. Please review the attached document
and send your feedback by Friday.

Best regards,
Project Manager
```

**Explanation:**  
This email uses professional language, contains no spam keywords, and has no links, which makes it legitimate.

---

## 10. Visualizations

The application generates the following visualizations using `matplotlib`:

### A. Class Distribution
Shows the ratio of spam vs legitimate emails and helps detect dataset imbalance.

### B. Confusion Matrix Heatmap
Visualizes model prediction performance for each class.

### C. Feature Importance
Displays logistic regression coefficients to show which features influence spam detection the most.

---

## 11. How to Run

1. Place the dataset and script in the same directory:
```
k_abashidze25_859458492.csv
task2_email_classifier.py
```

2. Run:
```bash
python task2_email_classifier.py
```

3. Output files are saved in:
```
task2_outputs/
```

---

## 12. Conclusion

This project demonstrates a complete implementation of email spam detection using Logistic Regression.  
The solution covers data preprocessing, model training, evaluation, interpretability, and real-world email classification.

---

**Course:** AI & Machine Learning for Cybersecurity  
**Assignment:** Midterm – Task 2  
**Student:** K. Abashidze
