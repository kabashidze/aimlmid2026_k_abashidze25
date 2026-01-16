# AI & Machine Learning for Cybersecurity – Midterm  
## Task 2: Email Spam Classification Using Logistic Regression

---

### 1. Dataset Upload (1 point)

The dataset provided at:

```
max.ge/aiml_midterm/859458492_csv
```

has been uploaded to this repository under the name:

```
k_abashidze25_859458492.csv
```

The dataset contains pre-extracted numerical features from email messages and a binary class label indicating whether an email is **spam (1)** or **legitimate (0)**. This dataset is used directly by the implemented Python console application without modification.

---

### 2. Model Training on 70% of the Data  
*(2 + 1 + 2 + 1 + 1 points)*

#### 2.1 Source Code (1 point)

All required functionality is implemented in a single Python file:

```
task2_email_classifier.py
```

The application executes sequentially and performs data loading, model training, validation, prediction, and visualization within one reproducible pipeline.

---

#### 2.2 Data Loading and Processing (2 points)

**Code location:** lines 90–128

```python
df = pd.read_csv("k_abashidze25_859458492.csv")

X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
```

**Academic explanation:**  
The dataset is loaded using the `pandas` library, which ensures efficient handling of tabular data. Feature selection is explicit to prevent data leakage and to ensure that only meaningful attributes are used for training. Stratified sampling is applied during the train-test split to preserve the original class distribution, which is critical for fair evaluation in binary classification problems.

---

#### 2.3 Logistic Regression Model (1 point)

**Code location:** lines 132–142

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    ))
])
model.fit(X_train, y_train)
```

**Academic explanation:**  
Logistic Regression is a linear probabilistic classifier that estimates the posterior probability of class membership using the sigmoid function. It is well suited for binary classification tasks such as spam detection due to its interpretability and robustness. Feature standardization is applied to normalize the scale of input variables, improving optimization stability and convergence when using gradient-based solvers such as LBFGS.

---

#### 2.4 Model Coefficients (1 point)

**Code location:** lines 147–163

```python
w_scaled = lr.coef_[0]
b_scaled = lr.intercept_[0]

w_original = w_scaled / scaler.scale_
b_original = b_scaled - np.sum(w_scaled * (scaler.mean_ / scaler.scale_))
```

**Academic explanation:**  
The learned coefficients represent the contribution of each feature to the log-odds of an email being classified as spam. Coefficients are transformed back to the original feature scale to enhance interpretability. Positive coefficients indicate features that increase spam probability, while negative coefficients indicate features associated with legitimate emails.

---

### 3. Model Validation  
*(1 + 2 points)*

#### 3.1 Confusion Matrix and Accuracy (1 point)

**Code location:** lines 168–174

```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```

**Academic explanation:**  
The confusion matrix provides a detailed breakdown of classification outcomes, including true positives, true negatives, false positives, and false negatives. Accuracy is used as a global performance metric representing the proportion of correctly classified samples. Together, these metrics allow both quantitative and qualitative assessment of model performance.

---

### 4. Email Text Classification Capability (3 points)

**Feature extraction code location:** lines 40–86

```python
tokens = TOKEN_RE.findall(text)
links = len(LINK_RE.findall(text))
capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
```

**Prediction code location:** lines 185–194

```python
features = extract_features_from_email(text)
prediction = model.predict([features])[0]
```

**Academic explanation:**  
This functionality extends the trained model to real-world usage by enabling classification of raw email text. The same feature space used during training is reconstructed through deterministic parsing rules, ensuring consistency between training and inference. This approach reflects standard practices in applied machine learning systems.

---

### 5. Manually Composed Spam Email (1 point)

**Email Text**
```
CONGRATULATIONS!!! YOU ARE A WINNER!
You have been selected to receive a FREE CASH BONUS.
Click the link below to claim your prize now:
https://free-prize-now.example
LIMITED TIME OFFER!!!
```

**Academic explanation:**  
This email was intentionally designed to activate multiple spam-related features, including excessive capitalization, urgency, spam keywords, and the presence of hyperlinks. These characteristics align with the learned decision boundary of the model, resulting in a spam classification.

---

### 6. Manually Composed Legitimate Email (1 point)

**Email Text**
```
Hi team,

Thanks for the meeting today. Please review the attached document
and send your feedback by Friday.

Best regards,
Project Manager
```

**Academic explanation:**  
This email demonstrates characteristics typical of legitimate communication, such as neutral language, absence of spam keywords, and lack of hyperlinks. Consequently, the extracted feature values correspond to a low spam probability.

---

### 7. Visualizations (4 points)

The application generates multiple visualizations using `matplotlib` to provide insights into the dataset and model behavior.

---

#### A. Class Distribution Study

**Code location:** lines 200–213

```python
plt.bar(labels, values)
plt.title("Class Distribution (Spam vs Legitimate)")
plt.xlabel("Class")
plt.ylabel("Number of emails")
```

**Academic explanation:**  
This visualization illustrates the balance between spam and legitimate emails in the dataset. Understanding class distribution is essential for evaluating potential bias and interpreting performance metrics such as accuracy.

---

#### B. Confusion Matrix Heatmap

**Code location:** lines 215–233

```python
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
```

**Academic explanation:**  
The heatmap provides an intuitive visual summary of classification performance, highlighting misclassification patterns and enabling deeper error analysis beyond scalar metrics.

---

#### C. Feature Importance (Logistic Regression Coefficients)

**Code location:** lines 235–251

```python
plt.bar(FEATURE_COLS, w_original)
plt.title("Logistic Regression Coefficients")
```

**Academic explanation:**  
This visualization emphasizes model interpretability by showing the relative importance of each feature. Such analysis is particularly valuable in cybersecurity applications, where explainability is often as important as predictive accuracy.

---

## How to Run

```bash
python task2_email_classifier.py
```

Generated output files are stored in:

```
task2_outputs/
```

---

**Course:** AI & Machine Learning for Cybersecurity  
**Assignment:** Midterm – Task 2  
**Student:** K. Abashidze












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
