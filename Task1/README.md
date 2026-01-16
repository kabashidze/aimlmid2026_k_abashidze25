
# AI & Machine Learning for Cybersecurity – Midterm  
## Task 1: Finding Logistic Regression Model Coefficients

---

## 1. Problem Description

The objective of this task is to build a **multiclass logistic regression** model that classifies two-dimensional data points into three classes based on predefined geometric and color-based rules.

### Class Definitions
- **Class 1** – All points located *above* the given reference line  
- **Class 2** – All **red** points  
- **Class 3** – All **green** points  

The task requires identifying model coefficients and visualizing the learned decision boundaries.

---

## 2. Dataset Preparation

The dataset was extracted from the interactive graph provided in the assignment.  
For each data point, the following attributes were collected:

- `x` – horizontal coordinate  
- `y` – vertical coordinate  
- `class` – target label (1, 2, or 3)

The prepared dataset is stored in CSV format:

```
task1_points.csv
```

This ensures full reproducibility and separation of data from code.

---

## 3. Model Selection

A **multiclass logistic regression** model with **softmax** output was selected.

For each class *k*, the model learns a linear decision function:

```
z_k = w_kx · x + w_ky · y + b_k
```

The softmax function converts these values into class probabilities, and the class with the highest probability is selected.

Feature standardization is applied before training to improve convergence and numerical stability.

---

## 4. Model Training

The model was trained using **scikit-learn** with the following configuration:

- Multiclass strategy: `multinomial`
- Solver: `LBFGS`
- Maximum iterations: `500`
- Feature scaling: `StandardScaler`

---

## 5. Learned Model Coefficients

After training, the coefficients were transformed back to the **original feature scale**.

### Coefficients per class

**Class 1**
```
w_x = -0.229474
w_y =  0.707050
b   =  1.138323
```

**Class 2**
```
w_x = -0.010340
w_y = -0.410475
b   =  1.472933
```

**Class 3**
```
w_x =  0.239814
w_y = -0.296576
b   = -2.611255
```

These coefficients define linear decision surfaces separating the three classes.

---

## 6. Visualization

The figure below illustrates:
- Original data points
- Decision regions learned by the logistic regression model

Each colored background region represents the predicted class.

![Decision Regions]([Task1/Finding the logistic regression model coefficients - Results.png](https://github.com/kabashidze/aimlmid2026_k_abashidze25/blob/c414297526655ee32f0534460b8c3f1505eeafb8/Task1/Finding%20the%20logistic%20regression%20model%20coefficients%20-%20Results.png))

---

## 7. Conclusion

The multiclass logistic regression model successfully separates the dataset into three distinct classes using linear decision boundaries.  
The learned decision regions closely align with the manually defined class rules, confirming the correctness of:

- data extraction
- class labeling
- model selection
- training process

This demonstrates that logistic regression is an effective baseline model for structured classification problems in cybersecurity contexts.

---

## 8. How to Run

```bash
pip install numpy pandas scikit-learn matplotlib
python task1_logistic_regression.py
```

---

**Course:** AI & Machine Learning for Cybersecurity  
**Assignment:** Midterm – Task 1
