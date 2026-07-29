<div align="center">

# 👨‍💼 Employee Attrition Prediction using Machine Learning & Explainable AI

### Predict employee attrition before it happens — and understand *why* with SHAP Explainable AI.

<img src="https://img.shields.io/badge/Python-3.13-blue?logo=python"/>
<img src="https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit"/>
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn"/>
<img src="https://img.shields.io/badge/SHAP-ExplainableAI-success"/>
<img src="https://img.shields.io/badge/License-MIT-green"/>

</div>

---

# 📖 Overview

Employee attrition is one of the biggest challenges faced by organisations. Losing experienced employees increases recruitment costs, training expenses and reduces overall productivity.

This project uses **Machine Learning** to predict whether an employee is likely to leave the company based on HR information such as:

- Age
- Monthly Income
- Overtime
- Job Role
- Job Satisfaction
- Environment Satisfaction
- Years at Company
- Work-Life Balance
- and many more...

Unlike traditional prediction systems, this project also provides **Explainable AI (XAI)** using **SHAP**, allowing HR teams to understand **why** a prediction was made instead of receiving only a Yes/No answer.

---

# 🚀 Live Demo

> 🔗 **Coming Soon**

Deploy on Streamlit Cloud and replace this section with your app link.

---

# 📸 Application Preview

## Dashboard

<img src="images/dashboard.png"/>

## Prediction Result

<img src="images/result.png"/>

## SHAP Explanation

<img src="images/shap.png"/>

*(Replace the images with your screenshots.)*

---

# ✨ Features

✅ Predict employee attrition risk instantly

✅ Clean Streamlit user interface

✅ 31 employee input features

✅ Confidence score for every prediction

✅ SHAP Explainable AI

✅ Feature contribution chart

✅ Feature impact table

✅ Multiple machine learning models compared

---

# 🧠 Machine Learning Workflow

```text
IBM HR Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Label Encoding
       │
       ▼
Train/Test Split
       │
       ▼
Train Multiple Models
       │
       ▼
Model Evaluation
       │
       ▼
Best Model Selection
       │
       ▼
SHAP Explainability
       │
       ▼
Streamlit Deployment
```

---

# 📂 Dataset

**Dataset:** IBM HR Analytics Employee Attrition Dataset

**Records:** 1470 Employees

**Features:** 35

Target Variable

```
Attrition
├── Yes
└── No
```

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed before training:

- Removed constant columns
- Encoded categorical variables
- Checked missing values
- Converted target labels
- Feature engineering
- Train-test split (80:20)

---

# 🤖 Models Evaluated

| Model | Accuracy |
|---------|----------|
| 🌳 Random Forest | **88.1%** |
| ✅ Logistic Regression | **87.8%** |
| ⚡ Support Vector Machine | 86.7% |
| 📊 Naive Bayes | 84.0% |
| 👥 KNN | 81.3% |
| 🌱 Decision Tree | 78.6% |

---

# 🏆 Final Model

Although Random Forest achieved slightly higher accuracy, it suffered from **overfitting**.

| Metric | Logistic Regression |
|---------|--------------------|
| Train Accuracy | Balanced |
| Test Accuracy | 87.8% |
| Generalisation | Excellent |
| SHAP Compatibility | Excellent |

For these reasons, **Logistic Regression** was selected.

---

# 🔍 Explainable AI (SHAP)

Instead of simply predicting whether an employee will leave, the model explains **why**.

Example:

```
Prediction:
Employee Likely to Leave

Top Reasons:

⬆ Overtime

⬆ Low Job Satisfaction

⬆ Low Monthly Income

⬇ Years at Company

⬇ Stock Option Level
```

Red bars increase attrition risk.

Blue bars reduce attrition risk.

This makes the model transparent and easier for HR professionals to trust.

---

# 📁 Project Structure

```
Employee-Attrition/

│
├── app.py
├── requirements.txt
│
├── preprocessing.ipynb
├── train_test.ipynb
│
├── Dataset.csv
├── processed_table.csv
│
├── images/
│   ├── dashboard.png
│   ├── result.png
│   └── shap.png
│
└── README.md
```

---

# 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Employee-Attrition.git
```

Move into project directory

```bash
cd Employee-Attrition
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 💻 Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming |
| Streamlit | Web Application |
| Scikit-Learn | Machine Learning |
| SHAP | Explainable AI |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Matplotlib | Visualization |

---

# 📊 Model Limitations

- Dataset is imbalanced
- Attrition class represents only ~16% of employees
- Accuracy alone is not sufficient for evaluation
- Future work can improve recall using:
  - SMOTE
  - Class Weights
  - XGBoost
  - Hyperparameter Tuning

---

# 🔮 Future Improvements

- Deploy on AWS/Azure
- Add employee PDF reports
- Email prediction reports
- Interactive analytics dashboard
- Database integration
- User authentication
- Model retraining pipeline

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository, open issues or submit pull requests.

---

# 📜 License

This project is licensed under the MIT License.

---

<div align="center">

### ⭐ If you found this project useful, don't forget to star the repository!

Made with ❤️ using Python, Streamlit and Explainable AI

</div>