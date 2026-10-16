
# Bank Customer Churn Analysis & Prediction

An end-to-end data analytics and machine learning project focused on understanding customer churn, identifying key retention factors, and predicting which bank customers are likely to leave.

## Overview

Customer churn is a major challenge for financial institutions, as losing customers can affect revenue, customer lifetime value, and long-term growth.

This project combines **exploratory data analysis, interactive visualizations, and machine learning** to transform customer data into actionable insights that can support data-driven retention strategies.

## Project Objectives

- Analyze customer demographics and banking behavior.
- Identify patterns and factors associated with customer churn.
- Explore relationships between customer attributes and churn outcomes.
- Develop a machine learning model to predict customer churn.
- Present analytical findings through an interactive Streamlit dashboard.

## Key Features

### 1. Customer Analytics
Explore customer characteristics, account activity, product usage, and churn patterns through interactive charts and filters.

### 2. Churn Prediction
Use a trained machine learning model to estimate the likelihood that a customer will leave the bank.

### 3. Model Insights
Review model performance and explore the factors associated with customer churn.

### 4. Interactive Dashboard
Navigate between analytical views, customer insights, and prediction tools in a unified interface.

## Dataset

The project uses the **Bank Customer Churn dataset**, which contains customer demographic and account-related information alongside a churn indicator.

Example features include:

- Credit score
- Geography
- Gender
- Age
- Tenure
- Account balance
- Number of bank products
- Credit card ownership
- Active membership status
- Estimated salary

**Target variable:** `Exited`

- `0` — Customer stayed
- `1` — Customer left

> Dataset source and licensing details should be added here once the original source has been verified.

## Machine Learning Workflow

1. Data inspection and preparation
2. Exploratory data analysis (EDA)
3. Feature preprocessing
4. Model training and evaluation
5. Customer churn prediction
6. Deployment through Streamlit

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Data visualization | Plotly |
| Machine learning | Scikit-learn |
| Model serialization | Joblib |
| Web application | Streamlit |
| Development environment | Google Colab |

## Project Structure

```text
bank-customer-churn/
├── app.py
├── requirements.txt
├── churn_model.joblib
├── Churn_Modelling.csv
└── README.md
```

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/rahaf1100/bank-customer-churn.git
cd bank-customer-churn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at the local URL provided by Streamlit.

## 🚀 Live Demo

Try the interactive Bank Customer Churn prediction app:

[Launch the App](https://bank-customer-churn-phesieuxpwbcmdxcktrfzu.streamlit.app/)
## Potential Business Value

The project demonstrates how customer data can be used to:

- Identify customer segments with elevated churn risk.
- Support proactive customer retention initiatives.
- Improve understanding of customer behavior.
- Help stakeholders make more informed, data-driven decisions.

These are potential applications of the analysis; actual business impact would require further validation.

## Future Improvements

- Compare additional classification algorithms.
- Improve model evaluation using recall, precision, F1-score, and ROC-AUC.
- Add explainability techniques such as SHAP.
- Introduce customer risk segmentation.
- Monitor model performance and data drift over time.

## Author

**Rahaf Alzahrani**

Computer Information Systems Graduate  
Interested in Data Analytics, Data Science, Machine Learning, and Data Visualization.

- GitHub: [@rahaf1100](https://github.com/rahaf1100)

---

*This project was developed for educational and portfolio purposes.*
