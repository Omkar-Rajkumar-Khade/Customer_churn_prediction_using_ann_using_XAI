# # Bank Customer Churn Prediction with eXplainable AI (XAI)
This project aims to predict customer churn for ABC Multistate Bank using Explainable AI

![Churn Prediction](https://dezyre.gumlet.io/images/blog/churn-models/Customer_Churn_Prediction_Models_in_Machine_Learning.png?w=330&dpr=2.6)

## Overview

This project is an ML-based interactive web application built with Streamlit that predicts whether a bank customer is likely to churn (leave) based on various customer attributes. The prediction model used in this application is powered by a Random Forest Classifier, XGBoost Classifier, and Decision Tree classifier. It is complemented by eXplainable Artificial Intelligence (XAI) using SHAP (SHapley Additive explanations) to & Lime (Local Interpretable Model-agnostic Explanations) to provide insights into the model's predictions, making it transparent and interpretable.The dataset used for this project contains various features that may influence customer churn, such as credit score, age, tenure, balance, product usage, credit card status, active membership, estimated salary, and more. The target variable, "churn," indicates whether a customer has left the bank during a specific period (1 if churned, 0 if not).

## Getting Started
To run the project, follow these steps:

1) Clone the repository:
```
git clone https://github.com/your-username/customer_churn_prediction_using_ann_using_XAI.git
```
2) Install the required libraries: 
```
pip install pandas numpy scikit-learn shap lime matplotlib streamlit
```
3) Open the Jupyter Notebook CCP_using_SML.ipynb using Jupyter Notebook or any compatible environment.

4) Open the terminal or command prompt and navigate to the repository directory.

5) Run the Streamlit app: `streamlit run streamlit_app.py`

6) The app will open in your default web browser, allowing you to input feature values and see churn predictions.

Note: Please update the file paths if necessary and ensure that the required libraries are installed.

### Usage
*app.py*
1) Use the sliders and input fields in the web app to input customer data such as credit score, age, tenure, balance, products, credit card status, active membership, and estimated salary.

2) Click the "Predict" button to obtain the model's prediction for customer churn.

3) The application will display whether the customer is likely to churn or not, along with the churn probability.

*CCP_using_SML.ipynb*
1) Explore the SHAP summary plot to understand the impact of each feature on the model's prediction, making it transparent and interpretable.



### Dataset
The dataset used for this project contains the following columns:
Dataset Download Link: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

`customer_id`: Unused variable.

`credit_score`: Used as an input.

`country`: Unused variable.

`gender`: Unused variable.

`age`: Used as an input.

`tenure`: Used as an input.

`balance`: Used as an input.

`products_number` : Used as an input.

`credit_card`: Used as an input.

`active_member`: Used as an input.

`estimated_salary`: Used as an input.

`churn`: Target variable. 1 if the client has left the bank during some period or 0 if he/she has not.

## Repository Files
The repository contains the following files:

`dataset` folder contains the Bank Customer Churn Prediction.csv dataset used in the project.

`app.py` is the streamlit application file that defines the API endpoints and loads the saved model.

`models` is a folder that contains the serialized machine learning models that is used for prediction.

`CCP_using_SML.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.

`README.md`: Project documentation and instructions.

### Model Accuracy : 
The project employs different classification algorithms, and here are their accuracy scores:

* Random Forest Classifier: 0.8565 (86%)
* Decision Tree Classifier: 0.7875(79%)
* XGBoost Classifier: 0.8465 (85%)
These accuracy scores reflect the performance of the respective models in predicting customer churn.

### Technologies Used: 
* Python
* Streamlit
* scikit-learn
* SHAP (SHapley Additive exPlanations)
* Lime (Local Interpretable Model-agnostic Explanations)

