# # Bank Customer Churn Prediction with eXplainable AI (XAI)
This project aims to predict customer churn for ABC Multistate Bank using Explainable AI

![Churn Prediction](https://dezyre.gumlet.io/images/blog/churn-models/Customer_Churn_Prediction_Models_in_Machine_Learning.png?w=330&dpr=2.6)

## Overview

This project is an ML-based interactive web application built with Streamlit that predicts whether a bank customer is likely to churn (leave) based on various customer attributes. The prediction model used in this application is powered by a Random Forest Classifier, XGBoost Classifier, and Decision Tree classifier. It is complemented by eXplainable Artificial Intelligence (XAI) using SHAP (SHapley Additive explanations) to & Lime (Local Interpretable Model-agnostic Explanations) to provide insights into the model's predictions, making it transparent and interpretable.The dataset used for this project contains various features that may influence customer churn, such as credit score, age, tenure, balance, product usage, credit card status, active membership, estimated salary, and more. The target variable, "churn," indicates whether a customer has left the bank during a specific period (1 if churned, 0 if not).

## Getting Started
To run the project, follow these steps:

1) Clone the repository:
```
https://github.com/Omkar-Rajkumar-Khade/Customer_churn_prediction_using_ann_using_XAI.git
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

### Technologies Used: 
* Python
* Streamlit
* scikit-learn ( machine learning)
* pickle
* SHAP (SHapley Additive exPlanations)
* Lime (Local Interpretable Model-agnostic Explanations)

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



Sure, here's a README.md file in Markdown format for your code:

Bank Customer Churn Prediction
This project focuses on predicting customer churn in a bank using machine learning techniques. Customer churn refers to the phenomenon where customers stop doing business with a company, in this case, leaving the bank. We will explore and visualize the data, preprocess it, train machine learning models, and use two explainability techniques, SHAP and LIME, to understand model predictions.

Getting Started
Prerequisites
Make sure you have the following libraries installed:

numpy
pandas
matplotlib
seaborn
sklearn
xgboost
shap
lime
You can install these libraries using pip:

Copy code
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime
Data
The dataset used for this project is stored in a CSV file named Bank Customer Churn Prediction.csv. The dataset contains information about bank customers, including features like credit score, age, tenure, balance, and more. The goal is to predict whether a customer will churn (leave the bank) or not.

Running the Code
Clone this repository:
bash
Copy code
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Place the dataset CSV file (Bank Customer Churn Prediction.csv) in the same directory as the code.

Run the code by executing the Python script.


### Code Structure
1) Data Loading and Preprocessing: The code begins by loading the dataset and performing initial data preprocessing, including dropping unnecessary columns like 'country' and 'gender'.

2) Data Visualization: The code visualizes the correlation matrix and the distribution of numeric features in the dataset using matplotlib and seaborn.

3) Data Split and Scaling: The dataset is split into training and testing sets, and feature scaling is applied using StandardScaler.

4) Model Training: Three different classification models are trained: Random Forest, Decision Tree, and XGBoost.

4) Model Evaluation: The code evaluates each model's accuracy on the test data and generates a confusion matrix.

5) Explainability: The code uses SHAP and LIME to explain model predictions for a specific data point.

5) Model Saving: The trained Random Forest model is saved using pickle.

6) Frontend: Interactive web application built with Streamlit

### Model Accuracy : 
The project employs different classification algorithms, and here are their accuracy scores:

* Random Forest Classifier: 0.8565 (86%)
* Decision Tree Classifier: 0.7875(79%)
* XGBoost Classifier: 0.8465 (85%)
These accuracy scores reflect the performance of the respective models in predicting customer churn.



