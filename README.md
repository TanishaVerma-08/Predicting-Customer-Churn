# Telco Customer Churn Prediction

## Project Overview  
This project analyzes customer churn in a telecom company using machine learning techniques. The dataset contains customer details, subscription plans, and service usage. A **Random Forest Classifier** is implemented to predict customer churn based on various factors.  

## Dataset  
The dataset used is **WA_Fn-UseC_-Telco-Customer-Churn.csv** and includes features such as:  
- Customer demographic details  
- Subscription plan details  
- Monthly and total charges  
- Customer churn status (target variable)  

## Features and Preprocessing  
The following steps were performed on the dataset:  
- **Data Cleaning:** Removed the `customerID` column and handled missing values in `TotalCharges`  
- **Data Transformation:** Converted categorical variables using **Label Encoding**  
- **Feature Scaling:** Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)  
- **Data Splitting:** Divided the dataset into training (80%) and testing (20%) sets  

## Model Development  
A **Random Forest Classifier** was trained to predict customer churn. The model was evaluated using the following metrics:  
- **Accuracy**: Measures the overall correctness of predictions  
- **Precision**: Evaluates the proportion of correctly identified churned customers  
- **Recall**: Measures the ability to identify all churned customers  
- **F1 Score**: Balances precision and recall for a better assessment  

## Model Evaluation  
The model’s performance was assessed using the **classification report** and the following metrics:  
- Accuracy Score  
- Precision Score  
- Recall Score  
- F1 Score  

## Feature Importance  
The **Random Forest Classifier** provides feature importance scores, helping to identify key factors influencing customer churn. A bar plot was generated to visualize feature importance.  

## Key Findings and Recommendations  
Based on the analysis, the following strategies can be implemented to reduce churn:  
- Focus on **customers with high monthly charges**, as they are more likely to churn  
- Improve retention strategies for **new customers with lower tenure**  
- Provide better support and services for customers with **frequent complaints**  

## Dependencies  
The project requires the following Python libraries:  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

Install dependencies using:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage  
1. Load the dataset and preprocess the data  
2. Train the **Random Forest Classifier**  
3. Evaluate model performance using classification metrics  
4. Visualize feature importance to gain insights  
5. Implement retention strategies based on key findings  

## Conclusion  
This project provides a machine learning-based approach to predict customer churn and identify factors influencing customer retention. Future work can include testing additional models and feature engineering for improved accuracy.
