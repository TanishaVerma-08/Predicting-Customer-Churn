# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

#Loading the Dataset
data_path = '/content/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(data_path)

#Data Exploration
print("Dataset Overview:")
print(data.head())
print("Missing Values:")
print(data.isnull().sum())
print("Data Info:")
data.info()

#Data Cleaning & Preprocessing
data.drop(columns=['customerID'], inplace=True)

#Convert 'TotalCharges' to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

#Fill missing values
data.fillna(method='ffill', inplace=True)

#Encode categorical variables
encoder = LabelEncoder()
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

#Normalize numerical features
scaler = StandardScaler()
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

#Split the Data
X = data.drop(columns=['Churn'])  # Features
y = data['Churn']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Development
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#Predictions and Evaluation
y_pred = model.predict(X_test)

print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)

#Visualize feature importance
plt.figure(figsize=(8, 5))
feature_importance.plot(kind='bar', color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

#Suggestions based on findings
print("Retention Stategies:")
print("- Focus on customers with high monthly charges to reduce churn.")
print("- Improve services for users with lower tenure.")
print("- Offer better support for customers with frequent complaints.")