"""
This project demonstrates the use of supervised learning algorithms using the Titanic dataset. The following steps are performed:

1. Load the dataset.
2. Data Preprocessing:
   - Drop irrelevant columns.
   - Handle missing values.
   - Convert categorical variables to dummy variables.
3. Correlation Analysis:
   - Compute and visualize the correlation matrix.
4. Split the data into features and target variable.
5. Split the data into training and testing sets.
6. Feature Scaling:
   - Standardize the range of independent variables.
7. Model Building:
   - Train a Logistic Regression model.
8. Model Evaluation:
   - Evaluate the model using accuracy, confusion matrix, and classification report.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Data Preprocessing
# Drop irrelevant columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1)

# Handle missing values by dropping rows with missing data
df = df.dropna()

# Convert categorical variables to dummy variables
df['Sex'] = df['Sex'].apply(lambda row: 0 if row == 'male' else 1)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 3. Correlation Analysis
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 4. Split the data into features and target variable
x_features = df.drop('Survived', axis=1)
y_target = df['Survived']

# 5. Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2, random_state=42)

# 6. Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 7. Model Building
model = LogisticRegression()
model.fit(x_train, y_train)

# 8. Model Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
