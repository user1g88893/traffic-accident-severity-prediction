#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

import joblib


# In[2]:


import pandas as pd
import os


# In[3]:


from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

file_path = askopenfilename(
    title="Select RTA Dataset CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

print("Selected File:", file_path)


# In[4]:


df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")

print(df.head())


# In[5]:


possible_targets = [
    'Accident_severity',
    'Accident Severity',
    'Severity',
    'severity'
]

target_column = None

for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

print("Target Column:", target_column)


# In[11]:


X = df.drop(target_column, axis=1)

y = df[target_column]

print(X.head())

print(y.head())


# In[12]:


categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

print("Categorical Features:", len(categorical_features))
print("Numerical Features:", len(numerical_features))


# In[16]:


target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)


# In[17]:


from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# In[19]:


models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "XGBoost": XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False
    )
}


# In[20]:


results = {}

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = {
        'model': pipeline,
        'accuracy': accuracy
    }

    print(f"{name} Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))


# In[21]:


best_model_name = max(results, key=lambda x: results[x]['accuracy'])

best_model = results[best_model_name]['model']

print("\nBest Model:", best_model_name)
print("Best Accuracy:", results[best_model_name]['accuracy'])


# In[22]:


joblib.dump(best_model, "traffic_severity_model.pkl")

print("Model Saved Successfully!")


# In[24]:


loaded_model = joblib.load("traffic_severity_model.pkl")


# In[25]:


sample_prediction = loaded_model.predict(X_test.iloc[:5])

decoded_predictions = target_encoder.inverse_transform(sample_prediction)

print(decoded_predictions)


# In[26]:


predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# In[27]:


model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]

plt.figure(figsize=(10,5))

sns.barplot(
    x=model_names,
    y=accuracies
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()


# In[28]:


rf_pipeline = results["Random Forest"]['model']

classifier = rf_pipeline.named_steps['classifier']

feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()

importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
).head(15)

plt.figure(figsize=(12,8))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Top 15 Important Features")

plt.show()


# In[ ]:




