#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:58:27 2024

@author: eepena
"""

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt



base_folder = os.path.expanduser("~/PodTracker_study")
data_folder = os.path.join(base_folder, "data/decision_tree_data")
results_folder = os.path.join(base_folder, "results")


df  = pd.read_csv(os.path.join(data_folder, 'balanced_df.csv'))

# Split the data into features (X) and target variable (y)
X = df[['area_cont','w_cont', 'l_cont']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier on the training data
clf.fit(X_train, y_train)


# Save the trained model to a file
with open('model_checkpoints/trained_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# Predict the size category on the testing data
y_pred = clf.predict(X_test)

report = classification_report(y_test, y_pred)

# Save classification report to a file
report_filename = os.path.join(results_folder, "tables/classification_report.csv")
with open(report_filename, "w") as report_file:
    report_file.write(report)
# Evaluate the performance of the decision tree classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define the labels for the confusion matrix
labels = clf.classes_


# Create a heatmap of the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt= 'd', xticklabels=labels, yticklabels=labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j +0.5, i+0.5, cm[i,j], ha = 'center', va='center', color= 'black')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix on the Validation Set')
plt.savefig(os.path.join(results_folder, 'plots/decisiontree_confusion_matrix.png'))