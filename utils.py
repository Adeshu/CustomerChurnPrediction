# Visualization Utilities for Customer Churn Prediction

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_churn_distribution(data):
    """
    Plots the distribution of churn data.
    :param data: DataFrame containing the customer data with a 'churn' column.
    """     
    plt.figure(figsize=(10, 6))
    sns.countplot(x='churn', data=data)
    plt.title('Customer Churn Distribution')
    plt.ylabel('Number of Customers')
    plt.xlabel('Churn')
    plt.show()


def plot_feature_importance(importances, feature_names):
    """
    Plots feature importance for model predictions.
    :param importances: Array of feature importance scores.
    :param feature_names: List of feature names corresponding to importances.
    """  
    feature_importances = pd.Series(importances, index=feature_names)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance')
    plt.show()


def plot_heatmap(correlation_matrix):
    """
    Plots a heatmap for the correlation matrix of features.
    :param correlation_matrix: DataFrame containing the correlation matrix.
    """  
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()