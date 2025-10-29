🌾 Crop Price Prediction using Ridge Regression

This project builds a machine learning model to predict crop prices based on cultivation costs, weather conditions, and regional factors using Ridge Regression. It demonstrates the use of data preprocessing pipelines, column transformations, and evaluation metrics for regression analysis.

📊 Project Overview

The goal of this project is to predict the Price of a crop using features such as:

State and Crop (categorical variables)

Cost of Cultivation, Production, Yield, Temperature, and Rainfall (numerical variables)

The dataset is preprocessed using OneHotEncoder for categorical columns and StandardScaler for numerical columns.
A Ridge Regression model is then trained using a Pipeline, ensuring smooth preprocessing and modeling in one workflow.

🧠 Tech Stack

Python 🐍

Pandas — Data manipulation

Scikit-learn (sklearn) — Preprocessing, pipeline creation, and regression

Matplotlib & Seaborn — Visualization

Ridge Regression — Model for regularized linear regression
