
# Machine Learning Classification Project

## Problem Statement
Implement multiple classification models and compare their performance using evaluation metrics.

## Dataset Description
- Dataset uploaded by user (.csv format)
- Preprocessed using encoding and train-test split

## Models Used
1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

## Comparison Table

|                     |   Accuracy |      AUC |   Precision |   Recall |       F1 |        MCC |
|:--------------------|-----------:|---------:|------------:|---------:|---------:|-----------:|
| Logistic Regression |   0.781818 | 0.840157 |    0.781818 | 0.781818 | 0.781818 |  0.425087  |
| Decision Tree       |   0.754545 | 0.670732 |    0.751702 | 0.754545 | 0.753058 |  0.345628  |
| KNN                 |   0.772727 | 0.804225 |    0.770111 | 0.772727 | 0.77135  |  0.394122  |
| Naive Bayes         |   0.254545 | 0.48824  |    0.436364 | 0.254545 | 0.118818 | -0.0766798 |
| Random Forest       |   0.790909 | 0.858885 |    0.775151 | 0.790909 | 0.776329 |  0.395497  |
| XGBoost             |   0.763636 | 0.792683 |    0.769394 | 0.763636 | 0.766253 |  0.391919  |

## Observations


| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Performs well when data is linearly separable and provides stable baseline performance. |
| Decision Tree | Captures nonlinear patterns but may overfit on training data. |
| KNN | Sensitive to feature scaling and works well with local patterns. |
| Naive Bayes | Fast and efficient, performs well on probabilistic datasets. |
| Random Forest | Robust ensemble model with good accuracy and reduced overfitting. |
| XGBoost | Typically provides highest performance due to boosting and optimization. |


## Project Structure

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    │-- Logistic_Regression.pkl
    │-- Decision_Tree.pkl
    │-- KNN.pkl
    │-- Naive_Bayes.pkl
    │-- Random_Forest.pkl
    │-- XGBoost.pkl

