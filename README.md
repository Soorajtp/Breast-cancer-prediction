## Breast Cancer Prediction using with ML

#### 1. Problem Statement
   
This project implements and compares multiple machine learning classification models to predict whether a breast tumor is Benign or Malignant using the Breast Cancer Wisconsin (Diagnostic) dataset. The objective is to evaluate different classifiers using standard performance metrics and deploy the solution as an interactive Streamlit web application.

#### 3. Dataset Description
    Dataset: Breast Cancer Wisconsin (Diagnostic)
    Source: UCI (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
    Samples: 569
    Features: 30 numerical features
    Target: Diagnosis (0 = Benign, 1 = Malignant)

Features are computed from digitized images of fine needle aspirate (FNA) of breast masses and describe characteristics of cell nuclei. The dataset satisfies the assignment constraints with more than 12 features and more than 500 instances.

#### 3. Models Implemented
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors (KNN)
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)

#### 4. Evaluation Metrics
    Each model is evaluated using:
    Accuracy, AUC, Precision, Recall, F1 Score, MCC etc.

#### 5.Comparison table

    |    ML Model    | Accuracy |  AUC  | Precision | Recall |   F1  |  MCC  |
    |:--------------:|:--------:|:-----:|:---------:|:------:|:-----:|:-----:|
    | Log.Regression | 0.965    | 0.996 | 0.975     | 0.928  | 0.951 | 0.925 |
    | Decision Tree  | 0.929    | 0.924 | .905      | 0.905  | 0.905 | 0.849 |
    | KNN            | 0.956    | 0.982 | 0.974     | 0.905  | 0.938 | 0.906 |
    | Naive Bayes    | 0.921    | 0.989 | 0.923     | 0.857  | 0.889 | 0.829 |
    | R.Forest       | 0.973    | 0.992 | 1.000     | 0.929  | 0.963 | 0.944 |
    | XG Boost       | 0.973    | 0.994 | 1.000     | 0.929  | 0.963 | 0.944 |


#### 6. Observations on model Performance
   
      | Model               | Observation                                                                                                                                                                |
      |---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
      | Logistic regression | Performs strongly compared to all                                                                                                                                          |
      | Decision Tree       | Plot depth Vs accuracy initially improved performance and settled at 9.<br> Showed comparatively lower performance.<br> Best max depth was 7 with accuracy 0.94                    |
      | KNN                 | Smaller values, we can see the noise and later it stabilizes for values of 7 and thereafter is constant.<br> Best value of K is 3  were accuracy was highest                   |
      | Naïve Bays          | Comparable or slightly better than Random Forest with strong generalization Random Forest:<br> Achieves high accuracy and MCC, demonstrating the benefit of ensemble learning. |
      | Random Forest       | Achieves high accuracy and MCC, demonstrating the benefit of ensemble learning.<br> Very good performance on unseen data                                                       |
      | XGBoost             | Slightly better than Random Forest with generalization as it gives good result with test data                                                                              |
