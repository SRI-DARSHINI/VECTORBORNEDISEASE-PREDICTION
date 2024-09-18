# VECTORBORNEDISEASE-PREDICTION

Machine Learning Algorithms for Prediction of Dengue Cases in India:

Dataset Description:
This work uses climate data from the Indian Meteorological Department (IMD) and dengue data from OPENDENGUE to predict the infectious disease dengue. These Datasets consist of data from 1st January 1991 to 31st December 2021 for dengue 
cases. Dengue Information consists of The total number of dengue cases impacted annually in India over the mentioned period and Climate data comprises India's annual temperature as well as minimum and maximum temperatures. The amount of 
rain that fell in North East India between 1991 and 2021, the amount that fell in North West India between 1991 and 2021, the amount that fell in the South Peninsula between 1991 and 2021, the amount that fell in Central India between 1991 and 2021, the amount that fell in the Monsoon session in All India between 1991 and 2021, and the total amount of rain that fell in India between 1991 and 2021. Dataset also contains Country Name, Calendar Start Date, Calendar End date corresponding to dengue data and climate data. 

Data Preprocessing:
Before applying any machine learning models, data preprocessing is crucial for ensuring model performance. Below are the main steps involved in preparing the dataset for analysis:

Handling Missing Values: Missing data in numeric features is filled with the mean, and for categorical features, the most frequent value is used (via SimpleImputer).
Feature Scaling: Numeric features are scaled to zero mean and unit variance using StandardScaler.
Handling Categorical Data: Categorical variables are encoded using one-hot encoding (OneHotEncoder).
Dimensionality Reduction: To simplify the dataset and reduce computational complexity, Principal Component Analysis (PCA) is applied to reduce the dataset's dimensionality to two principal components.
The preprocessing is managed through a pipeline using Pipeline and ColumnTransformer from scikit-learn, ensuring efficient and modular data preparation. The data is split into training and testing sets using train_test_split.

Random Forest Regressor:
Random Forest is an ensemble learning technique that constructs multiple decision trees for both classification and regression tasks. It combines predictions from each tree (through voting or averaging) to generate a final output.

Random Sampling: Uses bootstrapping (random sampling with replacement) to create subsets of the training data.
Tree Construction: Decision trees are built for each subset, using a random subset of features at each split.
Final Prediction: For regression tasks like dengue prediction, the average of all trees’ predictions is used.
Advantages:

High accuracy, handles large datasets with high dimensionality, and reduces overfitting through randomization.
It can estimate feature importance, helping in selecting important predictors.
Implementation: The code initializes a RandomForestRegressor, selects climate data as features, and predicts total dengue cases. The results are visualized and evaluated using metrics like MAE, MSE, RMSE, and MAPE.

Support Vector Regression (SVR):
SVR aims to fit a regression model while keeping most of the data within a defined margin (epsilon).

Kernel Trick: SVR can use different kernels (e.g., linear, polynomial) to handle non-linear relationships.
Loss Function: The model minimizes error while controlling complexity through regularization parameters like C and epsilon.
Advantages:

Effective for non-linear data, useful in small-to-medium datasets, and handles high-dimensional data.
Implementation: The SVR model is trained on climate features to predict dengue cases. The model's performance is evaluated using MAE, MSE, RMSE, and MAPE.

XGBoost:
XGBoost is a powerful, efficient implementation of gradient boosting that constructs an ensemble of decision trees sequentially. Each tree corrects the errors of the previous one.

Key Features:
Uses both L1 and L2 regularization to prevent overfitting.
Parallel and distributed computing for fast training.
Customizable loss functions and built-in cross-validation.
Feature Importance: Provides scores to highlight the importance of features in the model.
Implementation: An XGBoost model is trained to predict dengue cases based on climate features. It offers high accuracy, making it ideal for structured data. Evaluation metrics such as MAE, MSE, RMSE, and MAPE are calculated for model assessment.

Ensemble Methods:
Ensemble methods like Gradient Boosting and AdaBoosting combine multiple models to improve predictive accuracy. Gradient Boosting builds models sequentially, minimizing prediction errors, while AdaBoost focuses on correcting misclassified examples.

Conclusion: By applying different machine learning models like Random Forest, SVR, and XGBoost to predict dengue cases, each model’s performance can be evaluated using metrics like MAE and RMSE. Preprocessing steps like scaling, handling missing values, and PCA play a crucial role in ensuring data quality, ultimately leading to better model accuracy.
