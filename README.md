# Hyperparameter Tuning and Comparative Analysis of Machine Learning Models for Predicting Energy Billing Amounts

**Steps for Hyperparameter Tuning:
**
1. Define the Parameter Grid: Specify the range of hyperparameters to be tested.
2. Initialize GridSearchCV: Use GridSearchCV to perform an exhaustive search over the specified parameter grid.
3. Fit the Model: Train the model using the training data.
4. Evaluate the Best Model: Evaluate the performance of the best model found during the search.

**Explanation:
**
1. Parameter Grid: Define the range of hyperparameters for Random Forest and Decision Tree.
2. GridSearchCV Initialization: Initialize GridSearchCV with the estimator, parameter grid, cross-validation strategy, and other settings.
3. Model Fitting: Fit the model using the training data.
4. Best Parameters and Score: Retrieve the best parameters and cross-validation score.
5. Model Evaluation: Evaluate the best model on the test set and print the Mean Squared Error (MSE) and R-squared (R²) score.
