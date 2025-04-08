# Hyperparameter Tuning and Comparative Analysis of Machine Learning Models for Predicting Energy Billing Amounts

**Columns and their description in the dataset:**

1. **customer_id**: Unique identifier for each customer.
2. **region**: Geographic region of the customer.
3. **energy_consumption_kwh**: Total energy consumption in kilowatt-hours.
4. **peak_hours_usage**: Energy usage during peak hours.
5. **off_peak_usage**: Energy usage during off-peak hours.
6. **renewable_energy_pct**: Percentage of energy from renewable sources.
7. **billing_amount**: Total billing amount.
8. **household_size**: Number of people in the household.
9. **temperature_avg**: Average temperature.
10. **income_bracket**: Income bracket of the household.
11. **smart_meter_installed**: Whether a smart meter is installed.
12. **time_of_day_pricing**: Whether time-of-day pricing is used.
13. **annual_energy_trend**: Annual trend in energy consumption.
14. **solar_panel**: Whether solar panels are installed.
15. **target_high_usage**: Whether the household is targeted for high usage.

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
