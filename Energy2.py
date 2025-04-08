import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'C:/Users/2273581/Downloads/Energy_dataset.csv'
df = pd.read_csv(file_path)

# Define the features (X) and target (y)
X = df[['energy_consumption_kwh', 'peak_hours_usage', 'off_peak_usage', 'renewable_energy_pct', 'household_size', 'temperature_avg']]
y = df['billing_amount']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=rf_param_grid,
                              cv=3,
                              n_jobs=-1,
                              verbose=2)

# Fit the model
rf_grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and best score
rf_best_params = rf_grid_search.best_params_
rf_best_score = rf_grid_search.best_score_

print(f"Best parameters for Random Forest: {rf_best_params}")
print(f"Best cross-validation score for Random Forest: {rf_best_score}")

# Evaluate the best Random Forest model on the test set
rf_best_model = rf_grid_search.best_estimator_
rf_y_pred = rf_best_model.predict(X_test_scaled)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

print(f"Random Forest - Mean Squared Error: {rf_mse}")
print(f"Random Forest - R-squared: {rf_r2}")

# Define the parameter grid for Decision Tree
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV for Decision Tree
dt_grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                              param_grid=dt_param_grid,
                              cv=3,
                              n_jobs=-1,
                              verbose=2)

# Fit the model
dt_grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and best score
dt_best_params = dt_grid_search.best_params_
dt_best_score = dt_grid_search.best_score_

print(f"Best parameters for Decision Tree: {dt_best_params}")
print(f"Best cross-validation score for Decision Tree: {dt_best_score}")

# Evaluate the best Decision Tree model on the test set
dt_best_model = dt_grid_search.best_estimator_
dt_y_pred = dt_best_model.predict(X_test_scaled)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)

print(f"Decision Tree - Mean Squared Error: {dt_mse}")
print(f"Decision Tree - R-squared: {dt_r2}")
