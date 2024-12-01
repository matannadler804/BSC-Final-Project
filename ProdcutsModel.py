# -*- coding: utf-8 -*-
"""
Created on Thu May 23 07:57:09 2024

@author: gtafu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import xgboost as xgb

file_path = 'OrdersDB.xlsx'
sheets = pd.read_excel(file_path, sheet_name=['Orders', 'Customer', 'Product', 'Category'])


orders_df = sheets['Orders']
customer_df = sheets['Customer']
product_df = sheets['Product']
category_df = sheets['Category']

print("Orders DataFrame:")
print(orders_df.head())
print("\nCustomer DataFrame:")
print(customer_df.head())
print("\nProduct DataFrame:")
print(product_df.head())
print("\nCategory DataFrame:")
print(category_df.head())

# ----------------------------------------------------------------------------
#-------------------ProductsModel----------------------------------------------
# ----------------------------------------------------------------------------

# Merge the data including the order_date
merged_df2 = orders_df.merge(customer_df, on='customer_id') \
                      .merge(product_df, on='product_id') \
                      .merge(category_df, on='category_id')

# Adding the order_date in the product_metrics
merged_df2['order_date'] = pd.to_datetime(merged_df2['order_date'])

# Extract the month from 'order_date'
merged_df2['order_month'] = merged_df2['order_date'].dt.month

# Group by 'product_id', 'category_id', and 'order_month'
product_metrics = merged_df2.groupby(['product_id', 'category_id', 'order_month']).agg({
    'sales': ['sum', 'mean'],
    'quantity': 'sum',
    'order_id_number': 'count'
}).reset_index()

# Flatten the multi-level columns
product_metrics.columns = ['product_id', 'category_id', 'order_month', 'TotalSales', 'AvgSales', 'TotalQuantity', 'OrderCount']

print(product_metrics)


X_product=product_metrics[['AvgSales','TotalQuantity','OrderCount']]
y_product=product_metrics['TotalSales']

scaler = StandardScaler()

X_product_scaled=scaler.fit_transform(X_product)

X_product_train,X_prodcut_test,y_product_train,y_product_test=train_test_split(X_product_scaled,y_product,test_size=0.4,random_state=42)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_product_train,y_product_train)
y_product_pred=regressor.predict(X_prodcut_test)

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mse=mean_squared_error(y_product_test, y_product_pred)
r2=r2_score(y_product_test, y_product_pred)
mae = mean_absolute_error(y_product_test, y_product_pred)
rmse = np.sqrt(mean_squared_error(y_product_test, y_product_pred))
mape = np.mean(np.abs((y_product_test - y_product_pred) / y_product_test)) * 100
mpe = np.mean((y_product_test - y_product_pred) / y_product_test) * 100
rmspe = np.sqrt(np.mean(((y_product_test - y_product_pred) / y_product_test) ** 2)) * 100

print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Percentage Error (MPE): {mpe:.2f}%")
print(f"Root Mean Squared Percentage Error (RMSPE): {rmspe:.2f}%")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Error : {mse}")
print(f"R-squared: {r2}")

print()

plt.scatter(y_product_test,y_product_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# ----------------------------------------------------------------------------
#-------------------Tries----------------------------------------------
# ---------------------------------------------------------------------------

features = ['category_id', 'order_month']
target_sales = 'TotalSales'
target_quantity = 'TotalQuantity'

# Split data into train and test sets for TotalSales
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(product_metrics[features], product_metrics[target_sales], test_size=0.2, random_state=42)

# Split data into train and test sets for TotalQuantity
X_train_quantity, X_test_quantity, y_train_quantity, y_test_quantity = train_test_split(product_metrics[features], product_metrics[target_quantity], test_size=0.2, random_state=42)

# Models to train
models_sales = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

models_quantity = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

# Function to train and evaluate models
def train_evaluate_and_r2(models, X_train, y_train, X_test, y_test, target_name):
    best_model = None
    best_mse = float('inf')
    best_r2 = float('-inf')
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"{name} {target_name} Mean Squared Error: {mse}")
        print(f"{name} {target_name} R-squared: {r2}")
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_r2 = r2
    return best_model, best_r2



def rate_model_performance(r2):
    if r2 >= 0.8:
        return "Excellent (80-100%)"
    elif r2 >= 0.6:
        return "Good (60-80%)"
    elif r2 >= 0.3:
        return "Moderate (30-60%)"
    else:
        return "Poor (0-30%)"

# Training and evaluating models for TotalSales
print("TotalSales Prediction:")
best_model_sales, best_r2_sales = train_evaluate_and_r2(models_sales, X_train_sales, y_train_sales, X_test_sales, y_test_sales, "TotalSales")

# Training and evaluating models for TotalQuantity
print("TotalQuantity Prediction:")
best_model_quantity, best_r2_quantity = train_evaluate_and_r2(models_quantity, X_train_quantity, y_train_quantity, X_test_quantity, y_test_quantity, "TotalQuantity")

# Example prediction
category_id = 1.2
order_month = 8
input_data = pd.DataFrame({'category_id': [category_id], 'order_month': [order_month]})

# Predicting with the best models
predicted_sales = best_model_sales.predict(input_data)
predicted_quantity = best_model_quantity.predict(input_data)


print(f"TotalSales R-squared: {best_r2_sales} ({rate_model_performance(best_r2_sales)})")
print(f"TotalQuantity R-squared: {best_r2_quantity} ({rate_model_performance(best_r2_quantity)})")
print(f"Input: Category ID = {category_id}, Order Month = {order_month}")
print(f"Predicted TotalSales: {predicted_sales[0]}")
print(f"Predicted TotalQuantity: {predicted_quantity[0]}")

print()

# ----------------------------------------------------------------------------
#-------------------Tries----------------------------------------------
# -------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder


predicted_category = 1.4  # Example category
predicted_month = 2  # Example month

# Filter product_metrics DataFrame based on category and month
predicted_data = product_metrics[(product_metrics['category_id'] == predicted_category) & 
                                 (product_metrics['order_month'] == predicted_month)]

# Find the product ID with the highest total sales
predicted_product = predicted_data.loc[predicted_data['TotalSales'].idxmax()]

# Extract the product ID
predicted_product_id = predicted_product['product_id']

print("Predicted Most Sales Product ID for Category {} in Month {}: {}".format(predicted_category, predicted_month, predicted_product_id))

predicted_data_quantity = product_metrics[(product_metrics['category_id'] == predicted_category) & 
                                          (product_metrics['order_month'] == predicted_month)]

# Find the product ID with the highest total quantity
predicted_product_quantity = predicted_data_quantity.loc[predicted_data_quantity['TotalQuantity'].idxmax()]

# Extract the product ID
predicted_product_id_quantity = predicted_product_quantity['product_id']

print("Predicted Product ID for Category {} in Month {} based on Quantity: {}".format(predicted_category, predicted_month, predicted_product_id_quantity))
print()

category_data = product_metrics[product_metrics['category_id'] == predicted_category]

# One-hot encode the 'product_id' column
encoder = OneHotEncoder(sparse=False)
product_id_encoded = encoder.fit_transform(category_data[['product_id']])

# Combine one-hot encoded 'product_id' with other features
X = np.concatenate([product_id_encoded, category_data[['order_month']].values], axis=1)
y = category_data['TotalSales']  # Assuming we're predicting TotalSales

# Train regression models
linear_model = LinearRegression().fit(X, y)
xgboost_model = XGBRegressor().fit(X, y)
gradient_boosting_model = GradientBoostingRegressor().fit(X, y)
random_forest_model = RandomForestRegressor().fit(X, y)

# Make predictions for each model
linear_predicted_sales = linear_model.predict(X)
xgboost_predicted_sales = xgboost_model.predict(X)
gradient_boosting_predicted_sales = gradient_boosting_model.predict(X)
random_forest_predicted_sales = random_forest_model.predict(X)

# Aggregate predicted sales for each product across all months
predicted_sales_aggregated = (
    category_data
    .assign(linear_predicted_sales=linear_predicted_sales,
            xgboost_predicted_sales=xgboost_predicted_sales,
            gradient_boosting_predicted_sales=gradient_boosting_predicted_sales,
            random_forest_predicted_sales=random_forest_predicted_sales)
    .groupby('product_id')
    .agg({
        'linear_predicted_sales': 'sum',
        'xgboost_predicted_sales': 'sum',
        'gradient_boosting_predicted_sales': 'sum',
        'random_forest_predicted_sales': 'sum'
    })
)

# Find the product with the highest overall predicted sales
predicted_product_id = predicted_sales_aggregated.sum(axis=1).idxmax()

print("Predicted Product ID with the highest overall sales for Category {}: {}".format(predicted_category, predicted_product_id))
print()
# ----------------------------------------------------------------------------
#-------------------Tries2----------------------------------------------
# ------------------------------------------------------------------------
from sklearn.feature_selection import SelectFromModel




features = ['category_id', 'order_month']
target_sales = 'TotalSales'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(product_metrics[features], product_metrics[target_sales], test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection with RandomForest
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
selector = SelectFromModel(model, prefit=True)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Hyperparameter Tuning for Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=5)
grid_search_gb.fit(X_train_selected, y_train)
best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test_selected)

# Evaluate Gradient Boosting model
r2_gb = r2_score(y_test, y_pred_gb)
print("Gradient Boosting Model (TotalSales) R-squared:", r2_gb)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_selected, y_train)
y_pred_lr = lr_model.predict(X_test_selected)

# Evaluate Linear Regression model
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression Model (TotalSales) R-squared:", r2_lr)

# XGBoost
xgb_model = XGBRegressor(random_state=42)
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5)
grid_search_xgb.fit(X_train_selected, y_train)
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test_selected)

# Evaluate XGBoost model
r2_xgb = r2_score(y_test, y_pred_xgb)
print("XGBoost Model (TotalSales) R-squared:", r2_xgb)

# Ridge Regression
ridge_model = Ridge(alpha=0.1)  # You can adjust the alpha parameter for regularization strength
ridge_model.fit(X_train_selected, y_train)
y_pred_ridge = ridge_model.predict(X_test_selected)

# Evaluate Ridge Regression model
r2_ridge = r2_score(y_test, y_pred_ridge)
print("Ridge Regression Model (TotalSales) R-squared:", r2_ridge)

