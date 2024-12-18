# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:05:38 2024

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

file_path2 = "worker_productivity.csv"


worker_productivity_df = pd.read_csv(file_path2)


worker_productivity_df['department'] = worker_productivity_df['department'].str.lower()


worker_productivity_df['department'] = worker_productivity_df['department'].str.strip()


print(worker_productivity_df['department'].unique())


unique_departments_count = worker_productivity_df['department'].nunique()
print("Number of unique department types:", unique_departments_count)

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
#-------------------CustomerModel----------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#-------------------KmeansModel----------------------------------------------
# ----------------------------------------------------------------------------

merged_df = orders_df.merge(customer_df, on='customer_id')\
                     .merge(product_df, on='product_id')\
                     .merge(category_df, on='category_id')

# Calculating customer metrics
customer_metrics = merged_df.groupby('customer_id').agg({
    'sales': ['sum', 'mean', 'count'],
    'product_id': 'nunique',
    'quantity': 'sum'
}).reset_index()


customer_metrics.columns = ['customer_id', 'TotalSpent', 'AvgSpent', 'OrderCount',
                            'UniqueProducts', 'TotalQuantity']


X = customer_metrics[['TotalSpent', 'AvgSpent', 'OrderCount', 'UniqueProducts', 'TotalQuantity']]


scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)


X_train, X_test = train_test_split(X_scaled, random_state=42)


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)


train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)


train_silhouette = silhouette_score(X_train, train_clusters)
test_silhouette = silhouette_score(X_test, test_clusters)
train_inertia = kmeans.inertia_

print(f"Train Inertia: {train_inertia}")
print(f"Train Silhouette Score: {train_silhouette}")
print(f"Test Silhouette Score: {test_silhouette}")

print('Feature indices after scaling:')
print(f"TotalSpent index: {list(customer_metrics.columns).index('TotalSpent') - 1}")  # Subtract 1 due to exclusion of 'customer_id'
print(f"AvgSpent index: {list(customer_metrics.columns).index('AvgSpent') - 1}")

plt.scatter(X_train[:,0],X_train[:,1],c=train_clusters,cmap="viridis",marker='o',alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c='red',marker='x')
plt.title('Customer Training Data')
plt.xlabel('TotalSpent')
plt.ylabel('AvgSpent')
plt.show()


total_spent_index = list(customer_metrics.columns).index('TotalSpent'.strip()) - 1  # -1 to adjust for 'customer_id' exclusion
order_count_index = list(customer_metrics.columns).index('OrderCount'.strip()) - 1  # -1 to adjust for 'customer_id' exclusion


print('Feature indices after scaling:')
print(f"TotalSpent index: {total_spent_index}")
print(f"OrderCount index: {order_count_index}")


assert total_spent_index >= 0, "TotalSpent index is invalid!"
assert order_count_index >= 0, "OrderCount index is invalid!"


plt.scatter(X_train[:, total_spent_index], X_train[:, order_count_index], c=train_clusters, cmap='viridis', marker='o', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, total_spent_index], kmeans.cluster_centers_[:, order_count_index], s=300, c='red', marker='x')
plt.title('Customer Training Data')
plt.xlabel('TotalSpent')
plt.ylabel('OrderCount')
plt.show()
print("Kmeans acc")
test_predicted_clusters = kmeans.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(test_clusters, test_predicted_clusters)

# Calculate MSE
mse = mean_squared_error(test_clusters, test_predicted_clusters)

# Calculate RMSE
rmse = np.sqrt(mse)

r2 = r2_score(test_clusters, test_predicted_clusters)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")


print()
#----------------------------------------------------------------------------
#-------------------HirarchicalCLuster---------------------------------------------
#----------------------------------------------------------------------------
print("HirarchicalCLuster")
merged_df2=orders_df.merge(customer_df,on='customer_id')\
    .merge(product_df,on='product_id')\
        .merge(category_df,on='category_id')

customer_metrics2=merged_df2.groupby('customer_id').agg({
    'sales':['sum','mean','count'],
    'product_id':'nunique',
    'quantity':'sum'
    }).reset_index()

customer_metrics2.columns=['customer_id','TotalSpent','AvgSpent','OrderCount','UniqueProducts','TotalQuantity']

X2=customer_metrics2[['TotalSpent','AvgSpent','OrderCount','UniqueProducts','TotalQuantity']]
scaler2=StandardScaler()
X_scaled2=scaler2.fit_transform(X2)

X2_train,X2_test=train_test_split(X_scaled2,random_state=42)
agg_cluster2=AgglomerativeClustering(n_clusters=2)
agg_cluster2.fit(X2_train)

train_clusters2=agg_cluster2.labels_
test_clusters2=agg_cluster2.fit_predict(X2_test)

train_silhouette2=silhouette_score(X2_train,train_clusters2)
test_silhouette2=silhouette_score(X2_test,test_clusters2)

print(f"Train Silhoutte Score: {train_silhouette2}")
print(f"Test Silhoutte Score: {test_silhouette2}")

linked=linkage(X2_train,'ward')
plt.figure(figsize=(10,7))
dendrogram(linked,orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Dendogram')
plt.show()

plt.scatter(X2_train[:,0],X2_train[:,1],c=train_clusters2,cmap='viridis',marker='o',alpha=0.5)
plt.title('Customer Training Data - Hierarchical Clustering')
plt.xlabel('TotalSpent')
plt.ylabel('AvgSpent')
plt.show()

total_spent_index2 = list(customer_metrics2.columns).index('TotalSpent') -1
order_count_index2 = list(customer_metrics2.columns).index('OrderCount') -1

plt.scatter(X2_train[:,total_spent_index2],X2_train[:,order_count_index2],c=train_clusters2,cmap='viridis',marker='o',alpha=0.5)
plt.title('Customer Training Data - Hierarchical Clustering')
plt.xlabel('TotalSpent')
plt.ylabel('OrderCount')
plt.show()

print()
#----------------------------------------------------------------------------# ----------------------------------------------------------------------------
#-------------------CustomerPredictions----------------------------------------------
# ----------------------------------------------------------------------------

X3 = customer_metrics2[['AvgSpent', 'UniqueProducts', 'TotalQuantity']]  # Removed OrderCount
y3_total_spent = customer_metrics2['TotalSpent']
y3_order_count = customer_metrics2['OrderCount']

# Split the data into train and test sets for TotalSpent prediction
X3_train_total_spent, X3_test_total_spent, y3_total_spent_train, y3_total_spent_test = train_test_split(
    X3, y3_total_spent, test_size=0.2, random_state=42
)

# Split the data into train and test sets for OrderCount prediction
X3_train_order_count, X3_test_order_count, y3_order_count_train, y3_order_count_test = train_test_split(
    X3, y3_order_count, test_size=0.2, random_state=42
)

# Define models
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

results = {}

# Train and evaluate models for TotalSpent prediction
for name, model in models.items():
    model.fit(X3_train_total_spent, y3_total_spent_train)
    y3_total_spent_pred = model.predict(X3_test_total_spent)
    
    mae = mean_absolute_error(y3_total_spent_test, y3_total_spent_pred)
    mse = mean_squared_error(y3_total_spent_test, y3_total_spent_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y3_total_spent_test, y3_total_spent_pred)
    
    results[f"{name}_TotalSpent"] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Train and evaluate models for OrderCount prediction
for name, model in models.items():
    model.fit(X3_train_order_count, y3_order_count_train)
    y3_order_count_pred = model.predict(X3_test_order_count)
    
    mae = mean_absolute_error(y3_order_count_test, y3_order_count_pred)
    mse = mean_squared_error(y3_order_count_test, y3_order_count_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y3_order_count_test, y3_order_count_pred)
    
    results[f"{name}_OrderCount"] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

# Print results
for key, value in results.items():
    print(f"{key} Results:")
    for metric, score in value.items():
        print(f"{metric}: {score}")
    print("\n")
    
print()
#----------------------------------------------------------------------------# ----------------------------------------------------------------------------
#-------------------Tries----------------------------------------------
# ----------------------------------------------------------------------------

X = customer_metrics2[['TotalSpent']]
y = customer_metrics2['OrderCount']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

# Loop through each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    
    
    # Function to predict OrderCount based on TotalSpent
    def predict_order_count(total_spent, model=model):
        return model.predict([[total_spent]])[0]
    
    # Example usage
    total_spent_input = 10000
    predicted_order_count = predict_order_count(total_spent_input)
    print(f"If you spend ${total_spent_input}, the {name} model predicts you will come to buy approximately {predicted_order_count} times.\n")
print()

X = customer_metrics2[['OrderCount']]
y = customer_metrics2['TotalSpent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

# Loop through each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    
    
    # Function to predict TotalSpent based on OrderCount
    def predict_total_spent(order_count, model=model):
        return model.predict([[order_count]])[0]
    
    # Example usage
    order_count_input = 14
    predicted_total_spent = predict_total_spent(order_count_input)
    print(f"If you come to buy {order_count_input} times, the {name} model predicts you will spend approximately ${predicted_total_spent:.2f}.\n")
print()




#----------------------------------------------------------------------------# ----------------------------------------------------------------------------
#-------------------ModelPredictionsInput----------------------------------------------
# ----------------------------------------------------------------------------




#----------------------------------------------------------------------------# ----------------------------------------------------------------------------
#-------------------ProductsModel----------------------------------------------
# ----------------------------------------------------------------------------
print()

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
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.show()

# ----------------------------------------------------------------------------
#-------------------ProductsPredictions----------------------------------------------
# ------------------------------------------------------------------

def train_models(product_metrics):
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42),
        "LinearRegression": LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        # Prepare the data
        X = product_metrics[['category_id', 'OrderCount', 'TotalQuantity', 'AvgSales']]
        y = product_metrics['TotalSales']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "X_train_mean": X_train.mean()  # Store mean values of features
        }
    
    return results

# Train the models
model_results = train_models(product_metrics)

# Function to predict total sales for a given category_id
def predict_total_metrics(category_id):
    predictions = {}
    
    for name, result in model_results.items():
        model = result['model']
        
        # Use the mean values of the training set features
        input_features = pd.DataFrame({
            'category_id': [category_id],
            'OrderCount': [result['X_train_mean']['OrderCount']],
            'TotalQuantity': [result['X_train_mean']['TotalQuantity']],
            'AvgSales': [result['X_train_mean']['AvgSales']]
        })
        
        # Predict total sales
        total_sales_pred = model.predict(input_features)[0]
        # Ensure the predicted value is non-negative
        total_sales_pred = max(0, total_sales_pred)
        
        # Predict total quantity
        total_quantity_pred = model.predict(input_features)[0]
        # Ensure the predicted value is non-negative
        total_quantity_pred = max(0, total_quantity_pred)
        
        predictions[name] = {'TotalSales': total_sales_pred, 'TotalQuantity': total_quantity_pred}
    
    return predictions

# Example usage
category_id_input = 1.5
predicted_metrics = predict_total_metrics(category_id_input)

print(f"For category ID {category_id_input}, the predicted total sales and quantity are:")
for model_name, metrics in predicted_metrics.items():
    print(f"{model_name}: Sales - ${metrics['TotalSales']:.2f}, Quantity - {metrics['TotalQuantity']:.2f}")
    
# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
for name, result in model_results.items():
    print(f"\n{name} Model:")
    print(f"Mean Squared Error (MSE): {result['mse']}")
    print(f"Root Mean Squared Error (RMSE): {result['rmse']}")
    print(f"Mean Absolute Error (MAE): {result['mae']}")
    print(f"R² Score: {result['r2']}")
    
print()
# ----------------------------------------------------------------------------
#-------------------PorductsTries----------------------------------------------
# -----------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





# Sort the data by 'category_id' and 'order_month'
product_metrics = product_metrics.sort_values(by=['category_id', 'order_month'])

# Create lagged features for the past three months
for lag in range(1, 4):
    product_metrics[f'TotalSales_lag{lag}'] = product_metrics.groupby('category_id')['TotalSales'].shift(lag)
    product_metrics[f'TotalQuantity_lag{lag}'] = product_metrics.groupby('category_id')['TotalQuantity'].shift(lag)

# Drop rows with NaN values created by lagging
product_metrics = product_metrics.dropna().reset_index(drop=True)

# Define features and target variables
features = ['category_id', 'order_month', 'TotalSales_lag1', 'TotalSales_lag2', 'TotalSales_lag3', 
            'TotalQuantity_lag1', 'TotalQuantity_lag2', 'TotalQuantity_lag3']

target_sales = 'TotalSales'
target_quantity = 'TotalQuantity'

# Train-test split
X = product_metrics[features]
y_sales = product_metrics[target_sales]
y_quantity = product_metrics[target_quantity]

X_train, X_test, y_sales_train, y_sales_test, y_quantity_train, y_quantity_test = train_test_split(
    X, y_sales, y_quantity, test_size=0.2, random_state=42)

# Train models
def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LinearRegression": LinearRegression()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# Train models for sales prediction
sales_models = train_models(X_train, y_sales_train)

# Train models for quantity prediction
quantity_models = train_models(X_train, y_quantity_train)

# Evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
    return results

# Evaluate sales models
sales_results = evaluate_models(sales_models, X_test, y_sales_test)
print("Sales Model Evaluation:", sales_results)

# Evaluate quantity models
quantity_results = evaluate_models(quantity_models, X_test, y_quantity_test)
print("Quantity Model Evaluation:", quantity_results)

# Predict function
def predict_next_month(category_id, sales_models, quantity_models):
    # Find the latest month in the data
    latest_month = product_metrics['order_month'].max()
    next_month = (latest_month % 12) + 1
    
    # Get the latest data for the category
    latest_data = product_metrics[product_metrics['category_id'] == category_id].tail(3)
    
    # Prepare the input data for prediction
    next_month_input = pd.DataFrame({
        'category_id': [category_id],
        'order_month': [next_month],
        'TotalSales_lag1': [latest_data['TotalSales'].iloc[-1]],
        'TotalSales_lag2': [latest_data['TotalSales'].iloc[-2]],
        'TotalSales_lag3': [latest_data['TotalSales'].iloc[-3]],
        'TotalQuantity_lag1': [latest_data['TotalQuantity'].iloc[-1]],
        'TotalQuantity_lag2': [latest_data['TotalQuantity'].iloc[-2]],
        'TotalQuantity_lag3': [latest_data['TotalQuantity'].iloc[-3]]
    })
    
    # Predict sales and quantity using the best models (based on evaluation)
    best_sales_model = sales_models['RandomForest']  # Example: choosing RandomForest
    best_quantity_model = quantity_models['RandomForest']  # Example: choosing RandomForest
    
    predicted_sales = best_sales_model.predict(next_month_input)
    predicted_quantity = best_quantity_model.predict(next_month_input)
    
    print(f"Predicted Sales for category {category_id} in next month: {predicted_sales[0]}")
    print(f"Predicted Quantity for category {category_id} in next month: {predicted_quantity[0]}")

# Example usage
category_id = 1.4
predict_next_month(category_id, sales_models, quantity_models)

# Print model evaluation metrics
print("\nModel Evaluation Metrics:")
for name, result in sales_results.items():
    print(f"\n{name} Sales Model:")
    print(f"Mean Squared Error (MSE): {result['mse']}")
    print(f"Root Mean Squared Error (RMSE): {result['rmse']}")
    print(f"Mean Absolute Error (MAE): {result['mae']}")
    print(f"R² Score: {result['r2']}")

for name, result in quantity_results.items():
    print(f"\n{name} Quantity Model:")
    print(f"Mean Squared Error (MSE): {result['mse']}")
    print(f"Root Mean Squared Error (RMSE): {result['rmse']}")
    print(f"R² Score: {result['r2']}")






# ----------------------------------------------------------------------------
#-------------------WorkersMoedl----------------------------------------------
# ----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


worker_productivity_df['department'] = worker_productivity_df['department'].str.lower()
worker_productivity_df['day'] = worker_productivity_df['day'].str.lower()


class_mapping = {'finishing': 1, 'sweing': 2}
day_mapping = {'sunday': 1, 'monday': 2, 'tuesday': 3, 'wednesday': 4, 'thursday': 5, 'friday': 6, 'saturday': 7}


filtered_df = worker_productivity_df[['team', 'department', 'day', 'targeted_productivity', 'actual_productivity']]


filtered_df['department'] = filtered_df['department'].map(class_mapping)
filtered_df['day'] = filtered_df['day'].map(day_mapping)


filtered_df.dropna(inplace=True)

X = filtered_df[['team', 'department', 'day']]
y = (filtered_df['actual_productivity'] >= filtered_df['targeted_productivity']).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LogisticRegression Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


cv_scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean Cross-validation Score:", cv_scores.mean())

input_dep = 'sweing'
input_team = 8
input_day = 'saturday'

input_dep_num = class_mapping[input_dep]
input_day_num = day_mapping[input_day]

input_features = pd.DataFrame({
    'team': [input_team],
    'department': [input_dep_num],
    'day': [input_day_num]
})


input_features_scaled = pipeline.named_steps['standardscaler'].transform(input_features)

probability = pipeline.predict_proba(input_features_scaled)[:, 1]
print("Probability they will succeed in Predicted Productivity:", probability[0] * 100, "%")


# ----------------------------------------------------------------------------
#-------------------WorkersReportTries----------------------------------------------
# ----------------------------------------------------------------

random_forest_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
random_forest_pipeline.fit(X_train, y_train)
y_pred_rf = random_forest_pipeline.predict(X_test)

# Decision Tree Classifier
decision_tree_pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))
decision_tree_pipeline.fit(X_train, y_train)
y_pred_dt = decision_tree_pipeline.predict(X_test)

# Evaluate Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Evaluate Decision Tree Classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Classifier:")
print("Accuracy:", accuracy_dt)
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))


