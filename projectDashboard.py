import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Multi-Segment Dashboard", layout="wide")

# Function to load data from Excel
@st.cache_data
def load_excel_data(sheet_name):
    data = pd.read_excel('OrdersDB.xlsx', sheet_name=sheet_name)
    return data

# Function to load data from CSV
@st.cache_data
def load_csv_data():
    data = pd.read_csv('worker_productivity.csv')
    return data

# Function for Customer Analysis
def customer_analysis():
    st.title("Customer Analysis")
    
    customer_data = load_excel_data('Customer')
    orders_data = load_excel_data('Orders')
    
    st.write("## Customer Data")
    st.dataframe(customer_data.head())
    
    st.write("## Data Description")
    st.write(customer_data.describe())
    
    # Merging customer data with orders for analysis
    merged_data = pd.merge(orders_data, customer_data, on='customer_id')
    
    st.write("## Merged Customer Orders Data")
    st.dataframe(merged_data.head())
    
    # Calculate total sales per customer
    customer_sales = merged_data.groupby('customer_id')['sales'].sum().reset_index()
    customer_sales = customer_sales.sort_values(by='sales', ascending=False)
    
    # Identify top 10 customers based on total sales
    top_customers = customer_sales.head(10)
    
    st.write("## Top 10 Most Profitable Customers")
    st.dataframe(top_customers)
    
    # Visualization: Total Sales by Customer
    st.write("## Total Sales by Customer")
    st.bar_chart(top_customers.set_index('customer_id')['sales'])
    
    # Additional recommendations
    st.write("### Recommendations")
    st.write("The top 10 customers based on total sales are identified. Focus your sales and marketing efforts on these customers to maximize profitability and strengthen relationships.")


# Function for Product Analysis
def product_analysis():
    st.title("Product Analysis")
    
    product_data = load_excel_data('Product')
    category_data = load_excel_data('Category')
    orders_data = load_excel_data('Orders')
    
    st.write("## Product Data")
    st.dataframe(product_data.head())
    
    st.write("## Data Description")
    st.write(product_data.describe())
    
    # Merging product data with category data
    merged_data = pd.merge(product_data, category_data, on='category_id')
    
    st.write("## Merged Product Category Data")
    st.dataframe(merged_data.head())
    
    # Merging with orders data to analyze sales
    merged_orders = pd.merge(merged_data, orders_data, on='product_id')
    
    # Calculate total sales for each product
    product_sales = merged_orders.groupby('product_name')['sales'].sum().reset_index()
    product_sales = product_sales.sort_values(by='sales', ascending=False)
    
    st.write("## Total Sales by Product")
    st.bar_chart(product_sales.set_index('product_name')['sales'])
    
    # Analyze average sales per product
    average_sales = merged_orders.groupby('product_name')['sales'].mean().reset_index()
    average_sales = average_sales.sort_values(by='sales', ascending=False)
    
    st.write("## Average Sales per Product")
    st.bar_chart(average_sales.set_index('product_name')['sales'])
    
    # Identifying top products based on total sales
    top_products = product_sales.head(10)
    st.write("## Top 10 Products by Total Sales")
    st.dataframe(top_products)
    
    # Additional analysis
    st.write("### Recommendations")
    st.write("Based on the total sales data, consider investing more in the products that appear in the top 10 list. These products are currently performing well and may have higher demand.")

# Function for Employee Analysis
def employee_analysis():
    st.title("Employee Analysis")
    
    data = load_csv_data()
    st.write("## Employee Productivity Data")
    st.dataframe(data.head())
    
    st.write("## Data Description")
    st.write(data.describe())
    
    # Check if 'actual_productivity' exists for prediction
    if 'actual_productivity' in data.columns:
        st.write("## Productivity Prediction using Random Forest")
        
        # Using 'department', 'team', and 'targeted_productivity' for prediction
        X = data[['department', 'team', 'targeted_productivity']]
        y = data['actual_productivity']
        
        # Encode 'department' using one-hot encoding
        X = pd.get_dummies(X, columns=['department'], drop_first=True)
        
        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Making predictions
        y_pred = model.predict(X_test)
        
        # Calculating and displaying Mean Squared Error
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"### Mean Squared Error: {mse}")
        
        # Visualizing the actual vs predicted productivity
        st.write("### Actual vs Predicted Productivity")
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.line_chart(result_df)
    else:
        st.warning("The dataset does not contain an 'actual_productivity' column for prediction.")

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Customers", "Products", "Employees"])

if selection == "Customers":
    customer_analysis()
elif selection == "Products":
    product_analysis()
elif selection == "Employees":
    employee_analysis()

# HTML content
# Complete HTML content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .navbar {
            display: flex;
            background-color: #333;
            padding: 10px;
            color: #fff;
        }
        .navbar a {
            color: #fff;
            padding: 14px 20px;
            text-decoration: none;
            text-align: center;
        }
        .navbar a:hover {
            background-color: #ddd;
            color: #333;
        }
        .tabs {
            display: none;
            padding: 20px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="#customers" onclick="openTab(event, 'customers')">Customers</a>
        <a href="#products" onclick="openTab(event, 'products')">Products</a>
        <a href="#employees" onclick="openTab(event, 'employees')">Employees</a>
    </div>

    <div id="customers" class="tab-content">
        <h2>Customer Analysis</h2>
        <p>Results and charts for customer analysis will be displayed here.</p>
    </div>

    <div id="products" class="tab-content">
        <h2>Product Analysis</h2>
        <p>Results and charts for product analysis will be displayed here.</p>
    </div>

    <div id="employees" class="tab-content">
        <h2>Employee Analysis</h2>
        <p>Results and charts for employee analysis will be displayed here.</p>
    </div>

    <script>
        function openTab(event, tabId) {
            var i, tabcontent;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            document.getElementById(tabId).style.display = "block";
        }

        // Default open tab
        document.addEventListener("DOMContentLoaded", function() {
            openTab(null, 'customers');
        });
    </script>
</body>
</html>
"""

# Saving HTML to a file
with open('dashboard.html', 'w') as file:
    file.write(html_content)


