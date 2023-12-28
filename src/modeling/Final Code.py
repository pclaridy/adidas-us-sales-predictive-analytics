# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Load and preprocess dataset
file_path = 'Adidas US Sales Datasets.xlsx'
df = pd.read_excel(file_path)
df.drop(columns=['Unnamed: 0'], inplace=True)  # Drop irrelevant columns
df = df.dropna()  # Remove rows with missing values 

# Function to remove outliers
def remove_outliers(df, y_col, method='IQR', bounds=(0.01, 0.99)):
    if method == 'IQR':
        Q1 = df[y_col].quantile(bounds[0])
        Q3 = df[y_col].quantile(bounds[1])
        IQR = Q3 - Q1
        mask = (df[y_col] >= (Q1 - 1.5 * IQR)) & (df[y_col] <= (Q3 + 1.5 * IQR))
    elif method == 'z-score':
        from scipy import stats
        z = np.abs(stats.zscore(df[y_col]))
        mask = z < 3
    else:
        raise ValueError("Invalid method for outlier handling")
    return df[mask], mask

df_filtered, mask = remove_outliers(df, 'Total Sales')  # Apply outlier removal

# Data Transformation: Calculating Percentage of Sales and Formatting Price Per Unit
df_sales = df_filtered.groupby('Product')['Total Sales'].sum().reset_index()
total_sales = df_sales['Total Sales'].sum()
df_sales['Percentage of Sales'] = (df_sales['Total Sales'] / total_sales) * 100

df_price = df_filtered.groupby('Product')['Price per Unit'].mean().reset_index()
df_price['Price per Unit'] = df_price['Price per Unit'].apply(lambda x: f"${x:.2f}")

# Data Preparation for Modeling
X = df_filtered.drop(['Total Sales'], axis=1)  # Features
y_reg = df_filtered['Total Sales']  # Regression target
y_class = pd.qcut(df_filtered['Total Sales'], 3, labels=["low", "medium", "high"])  # Classification target

# Preprocessing pipeline for numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Model Pipelines
lin_reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
ada_reg = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', AdaBoostRegressor(random_state=42))])
log_reg = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
ada_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', AdaBoostClassifier(random_state=42))])

# Splitting the dataset for training and testing
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Model Training and Evaluation
lin_reg.fit(X_train_reg, y_train_reg)
ada_reg.fit(X_train_reg, y_train_reg)
log_reg.fit(X_train_class, y_train_class)
ada_clf.fit(X_train_class, y_train_class)

y_pred_lin_reg = lin_reg.predict(X_test_reg)
y_pred_ada_reg = ada_reg.predict(X_test_reg)
y_pred_log_reg = log_reg.predict(X_test_class)
y_pred_ada_clf = ada_clf.predict(X_test_class)

mse_lin_reg = mean_squared_error(y_test_reg, y_pred_lin_reg)
mse_ada_reg = mean_squared_error(y_test_reg, y_pred_ada_reg)
acc_log_reg = accuracy_score(y_test_class, y_pred_log_reg)
acc_ada_clf = accuracy_score(y_test_class, y_pred_ada_clf)

# Display model performance
print("Linear Regression MSE:", mse_lin_reg)
print("AdaBoost Regression MSE:", mse_ada_reg)
print("Logistic Regression Accuracy:", acc_log_reg)
print("AdaBoost Classification Accuracy:", acc_ada_clf)

# Visualization 1: Bar Chart of Total Sales by Product
product_sales = df_filtered.groupby('Product')['Total Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(15, 10))
product_sales.plot(kind='bar')
plt.title('Total Sales by Product')
plt.ylabel('Total Sales')
plt.xlabel('Product')
plt.xticks(rotation=15)
plt.show()

# Convert 'Invoice Date' to datetime format and set it as the index for df_filtered
df_filtered['Invoice Date'] = pd.to_datetime(df_filtered['Invoice Date'])
df_filtered.set_index('Invoice Date', inplace=True)

# Now, you can perform the resampling on the 'Total Sales' column
monthly_sales = df_filtered['Total Sales'].resample('M').sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title('Monthly Total Sales Over Time')
plt.ylabel('Total Sales')
plt.xlabel('Date')
plt.show()


# Visualization 3: Pie Chart of Sales by Region
region_sales = df_filtered.groupby('Region')['Total Sales'].sum()
plt.figure(figsize=(8, 8))
region_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales Distribution by Region')
plt.ylabel('')
plt.show()

# Visualization 4: Scatter Plot of Operating Profit vs. Total Sales
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Total Sales'], df_filtered['Operating Profit'])
plt.title('Operating Profit vs. Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Operating Profit')
plt.show()

# Boxplot for Prices per Unit by Product
plt.figure(figsize=(15, 6))  # Increase the figure size for better spacing
df.boxplot(column='Price per Unit', by='Product', rot=90)  # Rotate labels to 90 degrees
plt.title('Price per Unit by Product')
plt.ylabel('Price per Unit')
plt.xlabel('Product')
plt.xticks(ha='right')  # Align the x-ticks (product names) to the right for better readability
plt.tight_layout()  # Automatically adjust subplot params for the subplot(s) to fit in the figure area
plt.show()

# Visualization 6: Histogram of Units Sold
plt.figure(figsize=(10, 6))
df_filtered['Units Sold'].hist(bins=30)
plt.title('Distribution of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.show()

# Time Series Analysis - Augmented Dickey-Fuller test
adf_test_original = adfuller(df_filtered['Total Sales'])
print("ADF Statistic:", adf_test_original[0])
print("p-value:", adf_test_original[1])

# Seasonality Analysis
seasonality_period = 12

# Ensure the index is sorted
df_filtered.sort_index(inplace=True)

# Aggregating sales data by date
df_filtered_aggregated = df_filtered.groupby(df_filtered.index).sum()

# Set the frequency of the aggregated time series data
df_filtered_aggregated = df_filtered_aggregated.asfreq('M')

# Define the SARIMA model with seasonality on the aggregated data
sarima_model = SARIMAX(df_filtered_aggregated['Total Sales'],
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, seasonality_period),
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_results = sarima_model.fit()

# Forecast the next 12 months
forecast = sarima_results.get_forecast(steps=12)
forecast_index = pd.date_range(df_filtered_aggregated.index[-1] + pd.Timedelta(days=1), periods=12, freq='M')

# Confidence intervals for the forecast
forecast_ci = forecast.conf_int()
forecast_ci.columns = ['Lower CI', 'Upper CI']

# Plotting the historical and forecasted data
plt.figure(figsize=(10, 6))
plt.plot(df_filtered_aggregated.index, df_filtered_aggregated['Total Sales'], label='Historical Sales')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecasted Sales', color='red')
plt.fill_between(forecast_index,
                 forecast_ci['Lower CI'],
                 forecast_ci['Upper CI'], color='pink', alpha=0.3)
plt.title('Sales Forecast with SARIMAX')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

summary_stats = df_filtered.describe(include=[np.number])
print(summary_stats)

model_performance = pd.DataFrame({
    'Model': ['Linear Regression', 'AdaBoost Regression', 'Logistic Regression', 'AdaBoost Classification'],
    'MSE': [mse_lin_reg, mse_ada_reg, np.nan, np.nan],
    'Accuracy': [np.nan, np.nan, acc_log_reg, acc_ada_clf]
})
print(model_performance)

sales_by_region = df_filtered.groupby('Region')['Total Sales'].sum().reset_index()
sales_by_region['Percentage'] = (sales_by_region['Total Sales'] / sales_by_region['Total Sales'].sum()) * 100
print(sales_by_region)

sales_over_time = df_filtered.resample('M')['Total Sales'].sum().reset_index()
print(sales_over_time)


# Calculate the total sales
total_sales = df_sales['Total Sales'].sum()

# Calculate the percentage of sales for each product
df_sales['Percentage of Sales'] = (df_sales['Total Sales'] / total_sales) * 100

print(df_sales)

top_selling_products = df_filtered.groupby('Product')['Total Sales'].sum().sort_values(ascending=False).reset_index()
print(top_selling_products.head(10))  # Adjust the number of top products as needed

# Example Data
sales_data = {
    'Product': ["Men's Street Footwear", "Women's Apparel", "Men's Athletic Footwear",
                "Women's Street Footwear", "Men's Apparel", "Women's Athletic Footwear"],
    'Total Sales': [208826244.0, 179038860.0, 153673680.0, 128002813.0, 123728632.0, 106631896.0]
}

df_sales = pd.DataFrame(sales_data)

price_analysis = df_filtered.groupby('Product')['Price per Unit'].mean().reset_index()
print(price_analysis)

# Example Data
price_data = {
    'Product': ["Men's Apparel", "Men's Athletic Footwear", "Men's Street Footwear",
                "Women's Apparel", "Women's Athletic Footwear", "Women's Street Footwear"],
    'Price per Unit': [50.321918, 43.779503, 44.236646, 51.600746, 41.112702, 40.252488]
}

df_price = pd.DataFrame(price_data)

# Format the price per unit as a dollar amount
df_price['Price per Unit'] = df_price['Price per Unit'].apply(lambda x: f"${x:.2f}")

print(df_price)



forecast_summary = pd.DataFrame({
    'Forecast Period': forecast_index,
    'Forecasted Sales': forecast.predicted_mean,
    'Lower CI': forecast_ci['Lower CI'],
    'Upper CI': forecast_ci['Upper CI']
})
print(forecast_summary)

# Assuming 'Invoice Date' is a datetime column
df_filtered['Time_Period'] = df_filtered.index.to_period('M')

# Now you can create the pivot table
sales_heatmap_data = df_filtered.pivot_table(index='Product', columns='Time_Period', values='Total Sales', aggfunc='sum')

# Increase the figure size and adjust label font size if necessary
plt.figure(figsize=(15, 10))  # Adjusted figure size
sns.heatmap(sales_heatmap_data, annot=False, cmap='viridis')
plt.title("Sales Heatmap by Product Over Time")
plt.ylabel("Product")
plt.xlabel("Time Period")
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()



# Pivot table for price per unit by product and region
price_heatmap_data = df_filtered.pivot_table(index='Product', columns='Region', values='Price per Unit', aggfunc='mean')

# Plotting the heatmap
plt.figure(figsize=(15,10))
sns.heatmap(price_heatmap_data, annot=True, fmt=".2f", cmap='viridis')
plt.title("Price per Unit Heatmap by Product and Region")
plt.ylabel("Product")
plt.xlabel("Region")
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()


# Pivot table for units sold by product and region
units_sold_heatmap_data = df_filtered.pivot_table(index='Product', columns='Region', values='Units Sold', aggfunc='sum')

# Plotting the heatmap
plt.figure(figsize=(15,10))
sns.heatmap(units_sold_heatmap_data, annot=True, fmt="d", cmap='viridis')
plt.title("Units Sold Heatmap by Product and Region")
plt.ylabel("Product")
plt.xlabel("Region")
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()


# Heatmap for Total Sales by Product and Region
product_region_sales = df_filtered.pivot_table(index='Product', columns='Region', values='Total Sales', aggfunc='sum')
plt.figure(figsize=(15,10))
sns.heatmap(product_region_sales, annot=True, fmt=".1f", cmap='viridis')
plt.title("Total Sales by Product and Region")
plt.ylabel("Product")
plt.xlabel("Region")
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()


# Pivot table for total sales by product and region
product_region_sales = df_filtered.pivot_table(index='Product', columns='Region', values='Total Sales', aggfunc='sum')


# Displaying Predictions
ada_reg_results = pd.DataFrame({'Actual': y_test_reg, 'Predicted': y_pred_ada_reg})
print(ada_reg_results.head())

log_reg_results = pd.DataFrame({'Actual': y_test_class, 'Predicted': y_pred_log_reg})
print(log_reg_results.head())

