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
df = pd.read_excel("../../data/raw/Adidas US Sales Datasets.xlsx")
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

