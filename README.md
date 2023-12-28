# Predictive Analytics in E-commerce: Adidas Sales Dataset

This project delves into Adidas US sales data, leveraging statistical and machine learning techniques to unearth sales trends and patterns in the competitive sports apparel market. The aim is to derive actionable insights for sales strategy optimization and to forecast future trends through advanced data analysis.

## Description

This project's key aspects include:

- **Comprehensive Dataset Analysis:** Focused analysis of Adidas US sales data, covering product lines, sales figures, and regional distribution.
- **Rigorous Data Preprocessing:** Steps involve data cleaning, outlier removal (IQR and Z-score methods), and handling missing values, ensuring data integrity.
- **Diverse Analytical Models:** Implementation of Linear Regression and AdaBoost Regression for sales prediction, and Logistic Regression and AdaBoost Classification for sales segmentation.
- **In-depth Time Series Analysis:** Utilizing SARIMAX model to forecast future sales trends, accounting for seasonality and market dynamics.
- **Elaborate Visualizations:** Creation of various visual aids such as bar charts, pie charts, scatter plots, boxplots, histograms, and heatmaps to effectively represent and analyze sales data.
- **Strategic Business Insights:** The project aims to provide valuable insights to Adidas for refining sales strategies and anticipating market trends.

## Data Source

The project uses a detailed dataset from  [Kaggle](https://www.kaggle.com/datasets/heemalichaudhari/adidas-sales-dataset), encompassing extensive data on Adidas's product sales, including variables like total sales, units sold, prices, and operating profits, specifically focusing on the U.S. market.

## Table of Contents

- [Description](#description)
- [Data Source](#data-source)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Model Evaluation and Validation](#model-evaluation-and-validation)
- [Visualization](#visualization)
- [Time Series Analysis](#time-series-analysis)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Installation

```bash
git clone https://github.com/pclaridy/adidas-us-sales-predictive-analytics
cd adidas-us-sales-predictive-analytics
```

## Data Preprocessing

Data preprocessing includes:

- **Data Cleaning**: Removal of irrelevant columns and handling missing data.
- **Outlier Detection and Removal**: Employing IQR and Z-score methods.
- **Feature Engineering**: Calculating the percentage contribution of each product to total sales and standardizing 'Price per Unit' into a readable format.

## Modeling

The project employs:

- **Regression Models**: Linear Regression and AdaBoost Regression for continuous sales prediction.
- **Classification Models**: Logistic Regression and AdaBoost Classification for categorizing sales into "low", "medium", and "high" segments.

## Model Evaluation and Validation

Model evaluation details:

- **Linear Regression**: Reported an MSE of approximately 834.2 million.
- **AdaBoost Regression**: Demonstrated an improved MSE of approximately 566.9 million, indicating more effective pattern capture.
- **Logistic Regression**: Achieved a high accuracy score of 96.2%, demonstrating robustness in categorizing sales.
- **AdaBoost Classification**: Followed with an accuracy score of 68.6%, highlighting its capability in sales segmentation.

## Visualization

Visualizations include:

- Various charts such as bar, pie, scatter, boxplot, histogram, and heatmap to depict different aspects of sales data.
- These visual tools are crucial for interpreting data and deriving strategic insights.

## Time Series Analysis

- **SARIMAX Model**: Used for forecasting future sales trends.
- Analysis includes an Augmented Dickey-Fuller test with an ADF statistic of -5.007 and a p-value of approximately 0.000021, indicating strong evidence against the null hypothesis of non-stationarity.

## Conclusion and Future Work

The project offers valuable insights into sales trends and drivers within the Adidas U.S. market. Future work involves expanding the dataset, exploring additional predictive models, and incorporating external factors for more precise sales predictions.
