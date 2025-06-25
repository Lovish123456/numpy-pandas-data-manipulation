"""
NumPy and Pandas Data Manipulation Project
==========================================

This project demonstrates essential data manipulation techniques using NumPy and Pandas.
It covers array operations, data cleaning, analysis, and visualization preparation.

Author: Devansh Tomar
Date: 2025-06-24
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("NumPy and Pandas Data Manipulation Demonstration")
print("=" * 60)

# ============================================================================
# PART 1: NumPy Fundamentals
# ============================================================================
print("\n🔢 PART 1: NumPy Array Operations")
print("-" * 40)

# 1.1 Array Creation and Basic Operations
print("1.1 Array Creation and Basic Operations:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr3 = np.random.randn(1000)  # Random normal distribution

print(f"1D Array: {arr1}")
print(f"2D Array:\n{arr2}")
print(f"Random array shape: {arr3.shape}")
print(f"Random array mean: {arr3.mean():.3f}")
print(f"Random array std: {arr3.std():.3f}")

# 1.2 Array Manipulation
print("\n1.2 Array Manipulation:")
reshaped = arr1.reshape(5, 1)
transposed = arr2.T
print(f"Reshaped array (5x1):\n{reshaped}")
print(f"Transposed 2D array:\n{transposed}")

# 1.3 Mathematical Operations
print("\n1.3 Mathematical Operations:")
print(f"Element-wise operations: {arr1 * 2}")
print(f"Matrix multiplication result shape: {np.dot(arr2, transposed).shape}")

# 1.4 Statistical Operations
print("\n1.4 Statistical Operations:")
data = np.random.normal(100, 15, 1000)  # Mean=100, std=15, 1000 samples
print(f"Generated data statistics:")
print(f"  Mean: {np.mean(data):.2f}")
print(f"  Median: {np.median(data):.2f}")
print(f"  Standard Deviation: {np.std(data):.2f}")
print(f"  Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")

# ============================================================================
# PART 2: Pandas DataFrames and Series
# ============================================================================
print("\n\n📊 PART 2: Pandas DataFrame Operations")
print("-" * 40)

# 2.1 Create Sample Dataset
print("2.1 Creating Sample Dataset:")
np.random.seed(42)  # For reproducible results

# Generate sample e-commerce data
n_records = 1000
dates = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_records)
products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Monitor', 'Keyboard', 'Mouse']
regions = ['North', 'South', 'East', 'West', 'Central']
sales_reps = [f'Rep_{i:03d}' for i in range(1, 21)]

df = pd.DataFrame({
    'date': dates,
    'product': np.random.choice(products, n_records),
    'region': np.random.choice(regions, n_records),
    'sales_rep': np.random.choice(sales_reps, n_records),
    'quantity': np.random.randint(1, 20, n_records),
    'unit_price': np.random.uniform(50, 2000, n_records),
    'customer_rating': np.random.uniform(1, 5, n_records)
})

# Calculate derived columns
df['total_sales'] = df['quantity'] * df['unit_price']
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_week'] = df['date'].dt.day_name()

print(f"Dataset created with {len(df)} records and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# 2.2 Data Exploration
print("\n\n2.2 Data Exploration:")
print("Dataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"\nData types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

# 2.3 Data Filtering and Selection
print("\n\n2.3 Data Filtering and Selection:")

# Filter high-value sales
high_value_sales = df[df['total_sales'] > df['total_sales'].quantile(0.9)]
print(f"High-value sales (top 10%): {len(high_value_sales)} records")

# Filter by multiple conditions
laptop_sales = df[(df['product'] == 'Laptop') & (df['quantity'] > 5)]
print(f"Laptop sales with quantity > 5: {len(laptop_sales)} records")

# Filter by date range
recent_sales = df[df['date'] > '2024-06-01']
print(f"Sales after June 2024: {len(recent_sales)} records")

# 2.4 Grouping and Aggregation
print("\n\n2.4 Grouping and Aggregation:")

# Group by product
product_summary = df.groupby('product').agg({
    'total_sales': ['sum', 'mean', 'count'],
    'quantity': 'sum',
    'customer_rating': 'mean'
}).round(2)
print("Sales by Product:")
print(product_summary)

# Group by region and month
regional_monthly = df.groupby(['region', 'month'])['total_sales'].sum().unstack(fill_value=0)
print(f"\nRegional monthly sales shape: {regional_monthly.shape}")

# Top performing sales reps
top_reps = df.groupby('sales_rep')['total_sales'].sum().sort_values(ascending=False).head()
print(f"\nTop 5 Sales Representatives:")
print(top_reps)

# 2.5 Data Transformation
print("\n\n2.5 Data Transformation:")

# Create categorical variables
df['price_category'] = pd.cut(df['unit_price'], 
                             bins=[0, 200, 500, 1000, 2000], 
                             labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])

df['rating_category'] = pd.cut(df['customer_rating'], 
                              bins=[0, 2, 3, 4, 5], 
                              labels=['Poor', 'Fair', 'Good', 'Excellent'])

print("Price categories distribution:")
print(df['price_category'].value_counts())

print("\nRating categories distribution:")
print(df['rating_category'].value_counts())

# 2.6 Time Series Operations
print("\n\n2.6 Time Series Operations:")

# Set date as index for time series operations
df_ts = df.set_index('date')
daily_sales = df_ts['total_sales'].resample('D').sum()
weekly_sales = df_ts['total_sales'].resample('W').sum()
monthly_sales = df_ts['total_sales'].resample('M').sum()

print(f"Daily sales data points: {len(daily_sales)}")
print(f"Weekly sales data points: {len(weekly_sales)}")
print(f"Monthly sales data points: {len(monthly_sales)}")

print(f"\nMonthly sales summary:")
print(monthly_sales.describe())

# 2.7 Advanced Operations
print("\n\n2.7 Advanced DataFrame Operations:")

# Pivot tables
pivot_table = df.pivot_table(
    values='total_sales',
    index='product',
    columns='region',
    aggfunc='sum',
    fill_value=0
)
print("Pivot Table - Sales by Product and Region:")
print(pivot_table)

# Rolling averages
df_sorted = df.sort_values('date')
df_sorted['sales_7day_avg'] = df_sorted['total_sales'].rolling(window=7).mean()
df_sorted['sales_30day_avg'] = df_sorted['total_sales'].rolling(window=30).mean()

print(f"\nRolling averages calculated for {len(df_sorted)} records")

# Correlation analysis
numeric_cols = ['quantity', 'unit_price', 'total_sales', 'customer_rating']
correlation_matrix = df[numeric_cols].corr()
print(f"\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# ============================================================================
# PART 3: Data Cleaning and Quality
# ============================================================================
print("\n\n🧹 PART 3: Data Cleaning Operations")
print("-" * 40)

# Create a copy with some data quality issues
df_dirty = df.copy()

# Introduce some data quality issues
np.random.seed(123)
dirty_indices = np.random.choice(df_dirty.index, size=50, replace=False)

# Add missing values
df_dirty.loc[dirty_indices[:20], 'customer_rating'] = np.nan
df_dirty.loc[dirty_indices[20:30], 'unit_price'] = np.nan

# Add duplicates
df_dirty = pd.concat([df_dirty, df_dirty.iloc[:10]], ignore_index=True)

# Add outliers
outlier_indices = np.random.choice(df_dirty.index, size=10, replace=False)
df_dirty.loc[outlier_indices, 'unit_price'] = df_dirty.loc[outlier_indices, 'unit_price'] * 10

print("3.1 Data Quality Issues Introduced:")
print(f"Missing values per column:")
print(df_dirty.isnull().sum())
print(f"Total duplicates: {df_dirty.duplicated().sum()}")
print(f"Dataset shape after issues: {df_dirty.shape}")

# Data cleaning operations
print("\n3.2 Data Cleaning Operations:")

# Handle missing values
df_clean = df_dirty.copy()

# Fill missing ratings with median
df_clean['customer_rating'].fillna(df_clean['customer_rating'].median(), inplace=True)

# Fill missing prices with product-specific median
df_clean['unit_price'] = df_clean.groupby('product')['unit_price'].transform(
    lambda x: x.fillna(x.median())
)

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Handle outliers using IQR method
Q1 = df_clean['unit_price'].quantile(0.25)
Q3 = df_clean['unit_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['unit_price'] < lower_bound) | 
                   (df_clean['unit_price'] > upper_bound)]
print(f"Outliers detected: {len(outliers)}")

# Cap outliers instead of removing them
df_clean.loc[df_clean['unit_price'] > upper_bound, 'unit_price'] = upper_bound
df_clean.loc[df_clean['unit_price'] < lower_bound, 'unit_price'] = lower_bound

# Recalculate derived columns
df_clean['total_sales'] = df_clean['quantity'] * df_clean['unit_price']

print(f"Clean dataset shape: {df_clean.shape}")
print(f"Missing values after cleaning:")
print(df_clean.isnull().sum())

# ============================================================================
# PART 4: Analysis and Insights
# ============================================================================
print("\n\n📈 PART 4: Data Analysis and Insights")
print("-" * 40)

print("4.1 Business Insights:")

# Total sales analysis
total_revenue = df_clean['total_sales'].sum()
avg_order_value = df_clean['total_sales'].mean()
total_orders = len(df_clean)

print(f"Overall Performance Metrics:")
print(f"  Total Revenue: ${total_revenue:,.2f}")
print(f"  Average Order Value: ${avg_order_value:.2f}")
print(f"  Total Orders: {total_orders:,}")

# Product performance
product_performance = df_clean.groupby('product').agg({
    'total_sales': 'sum',
    'quantity': 'sum',
    'customer_rating': 'mean'
}).sort_values('total_sales', ascending=False)

print(f"\nTop Products by Revenue:")
for product, data in product_performance.head().iterrows():
    print(f"  {product}: ${data['total_sales']:,.2f} (Rating: {data['customer_rating']:.2f})")

# Regional analysis
regional_performance = df_clean.groupby('region')['total_sales'].sum().sort_values(ascending=False)
print(f"\nRegional Performance:")
for region, sales in regional_performance.items():
    percentage = (sales / total_revenue) * 100
    print(f"  {region}: ${sales:,.2f} ({percentage:.1f}%)")

# Seasonal analysis
seasonal_analysis = df_clean.groupby('quarter')['total_sales'].sum()
print(f"\nSeasonal Analysis (by Quarter):")
for quarter, sales in seasonal_analysis.items():
    print(f"  Q{quarter}: ${sales:,.2f}")

# Customer satisfaction analysis
satisfaction_by_product = df_clean.groupby('product')['customer_rating'].mean().sort_values(ascending=False)
print(f"\nCustomer Satisfaction by Product:")
for product, rating in satisfaction_by_product.items():
    print(f"  {product}: {rating:.2f}/5.0")

print("\n" + "=" * 60)
print("Data Manipulation Demonstration Complete!")
print("=" * 60)

# Save cleaned dataset
output_file = 'data/cleaned_sales_data.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n💾 Cleaned dataset saved to: {output_file}")

print(f"\n📋 Summary of Operations Performed:")
print(f"✅ NumPy: Array operations, mathematical computations, statistical analysis")
print(f"✅ Pandas: DataFrame creation, filtering, grouping, aggregation")
print(f"✅ Data Cleaning: Missing values, duplicates, outliers")
print(f"✅ Time Series: Resampling, rolling averages")
print(f"✅ Advanced: Pivot tables, correlation analysis")
print(f"✅ Business Analysis: KPIs, trends, insights")
