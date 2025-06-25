"""
Advanced Pandas Operations
=========================

This script demonstrates advanced Pandas operations for data manipulation,
including complex data transformations, time series analysis, and advanced
grouping operations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("📊 Advanced Pandas Operations Demonstration")
print("=" * 50)

# ============================================================================
# 1. Advanced DataFrame Creation and Manipulation
# ============================================================================
print("\n1. Advanced DataFrame Creation")
print("-" * 32)

# Create multi-level sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365, freq='D')
stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
products = ['Product_1', 'Product_2', 'Product_3', 'Product_4', 'Product_5']

# Create hierarchical index
multi_index = pd.MultiIndex.from_product([stores, products], names=['Store', 'Product'])

# Generate sample sales data
n_records = len(multi_index) * 50  # ~4,000 records
data = []

for _ in range(50):  # 50 time periods
    date = np.random.choice(dates)
    for store in stores:
        for product in products:
            record = {
                'Date': date,
                'Store': store,
                'Product': product,
                'Sales': np.random.normal(1000, 200),
                'Units_Sold': np.random.randint(10, 100),
                'Cost': np.random.normal(500, 100),
                'Customer_Count': np.random.randint(50, 200),
                'Promotion': np.random.choice([True, False], p=[0.3, 0.7])
            }
            data.append(record)

df = pd.DataFrame(data)
df['Profit'] = df['Sales'] - df['Cost']
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100

print(f"Created dataset with {len(df)} records")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# ============================================================================
# 2. Advanced Indexing and Selection
# ============================================================================
print("\n\n2. Advanced Indexing and Selection")
print("-" * 36)

# Set multi-level index
df_multi = df.set_index(['Store', 'Product', 'Date']).sort_index()
print(f"Multi-index DataFrame shape: {df_multi.shape}")
print(f"Index levels: {df_multi.index.names}")

# Advanced selection examples
store_a_data = df_multi.loc['Store_A']
product_1_all_stores = df_multi.xs('Product_1', level='Product')
specific_selection = df_multi.loc[('Store_A', 'Product_1'), :]

print(f"Store A data shape: {store_a_data.shape}")
print(f"Product 1 across all stores shape: {product_1_all_stores.shape}")
print(f"Specific selection shape: {specific_selection.shape}")

# Query method for complex filtering
high_profit_sales = df.query('Profit > 600 and Units_Sold > 50')
promotional_sales = df.query('Promotion == True and Sales > @df.Sales.quantile(0.75)')

print(f"High profit sales: {len(high_profit_sales)} records")
print(f"High-value promotional sales: {len(promotional_sales)} records")

# ============================================================================
# 3. Advanced Grouping and Aggregation
# ============================================================================
print("\n\n3. Advanced Grouping and Aggregation")
print("-" * 37)

# Complex grouping operations
store_product_summary = df.groupby(['Store', 'Product']).agg({
    'Sales': ['sum', 'mean', 'std', 'count'],
    'Profit': ['sum', 'mean'],
    'Units_Sold': 'sum',
    'Customer_Count': 'mean',
    'Promotion': lambda x: (x == True).sum()  # Count promotions
}).round(2)

# Flatten column names
store_product_summary.columns = ['_'.join(col).strip() for col in store_product_summary.columns]
print("Store-Product summary shape:", store_product_summary.shape)
print("Summary columns:", list(store_product_summary.columns))

# Custom aggregation functions
def profit_efficiency(group):
    return group['Profit'].sum() / group['Units_Sold'].sum()

def sales_volatility(group):
    return group['Sales'].std() / group['Sales'].mean()

custom_metrics = df.groupby(['Store']).apply(lambda x: pd.Series({
    'Profit_per_Unit': profit_efficiency(x),
    'Sales_Volatility': sales_volatility(x),
    'Avg_Customer_Count': x['Customer_Count'].mean(),
    'Promotion_Rate': (x['Promotion'] == True).sum() / len(x)
})).round(3)

print("\nCustom metrics by store:")
print(custom_metrics)

# Transform operations
df['Sales_Store_Mean'] = df.groupby('Store')['Sales'].transform('mean')
df['Sales_Deviation'] = df['Sales'] - df['Sales_Store_Mean']
df['Rolling_Sales_Mean'] = df.groupby(['Store', 'Product'])['Sales'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

print(f"Transform operations completed")

# ============================================================================
# 4. Time Series Operations
# ============================================================================
print("\n\n4. Time Series Operations")
print("-" * 27)

# Create time series focused dataset
df_ts = df.copy()
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
df_ts = df_ts.set_index('Date').sort_index()

# Resampling operations
daily_sales = df_ts.groupby(['Store'])['Sales'].resample('D').sum()
weekly_sales = df_ts.groupby(['Store'])['Sales'].resample('W').sum()
monthly_sales = df_ts.groupby(['Store'])['Sales'].resample('M').sum()

print(f"Daily sales data points: {len(daily_sales)}")
print(f"Weekly sales data points: {len(weekly_sales)}")
print(f"Monthly sales data points: {len(monthly_sales)}")

# Time-based grouping
df_ts['Year'] = df_ts.index.year
df_ts['Month'] = df_ts.index.month
df_ts['Quarter'] = df_ts.index.quarter
df_ts['DayOfWeek'] = df_ts.index.day_name()

seasonal_analysis = df_ts.groupby(['Store', 'Quarter'])['Sales'].agg(['sum', 'mean', 'count'])
print("\nSeasonal analysis shape:", seasonal_analysis.shape)

# Rolling operations
window_size = 7
df_ts_reset = df_ts.reset_index()
df_ts_reset = df_ts_reset.sort_values(['Store', 'Product', 'Date'])

df_ts_reset['Sales_7day_MA'] = df_ts_reset.groupby(['Store', 'Product'])['Sales'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

df_ts_reset['Sales_7day_Std'] = df_ts_reset.groupby(['Store', 'Product'])['Sales'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).std()
)

print(f"Rolling averages calculated with window size: {window_size}")

# ============================================================================
# 5. Advanced Data Cleaning and Transformation
# ============================================================================
print("\n\n5. Data Cleaning and Transformation")
print("-" * 37)

# Create a dataset with quality issues
df_dirty = df.copy()
np.random.seed(123)

# Introduce missing values
missing_indices = np.random.choice(df_dirty.index, size=100, replace=False)
df_dirty.loc[missing_indices[:50], 'Sales'] = np.nan
df_dirty.loc[missing_indices[50:], 'Cost'] = np.nan

# Introduce duplicates
df_dirty = pd.concat([df_dirty, df_dirty.sample(20)], ignore_index=True)

# Introduce outliers
outlier_indices = np.random.choice(df_dirty.index, size=30, replace=False)
df_dirty.loc[outlier_indices, 'Sales'] = df_dirty.loc[outlier_indices, 'Sales'] * 5

print("Data quality issues introduced:")
print(f"Missing values:\n{df_dirty.isnull().sum()}")
print(f"Duplicates: {df_dirty.duplicated().sum()}")
print(f"Dataset shape: {df_dirty.shape}")

# Advanced cleaning operations
df_clean = df_dirty.copy()

# Handle missing values with sophisticated methods
# Forward fill within groups
df_clean['Sales'] = df_clean.groupby(['Store', 'Product'])['Sales'].fillna(method='ffill')
df_clean['Sales'] = df_clean.groupby(['Store', 'Product'])['Sales'].fillna(method='bfill')

# Fill remaining with group median
df_clean['Sales'] = df_clean.groupby(['Store', 'Product'])['Sales'].transform(
    lambda x: x.fillna(x.median())
)

# Similar approach for Cost
df_clean['Cost'] = df_clean.groupby(['Store', 'Product'])['Cost'].fillna(method='ffill')
df_clean['Cost'] = df_clean.groupby(['Store', 'Product'])['Cost'].fillna(method='bfill')
df_clean['Cost'] = df_clean.groupby(['Store', 'Product'])['Cost'].transform(
    lambda x: x.fillna(x.median())
)

# Remove duplicates with sophisticated logic
df_clean = df_clean.drop_duplicates(subset=['Store', 'Product', 'Date'], keep='first')

# Handle outliers using group-specific IQR
def remove_outliers_iqr(group, column):
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing
    group[column] = group[column].clip(lower_bound, upper_bound)
    return group

df_clean = df_clean.groupby(['Store', 'Product']).apply(
    lambda x: remove_outliers_iqr(x, 'Sales')
).reset_index(drop=True)

# Recalculate derived columns
df_clean['Profit'] = df_clean['Sales'] - df_clean['Cost']
df_clean['Profit_Margin'] = (df_clean['Profit'] / df_clean['Sales']) * 100

print(f"\nAfter cleaning:")
print(f"Missing values:\n{df_clean.isnull().sum()}")
print(f"Final shape: {df_clean.shape}")

# ============================================================================
# 6. Advanced Pivot Tables and Reshaping
# ============================================================================
print("\n\n6. Pivot Tables and Reshaping")
print("-" * 31)

# Complex pivot table
pivot_complex = pd.pivot_table(
    df_clean,
    values=['Sales', 'Profit', 'Units_Sold'],
    index=['Store'],
    columns=['Product'],
    aggfunc={
        'Sales': 'sum',
        'Profit': 'mean',
        'Units_Sold': 'sum'
    },
    fill_value=0,
    margins=True
)

print("Complex pivot table shape:", pivot_complex.shape)
print("Pivot table levels:", pivot_complex.columns.nlevels)

# Melting and reshaping
df_pivot = df_clean.pivot_table(
    values='Sales',
    index=['Store', 'Date'],
    columns='Product',
    aggfunc='sum',
    fill_value=0
)

# Melt back to long format
df_melted = df_pivot.reset_index().melt(
    id_vars=['Store', 'Date'],
    var_name='Product',
    value_name='Sales'
)

print(f"Original pivot shape: {df_pivot.shape}")
print(f"Melted shape: {df_melted.shape}")

# Cross-tabulation
crosstab = pd.crosstab(
    df_clean['Store'],
    df_clean['Promotion'],
    values=df_clean['Sales'],
    aggfunc='sum',
    margins=True
)

print("Cross-tabulation shape:", crosstab.shape)

# ============================================================================
# 7. Advanced String Operations
# ============================================================================
print("\n\n7. String Operations")
print("-" * 21)

# Create text data for string operations
df_text = pd.DataFrame({
    'Customer_ID': [f'CUST_{i:05d}' for i in range(1000)],
    'Email': [f'customer_{i}@{"gmail.com" if i%2==0 else "yahoo.com"}' for i in range(1000)],
    'Phone': [f'({np.random.randint(200,999)}) {np.random.randint(200,999)}-{np.random.randint(1000,9999)}' for _ in range(1000)],
    'Address': [f'{np.random.randint(1,9999)} {"Main St" if i%3==0 else "Oak Ave" if i%3==1 else "Park Blvd"}' for i in range(1000)]
})

# String operations
df_text['Email_Domain'] = df_text['Email'].str.split('@').str[1]
df_text['Area_Code'] = df_text['Phone'].str.extract(r'\((\d{3})\)')
df_text['Street_Type'] = df_text['Address'].str.extract(r'(\w+)$')
df_text['Customer_Number'] = df_text['Customer_ID'].str.extract(r'(\d+)').astype(int)

# String filtering and validation
gmail_customers = df_text[df_text['Email'].str.contains('gmail')]
main_st_customers = df_text[df_text['Address'].str.contains('Main St')]

print(f"Gmail customers: {len(gmail_customers)}")
print(f"Main St customers: {len(main_st_customers)}")
print(f"Unique domains: {df_text['Email_Domain'].nunique()}")
print(f"Unique area codes: {df_text['Area_Code'].nunique()}")

# ============================================================================
# 8. Advanced Merging and Joining
# ============================================================================
print("\n\n8. Merging and Joining")
print("-" * 23)

# Create related datasets
products_df = pd.DataFrame({
    'Product': products,
    'Category': ['Electronics', 'Electronics', 'Home', 'Fashion', 'Sports'],
    'Launch_Date': pd.date_range('2020-01-01', periods=5, freq='3M'),
    'Supplier': ['Supplier_A', 'Supplier_B', 'Supplier_A', 'Supplier_C', 'Supplier_B']
})

stores_df = pd.DataFrame({
    'Store': stores,
    'Region': ['North', 'South', 'East', 'West'],
    'Manager': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Opening_Date': pd.date_range('2019-01-01', periods=4, freq='6M')
})

# Various merge operations
merged_full = df_clean.merge(products_df, on='Product', how='left')
merged_full = merged_full.merge(stores_df, on='Store', how='left')

print(f"Original data shape: {df_clean.shape}")
print(f"After merging with products and stores: {merged_full.shape}")

# Advanced merge with indicator
merge_indicator = df_clean.merge(
    products_df, 
    on='Product', 
    how='outer', 
    indicator=True
)

print(f"Merge indicator results:")
print(merge_indicator['_merge'].value_counts())

# Concatenation operations
df_2024 = df_clean.copy()
df_2024['Date'] = df_2024['Date'] + pd.DateOffset(years=1)

combined_years = pd.concat([df_clean, df_2024], keys=['2023', '2024'])
print(f"Combined dataset shape: {combined_years.shape}")

# ============================================================================
# 9. Performance Optimization
# ============================================================================
print("\n\n9. Performance Optimization")
print("-" * 30)

# Data type optimization
def optimize_dtypes(df):
    """Optimize DataFrame memory usage by converting data types."""
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
        else:
            if df[col].min() > -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() > -32768 and df[col].max() < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Convert to categorical where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum()
    reduction = (original_memory - optimized_memory) / original_memory * 100
    
    return df, original_memory, optimized_memory, reduction

df_optimized, orig_mem, opt_mem, reduction = optimize_dtypes(df_clean.copy())

print(f"Memory optimization results:")
print(f"Original memory: {orig_mem / 1024 / 1024:.2f} MB")
print(f"Optimized memory: {opt_mem / 1024 / 1024:.2f} MB")
print(f"Reduction: {reduction:.1f}%")

# Chunking for large datasets
def process_in_chunks(df, chunk_size=1000):
    """Process DataFrame in chunks for memory efficiency."""
    results = []
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]
        
        # Example processing: calculate running statistics
        chunk_result = {
            'chunk_start': start,
            'chunk_end': end,
            'mean_sales': chunk['Sales'].mean(),
            'total_profit': chunk['Profit'].sum()
        }
        results.append(chunk_result)
    
    return pd.DataFrame(results)

chunk_results = process_in_chunks(df_clean, chunk_size=500)
print(f"Processed {len(df_clean)} records in {len(chunk_results)} chunks")

print("\n" + "=" * 50)
print("Advanced Pandas Operations Complete!")
print("=" * 50)

print(f"\n📋 Pandas Operations Covered:")
print(f"✅ Multi-index DataFrames and advanced indexing")
print(f"✅ Complex grouping and aggregation operations")
print(f"✅ Time series analysis and resampling")
print(f"✅ Advanced data cleaning techniques")
print(f"✅ Pivot tables and data reshaping")
print(f"✅ String operations and text processing")
print(f"✅ Advanced merging and joining")
print(f"✅ Performance optimization strategies")
print(f"✅ Memory-efficient data processing")

# Save results
output_dir = 'data'
df_clean.to_csv(f'{output_dir}/advanced_cleaned_data.csv', index=False)
store_product_summary.to_csv(f'{output_dir}/store_product_summary.csv')
pivot_complex.to_csv(f'{output_dir}/pivot_analysis.csv')

print(f"\n💾 Analysis results saved to {output_dir}/ directory")
