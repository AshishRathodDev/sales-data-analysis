
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

def create_sample_data():
    # Create sample data
    np.random.seed(42)
    
    # Generate dates for one year
    dates = [datetime(2023, 1, 1) + timedelta(days=x) for x in range(365)]
    
    # Generate 1000 sales records
    data = {
        'date': np.random.choice(dates, 1000),
        'customer_id': np.random.randint(1, 101, 1000),  # 100 customers
        'product_id': np.random.randint(1, 51, 1000),    # 50 products
        'amount': np.random.uniform(10, 1000, 1000).round(2)
    }
    
    return pd.DataFrame(data)

def sales_analysis(df):
    # Monthly sales trends
    monthly_sales = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()

    # Customer segmentation
    customer_features = df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count'],
        'product_id': 'nunique'
    })

    # Normalize features for clustering
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(customer_features)

    # Customer segmentation using K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_features['segment'] = kmeans.fit_predict(features_normalized)

    return monthly_sales, customer_features

def create_visualizations(monthly_sales, customer_features, df):
    # Set the style
    sns.set_style("whitegrid")
    
    # Create sales trend plot
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Monthly Sales Trend (2023)')
    plt.xlabel('Month')
    plt.ylabel('Sales Amount ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sales_trend.png')
    plt.close()

    # Create customer segmentation plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_features.reset_index(),
        x=('amount', 'sum'),
        y=('amount', 'count'),
        hue='segment',
        palette='deep'
    )
    plt.title('Customer Segments')
    plt.xlabel('Total Purchase Amount ($)')
    plt.ylabel('Number of Purchases')
    plt.tight_layout()
    plt.savefig('customer_segments.png')
    plt.close()

    # Print analysis summary
    print("\nAnalysis Summary:")
    print("-----------------")
    print(f"Total Sales: ${df['amount'].sum():,.2f}")
    print(f"Number of Customers: {len(customer_features)}")
    print(f"Average Order Value: ${df['amount'].mean():,.2f}")
    print("\nCustomer Segments:")
    for segment in range(3):
        segment_size = (customer_features['segment'] == segment).sum()
        segment_percent = (segment_size / len(customer_features)) * 100
        print(f"Segment {segment}: {segment_size} customers ({segment_percent:.1f}%)")

def main():
    # Create and analyze sample data
    df = create_sample_data()
    monthly_sales, customer_features = sales_analysis(df)
    create_visualizations(monthly_sales, customer_features, df)

if __name__ == "__main__":
    main()
