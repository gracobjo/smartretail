"""
Data generator for sales analytics dashboard.
Generates realistic sales and customer data for demonstration purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataGenerator:
    """Generator for realistic sales and customer data."""
    
    def __init__(self, seed=42):
        """Initialize data generator with seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Product categories and prices
        self.product_categories = {
            'Electronics': {'min_price': 50, 'max_price': 2000, 'frequency': 0.3},
            'Clothing': {'min_price': 20, 'max_price': 200, 'frequency': 0.25},
            'Home & Garden': {'min_price': 30, 'max_price': 500, 'frequency': 0.2},
            'Books': {'min_price': 10, 'max_price': 50, 'frequency': 0.15},
            'Sports': {'min_price': 40, 'max_price': 300, 'frequency': 0.1}
        }
        
        # Customer segments
        self.customer_segments = {
            'Premium': {'avg_order_value': 500, 'frequency': 0.1, 'retention': 0.9},
            'Regular': {'avg_order_value': 150, 'frequency': 0.3, 'retention': 0.7},
            'Occasional': {'avg_order_value': 80, 'frequency': 0.4, 'retention': 0.5},
            'New': {'avg_order_value': 50, 'frequency': 0.2, 'retention': 0.3}
        }
        
        # Sales channels
        self.sales_channels = ['Online', 'Retail', 'Mobile App', 'Social Media']
        
        # Geographic regions
        self.regions = ['North', 'South', 'East', 'West', 'Central']
        
    def generate_customer_data(self, n_customers: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate customer data with realistic patterns.
        
        Args:
            n_customers: Number of customers to generate
            start_date: Start date for customer acquisition
            end_date: End date for analysis
            
        Returns:
            DataFrame with customer information
        """
        print(f"Generating data for {n_customers} customers...")
        
        customer_data = []
        
        for i in range(n_customers):
            # Generate customer acquisition date
            acquisition_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            
            # Assign customer segment
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=[self.customer_segments[s]['frequency'] for s in self.customer_segments.keys()]
            )
            
            # Generate customer attributes
            customer = {
                'customer_id': f'CUST_{i:06d}',
                'acquisition_date': acquisition_date,
                'segment': segment,
                'region': random.choice(self.regions),
                'sales_channel': random.choice(self.sales_channels),
                'age': random.randint(18, 75),
                'gender': random.choice(['M', 'F']),
                'is_active': True,
                'last_purchase_date': None,
                'total_purchases': 0,
                'total_revenue': 0.0
            }
            
            customer_data.append(customer)
        
        return pd.DataFrame(customer_data)
    
    def generate_sales_data(self, customers_df: pd.DataFrame, months: int) -> pd.DataFrame:
        """
        Generate sales transactions for customers.
        
        Args:
            customers_df: Customer data
            months: Number of months to generate data for
            
        Returns:
            DataFrame with sales transactions
        """
        print("Generating sales transactions...")
        
        sales_data = []
        
        for _, customer in customers_df.iterrows():
            # Get customer segment characteristics
            segment_info = self.customer_segments[customer['segment']]
            
            # Calculate purchase frequency (purchases per month)
            avg_frequency = segment_info['frequency']
            
            # Generate purchase dates
            start_date = customer['acquisition_date']
            end_date = start_date + timedelta(days=months * 30)
            
            current_date = start_date
            while current_date <= end_date:
                # Determine if customer makes a purchase
                if np.random.random() < avg_frequency:
                    # Generate purchase details
                    purchase = self._generate_purchase(customer, current_date)
                    sales_data.append(purchase)
                
                # Move to next month
                current_date += timedelta(days=30)
        
        return pd.DataFrame(sales_data)
    
    def _generate_purchase(self, customer: pd.Series, purchase_date: datetime) -> Dict:
        """
        Generate a single purchase transaction.
        
        Args:
            customer: Customer information
            purchase_date: Date of purchase
            
        Returns:
            Dictionary with purchase details
        """
        # Select product category
        category = np.random.choice(
            list(self.product_categories.keys()),
            p=[self.product_categories[c]['frequency'] for c in self.product_categories.keys()]
        )
        
        # Generate product price
        price_range = self.product_categories[category]
        price = np.random.uniform(price_range['min_price'], price_range['max_price'])
        
        # Apply segment-based adjustments
        segment_info = self.customer_segments[customer['segment']]
        price *= segment_info['avg_order_value'] / 150  # Normalize to regular segment
        
        # Add some randomness
        price *= np.random.uniform(0.8, 1.2)
        
        # Generate quantity
        quantity = np.random.poisson(1.5) + 1  # At least 1 item
        
        # Calculate total
        total = price * quantity
        
        return {
            'transaction_id': f'TXN_{random.randint(10000000, 99999999)}',
            'customer_id': customer['customer_id'],
            'purchase_date': purchase_date,
            'product_category': category,
            'product_name': f'{category}_Product_{random.randint(1, 100)}',
            'unit_price': round(price, 2),
            'quantity': quantity,
            'total_amount': round(total, 2),
            'sales_channel': customer['sales_channel'],
            'region': customer['region'],
            'customer_segment': customer['segment']
        }
    
    def generate_complete_dataset(self, n_customers: int = 10000, months: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with customers and sales.
        
        Args:
            n_customers: Number of customers
            months: Number of months of data
            
        Returns:
            Tuple of (customers_df, sales_df)
        """
        print(f"Generating complete dataset: {n_customers} customers, {months} months")
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)
        
        # Generate customer data
        customers_df = self.generate_customer_data(n_customers, start_date, end_date)
        
        # Generate sales data
        sales_df = self.generate_sales_data(customers_df, months)
        
        # Update customer summary statistics
        customers_df = self._update_customer_summaries(customers_df, sales_df)
        
        print(f"Generated {len(customers_df)} customers with {len(sales_df)} transactions")
        
        return customers_df, sales_df
    
    def _update_customer_summaries(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update customer data with summary statistics from sales.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            Updated customer data
        """
        # Calculate customer summaries
        customer_summaries = sales_df.groupby('customer_id').agg({
            'total_amount': ['sum', 'count'],
            'purchase_date': 'max'
        }).reset_index()
        
        customer_summaries.columns = ['customer_id', 'total_revenue', 'total_purchases', 'last_purchase_date']
        
        # Merge with customer data
        customers_df = customers_df.merge(customer_summaries, on='customer_id', how='left')
        
        # Fill missing values
        customers_df['total_revenue'] = customers_df['total_revenue'].fillna(0)
        customers_df['total_purchases'] = customers_df['total_purchases'].fillna(0)
        customers_df['last_purchase_date'] = customers_df['last_purchase_date'].fillna(customers_df['acquisition_date'])
        
        # Update active status based on recent purchases
        cutoff_date = datetime.now() - timedelta(days=90)
        customers_df['is_active'] = customers_df['last_purchase_date'] >= cutoff_date
        
        return customers_df
    
    def generate_churn_data(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn analysis data.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            DataFrame with churn analysis
        """
        print("Generating churn analysis data...")
        
        churn_data = []
        
        for _, customer in customers_df.iterrows():
            # Get customer's purchase history
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                continue
            
            # Calculate churn metrics
            first_purchase = customer_sales['purchase_date'].min()
            last_purchase = customer_sales['purchase_date'].max()
            days_since_last = (datetime.now() - last_purchase).days
            
            # Determine churn status
            churned = days_since_last > 90  # 90 days without purchase
            
            # Calculate purchase frequency
            purchase_frequency = len(customer_sales) / max(1, (last_purchase - first_purchase).days / 30)
            
            # Calculate average order value
            avg_order_value = customer_sales['total_amount'].mean()
            
            churn_record = {
                'customer_id': customer['customer_id'],
                'segment': customer['segment'],
                'first_purchase_date': first_purchase,
                'last_purchase_date': last_purchase,
                'days_since_last_purchase': days_since_last,
                'total_purchases': len(customer_sales),
                'total_revenue': customer_sales['total_amount'].sum(),
                'avg_order_value': avg_order_value,
                'purchase_frequency': purchase_frequency,
                'churned': churned,
                'churn_probability': self._calculate_churn_probability(customer, days_since_last, purchase_frequency)
            }
            
            churn_data.append(churn_record)
        
        return pd.DataFrame(churn_data)
    
    def _calculate_churn_probability(self, customer: pd.Series, days_since_last: int, frequency: float) -> float:
        """
        Calculate churn probability based on customer behavior.
        
        Args:
            customer: Customer information
            days_since_last: Days since last purchase
            frequency: Purchase frequency
            
        Returns:
            Churn probability (0-1)
        """
        # Base churn probability
        base_prob = 0.1
        
        # Adjust for days since last purchase
        if days_since_last > 90:
            base_prob += 0.4
        elif days_since_last > 60:
            base_prob += 0.2
        elif days_since_last > 30:
            base_prob += 0.1
        
        # Adjust for purchase frequency
        if frequency < 0.5:
            base_prob += 0.2
        elif frequency < 1.0:
            base_prob += 0.1
        
        # Adjust for customer segment
        segment_info = self.customer_segments[customer['segment']]
        base_prob *= (1 - segment_info['retention'])
        
        return min(base_prob, 0.95)
    
    def generate_cohort_data(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cohort analysis data.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            DataFrame with cohort analysis
        """
        print("Generating cohort analysis data...")
        
        cohort_data = []
        
        for _, customer in customers_df.iterrows():
            # Get customer's purchase history
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                continue
            
            # Calculate cohort metrics
            acquisition_month = customer['acquisition_date'].replace(day=1)
            first_purchase = customer_sales['purchase_date'].min()
            last_purchase = customer_sales['purchase_date'].max()
            
            # Calculate months since acquisition
            months_since_acquisition = (last_purchase - acquisition_month).days / 30
            
            # Calculate retention periods
            for month in range(int(months_since_acquisition) + 1):
                period_start = acquisition_month + timedelta(days=month * 30)
                period_end = period_start + timedelta(days=30)
                
                # Check if customer made purchases in this period
                period_sales = customer_sales[
                    (customer_sales['purchase_date'] >= period_start) &
                    (customer_sales['purchase_date'] < period_end)
                ]
                
                retained = len(period_sales) > 0
                revenue = period_sales['total_amount'].sum() if retained else 0
                
                cohort_record = {
                    'customer_id': customer['customer_id'],
                    'cohort_month': acquisition_month,
                    'period': month,
                    'retained': retained,
                    'revenue': revenue,
                    'segment': customer['segment'],
                    'sales_channel': customer['sales_channel'],
                    'region': customer['region']
                }
                
                cohort_data.append(cohort_record)
        
        return pd.DataFrame(cohort_data)
    
    def save_data(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame, 
                  churn_df: pd.DataFrame = None, cohort_df: pd.DataFrame = None,
                  output_dir: str = 'data/processed/'):
        """
        Save generated data to files.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            churn_df: Churn analysis data
            cohort_df: Cohort analysis data
            output_dir: Output directory
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
        sales_df.to_csv(f'{output_dir}/sales.csv', index=False)
        
        if churn_df is not None:
            churn_df.to_csv(f'{output_dir}/churn_analysis.csv', index=False)
        
        if cohort_df is not None:
            cohort_df.to_csv(f'{output_dir}/cohort_analysis.csv', index=False)
        
        print(f"Data saved to {output_dir}")
    
    def generate_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate sample dataset for demonstration.
        
        Returns:
            Tuple of (customers_df, sales_df, churn_df, cohort_df)
        """
        print("Generating sample dataset...")
        
        # Generate base data
        customers_df, sales_df = self.generate_complete_dataset(n_customers=5000, months=18)
        
        # Generate analysis data
        churn_df = self.generate_churn_data(customers_df, sales_df)
        cohort_df = self.generate_cohort_data(customers_df, sales_df)
        
        # Save data
        self.save_data(customers_df, sales_df, churn_df, cohort_df)
        
        return customers_df, sales_df, churn_df, cohort_df

def main():
    """Generate sample data for the dashboard."""
    generator = DataGenerator(seed=42)
    
    # Generate sample data
    customers_df, sales_df, churn_df, cohort_df = generator.generate_sample_data()
    
    print("\nData generation completed!")
    print(f"Customers: {len(customers_df)}")
    print(f"Sales transactions: {len(sales_df)}")
    print(f"Churn records: {len(churn_df)}")
    print(f"Cohort records: {len(cohort_df)}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Revenue: ${sales_df['total_amount'].sum():,.2f}")
    print(f"Average Order Value: ${sales_df['total_amount'].mean():.2f}")
    print(f"Churn Rate: {(churn_df['churned'].sum() / len(churn_df) * 100):.1f}%")
    print(f"Active Customers: {customers_df['is_active'].sum()}")

if __name__ == "__main__":
    main() 