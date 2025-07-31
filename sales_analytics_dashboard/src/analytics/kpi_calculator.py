"""
KPI Calculator for sales analytics dashboard.
Implements advanced KPIs including CLV, churn rate, and cohort analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class KPICalculator:
    """Advanced KPI calculator for sales analytics."""
    
    def __init__(self, config=None):
        """Initialize KPI calculator with configuration."""
        self.config = config or {
            'clv': {
                'prediction_horizon': 12,  # months
                'discount_rate': 0.1,
                'calculation_method': 'historical'
            },
            'churn': {
                'definition': 'no_purchase_90_days',
                'prediction_threshold': 0.7,
                'alert_threshold': 0.05
            },
            'cohorts': {
                'time_granularity': 'month',
                'retention_periods': 12,
                'revenue_analysis': True
            }
        }
    
    def calculate_clv(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame, 
                     method: str = 'historical') -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV).
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            method: Calculation method ('historical' or 'predictive')
            
        Returns:
            DataFrame with CLV calculations
        """
        print("Calculating Customer Lifetime Value...")
        
        clv_data = []
        
        for _, customer in customers_df.iterrows():
            # Get customer's sales data
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                # New customer with no purchases
                clv_record = {
                    'customer_id': customer['customer_id'],
                    'segment': customer['segment'],
                    'historical_clv': 0.0,
                    'predictive_clv': 0.0,
                    'avg_order_value': 0.0,
                    'purchase_frequency': 0.0,
                    'customer_lifespan': 0.0
                }
            else:
                # Calculate historical CLV
                historical_clv = customer_sales['total_amount'].sum()
                
                # Calculate average order value
                avg_order_value = customer_sales['total_amount'].mean()
                
                # Calculate purchase frequency
                first_purchase = customer_sales['purchase_date'].min()
                last_purchase = customer_sales['purchase_date'].max()
                customer_lifespan = (last_purchase - first_purchase).days / 30  # months
                
                if customer_lifespan > 0:
                    purchase_frequency = len(customer_sales) / customer_lifespan
                else:
                    purchase_frequency = len(customer_sales)
                
                # Calculate predictive CLV
                if method == 'predictive':
                    predictive_clv = self._calculate_predictive_clv(
                        historical_clv, purchase_frequency, customer['segment']
                    )
                else:
                    predictive_clv = historical_clv
                
                clv_record = {
                    'customer_id': customer['customer_id'],
                    'segment': customer['segment'],
                    'historical_clv': historical_clv,
                    'predictive_clv': predictive_clv,
                    'avg_order_value': avg_order_value,
                    'purchase_frequency': purchase_frequency,
                    'customer_lifespan': customer_lifespan
                }
            
            clv_data.append(clv_record)
        
        return pd.DataFrame(clv_data)
    
    def _calculate_predictive_clv(self, historical_clv: float, frequency: float, 
                                 segment: str) -> float:
        """
        Calculate predictive CLV based on historical data and segment.
        
        Args:
            historical_clv: Historical CLV
            frequency: Purchase frequency
            segment: Customer segment
            
        Returns:
            Predictive CLV
        """
        # Segment-based retention rates
        retention_rates = {
            'Premium': 0.9,
            'Regular': 0.7,
            'Occasional': 0.5,
            'New': 0.3
        }
        
        retention_rate = retention_rates.get(segment, 0.5)
        
        # Calculate predictive CLV
        # Formula: CLV = Historical CLV * Retention Rate * Frequency Factor
        frequency_factor = min(frequency * 2, 3)  # Cap frequency factor
        predictive_clv = historical_clv * retention_rate * frequency_factor
        
        return predictive_clv
    
    def calculate_churn_rate(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame,
                           period: str = 'monthly') -> Dict:
        """
        Calculate churn rate and related metrics.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            period: Analysis period ('monthly', 'quarterly', 'yearly')
            
        Returns:
            Dictionary with churn metrics
        """
        print("Calculating churn rate...")
        
        # Calculate churn data
        churn_data = []
        
        for _, customer in customers_df.iterrows():
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                continue
            
            last_purchase = customer_sales['purchase_date'].max()
            days_since_last = (datetime.now() - last_purchase).days
            
            # Determine churn status based on definition
            churn_definition = self.config['churn']['definition']
            
            if churn_definition == 'no_purchase_90_days':
                churned = days_since_last > 90
            elif churn_definition == 'no_purchase_60_days':
                churned = days_since_last > 60
            else:
                churned = days_since_last > 90  # Default
            
            churn_record = {
                'customer_id': customer['customer_id'],
                'segment': customer['segment'],
                'days_since_last_purchase': days_since_last,
                'churned': churned,
                'last_purchase_date': last_purchase
            }
            
            churn_data.append(churn_record)
        
        churn_df = pd.DataFrame(churn_data)
        
        # Calculate overall churn rate
        total_customers = len(churn_df)
        churned_customers = churn_df['churned'].sum()
        overall_churn_rate = churned_customers / total_customers if total_customers > 0 else 0
        
        # Calculate churn rate by segment
        segment_churn = churn_df.groupby('segment')['churned'].agg(['sum', 'count']).reset_index()
        segment_churn['churn_rate'] = segment_churn['sum'] / segment_churn['count']
        
        # Calculate churn rate by period
        if period == 'monthly':
            churn_df['month'] = churn_df['last_purchase_date'].dt.to_period('M')
            period_churn = churn_df.groupby('month')['churned'].agg(['sum', 'count']).reset_index()
            period_churn['churn_rate'] = period_churn['sum'] / period_churn['count']
        
        return {
            'overall_churn_rate': overall_churn_rate,
            'churned_customers': churned_customers,
            'total_customers': total_customers,
            'segment_churn': segment_churn,
            'period_churn': period_churn if period == 'monthly' else None,
            'churn_data': churn_df
        }
    
    def calculate_cohort_analysis(self, customers_df: pd.DataFrame, 
                                sales_df: pd.DataFrame) -> Dict:
        """
        Calculate cohort analysis metrics.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            Dictionary with cohort analysis results
        """
        print("Calculating cohort analysis...")
        
        cohort_data = []
        
        for _, customer in customers_df.iterrows():
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                continue
            
            # Get cohort month (acquisition month)
            cohort_month = customer['acquisition_date'].replace(day=1)
            
            # Calculate retention periods
            max_periods = self.config['cohorts']['retention_periods']
            
            for period in range(max_periods):
                period_start = cohort_month + timedelta(days=period * 30)
                period_end = period_start + timedelta(days=30)
                
                # Check if customer made purchases in this period
                period_sales = customer_sales[
                    (customer_sales['purchase_date'] >= period_start) &
                    (customer_sales['purchase_date'] < period_end)
                ]
                
                retained = len(period_sales) > 0
                revenue = period_sales['total_amount'].sum() if retained else 0
                
                cohort_record = {
                    'cohort_month': cohort_month,
                    'period': period,
                    'customer_id': customer['customer_id'],
                    'segment': customer['segment'],
                    'retained': retained,
                    'revenue': revenue,
                    'purchase_count': len(period_sales)
                }
                
                cohort_data.append(cohort_record)
        
        cohort_df = pd.DataFrame(cohort_data)
        
        # Calculate cohort retention matrix
        retention_matrix = cohort_df.groupby(['cohort_month', 'period'])['retained'].agg([
            'sum', 'count'
        ]).reset_index()
        retention_matrix['retention_rate'] = retention_matrix['sum'] / retention_matrix['count']
        
        # Calculate cohort revenue matrix
        revenue_matrix = cohort_df.groupby(['cohort_month', 'period'])['revenue'].agg([
            'sum', 'mean'
        ]).reset_index()
        
        # Calculate cohort size
        cohort_size = cohort_df.groupby('cohort_month')['customer_id'].nunique().reset_index()
        cohort_size.columns = ['cohort_month', 'cohort_size']
        
        return {
            'cohort_data': cohort_df,
            'retention_matrix': retention_matrix,
            'revenue_matrix': revenue_matrix,
            'cohort_size': cohort_size
        }
    
    def calculate_sales_kpis(self, sales_df: pd.DataFrame) -> Dict:
        """
        Calculate sales-related KPIs.
        
        Args:
            sales_df: Sales data
            
        Returns:
            Dictionary with sales KPIs
        """
        print("Calculating sales KPIs...")
        
        # Revenue metrics
        total_revenue = sales_df['total_amount'].sum()
        avg_order_value = sales_df['total_amount'].mean()
        
        # Monthly Recurring Revenue (MRR)
        sales_df['month'] = sales_df['purchase_date'].dt.to_period('M')
        mrr = sales_df.groupby('month')['total_amount'].sum().mean()
        
        # Annual Recurring Revenue (ARR)
        arr = mrr * 12
        
        # Revenue growth
        monthly_revenue = sales_df.groupby('month')['total_amount'].sum().reset_index()
        monthly_revenue['month'] = monthly_revenue['month'].astype(str)
        monthly_revenue = monthly_revenue.sort_values('month')
        
        if len(monthly_revenue) > 1:
            revenue_growth = ((monthly_revenue['total_amount'].iloc[-1] - 
                             monthly_revenue['total_amount'].iloc[-2]) / 
                            monthly_revenue['total_amount'].iloc[-2]) * 100
        else:
            revenue_growth = 0
        
        # Conversion metrics
        unique_customers = sales_df['customer_id'].nunique()
        total_transactions = len(sales_df)
        avg_purchase_frequency = total_transactions / unique_customers if unique_customers > 0 else 0
        
        # Product performance
        product_performance = sales_df.groupby('product_category').agg({
            'total_amount': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).reset_index()
        
        product_performance.columns = ['product_category', 'total_revenue', 'transaction_count', 
                                     'avg_order_value', 'unique_customers']
        
        # Channel performance
        channel_performance = sales_df.groupby('sales_channel').agg({
            'total_amount': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).reset_index()
        
        channel_performance.columns = ['sales_channel', 'total_revenue', 'transaction_count', 
                                     'avg_order_value', 'unique_customers']
        
        return {
            'total_revenue': total_revenue,
            'avg_order_value': avg_order_value,
            'mrr': mrr,
            'arr': arr,
            'revenue_growth': revenue_growth,
            'unique_customers': unique_customers,
            'total_transactions': total_transactions,
            'avg_purchase_frequency': avg_purchase_frequency,
            'product_performance': product_performance,
            'channel_performance': channel_performance,
            'monthly_revenue': monthly_revenue
        }
    
    def calculate_customer_segmentation(self, customers_df: pd.DataFrame, 
                                      sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform RFM (Recency, Frequency, Monetary) customer segmentation.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            DataFrame with customer segments
        """
        print("Calculating customer segmentation...")
        
        # Calculate RFM metrics for each customer
        rfm_data = []
        
        for _, customer in customers_df.iterrows():
            customer_sales = sales_df[sales_df['customer_id'] == customer['customer_id']]
            
            if len(customer_sales) == 0:
                # Customer with no purchases
                rfm_record = {
                    'customer_id': customer['customer_id'],
                    'recency': 999,  # High value for no purchases
                    'frequency': 0,
                    'monetary': 0,
                    'rfm_score': 0,
                    'segment': 'Lost'
                }
            else:
                # Calculate RFM metrics
                last_purchase = customer_sales['purchase_date'].max()
                recency = (datetime.now() - last_purchase).days
                frequency = len(customer_sales)
                monetary = customer_sales['total_amount'].sum()
                
                # Calculate RFM score (1-5 scale)
                r_score = self._calculate_r_score(recency)
                f_score = self._calculate_f_score(frequency)
                m_score = self._calculate_m_score(monetary)
                
                rfm_score = r_score + f_score + m_score
                
                # Assign segment based on RFM score
                segment = self._assign_rfm_segment(rfm_score, r_score, f_score, m_score)
                
                rfm_record = {
                    'customer_id': customer['customer_id'],
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary,
                    'r_score': r_score,
                    'f_score': f_score,
                    'm_score': m_score,
                    'rfm_score': rfm_score,
                    'segment': segment
                }
            
            rfm_data.append(rfm_record)
        
        return pd.DataFrame(rfm_data)
    
    def _calculate_r_score(self, recency: int) -> int:
        """Calculate recency score (1-5)."""
        if recency <= 30:
            return 5
        elif recency <= 60:
            return 4
        elif recency <= 90:
            return 3
        elif recency <= 180:
            return 2
        else:
            return 1
    
    def _calculate_f_score(self, frequency: int) -> int:
        """Calculate frequency score (1-5)."""
        if frequency >= 10:
            return 5
        elif frequency >= 5:
            return 4
        elif frequency >= 3:
            return 3
        elif frequency >= 2:
            return 2
        else:
            return 1
    
    def _calculate_m_score(self, monetary: float) -> int:
        """Calculate monetary score (1-5)."""
        if monetary >= 1000:
            return 5
        elif monetary >= 500:
            return 4
        elif monetary >= 200:
            return 3
        elif monetary >= 100:
            return 2
        else:
            return 1
    
    def _assign_rfm_segment(self, rfm_score: int, r_score: int, f_score: int, m_score: int) -> str:
        """Assign customer segment based on RFM scores."""
        if rfm_score >= 13:
            return 'Champions'
        elif rfm_score >= 10:
            return 'Loyal Customers'
        elif r_score >= 4 and f_score >= 3:
            return 'Recent Customers'
        elif m_score >= 4:
            return 'Big Spenders'
        elif f_score >= 4:
            return 'Frequent Buyers'
        elif r_score >= 3:
            return 'At Risk'
        elif r_score <= 2 and f_score <= 2:
            return 'Lost'
        else:
            return 'Average'
    
    def calculate_all_kpis(self, customers_df: pd.DataFrame, sales_df: pd.DataFrame) -> Dict:
        """
        Calculate all KPIs in one call.
        
        Args:
            customers_df: Customer data
            sales_df: Sales data
            
        Returns:
            Dictionary with all KPI calculations
        """
        print("Calculating all KPIs...")
        
        # Calculate CLV
        clv_df = self.calculate_clv(customers_df, sales_df)
        
        # Calculate churn rate
        churn_results = self.calculate_churn_rate(customers_df, sales_df)
        
        # Calculate cohort analysis
        cohort_results = self.calculate_cohort_analysis(customers_df, sales_df)
        
        # Calculate sales KPIs
        sales_kpis = self.calculate_sales_kpis(sales_df)
        
        # Calculate customer segmentation
        segmentation_df = self.calculate_customer_segmentation(customers_df, sales_df)
        
        return {
            'clv': clv_df,
            'churn': churn_results,
            'cohorts': cohort_results,
            'sales_kpis': sales_kpis,
            'segmentation': segmentation_df
        }

def main():
    """Test KPI calculator with sample data."""
    from src.data.data_generator import DataGenerator
    
    # Generate sample data
    generator = DataGenerator()
    customers_df, sales_df, churn_df, cohort_df = generator.generate_sample_data()
    
    # Calculate KPIs
    calculator = KPICalculator()
    kpi_results = calculator.calculate_all_kpis(customers_df, sales_df)
    
    # Print summary
    print("\nKPI Summary:")
    print(f"Average CLV: ${kpi_results['clv']['predictive_clv'].mean():.2f}")
    print(f"Churn Rate: {kpi_results['churn']['overall_churn_rate']*100:.1f}%")
    print(f"Total Revenue: ${kpi_results['sales_kpis']['total_revenue']:,.2f}")
    print(f"MRR: ${kpi_results['sales_kpis']['mrr']:,.2f}")

if __name__ == "__main__":
    main() 