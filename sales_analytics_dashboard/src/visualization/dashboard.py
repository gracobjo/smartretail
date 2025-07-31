"""
Sales Analytics Dashboard using Dash.
Interactive dashboard with advanced KPIs and visualizations.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import local modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_generator import DataGenerator
from analytics.kpi_calculator import KPICalculator

class SalesDashboard:
    """Interactive sales analytics dashboard."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
        ])
        
        self.app.title = "Sales Analytics Dashboard - SmartRetail"
        
        # Load or generate data
        self.load_data()
        
        # Calculate KPIs
        self.calculate_kpis()
        
        # Setup layout
        self.setup_layout()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def load_data(self):
        """Load or generate sample data."""
        try:
            # Try to load existing data
            customers_df = pd.read_csv('data/processed/customers.csv')
            sales_df = pd.read_csv('data/processed/sales.csv')
            
            # Convert date columns
            customers_df['acquisition_date'] = pd.to_datetime(customers_df['acquisition_date'])
            customers_df['last_purchase_date'] = pd.to_datetime(customers_df['last_purchase_date'])
            sales_df['purchase_date'] = pd.to_datetime(sales_df['purchase_date'])
            
            print("Loaded existing data")
            
        except FileNotFoundError:
            # Generate new data
            print("Generating sample data...")
            generator = DataGenerator()
            customers_df, sales_df, _, _ = generator.generate_sample_data()
        
        self.customers_df = customers_df
        self.sales_df = sales_df
    
    def calculate_kpis(self):
        """Calculate all KPIs."""
        print("Calculating KPIs...")
        calculator = KPICalculator()
        self.kpi_results = calculator.calculate_all_kpis(self.customers_df, self.sales_df)
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Sales Analytics Dashboard", className="text-center mb-4"),
                html.P("Advanced KPIs and Analytics for Sales Performance", 
                      className="text-center text-muted")
            ], className="container-fluid bg-primary text-white py-3"),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Date Range:", className="form-label"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=self.sales_df['purchase_date'].min().date(),
                        end_date=self.sales_df['purchase_date'].max().date(),
                        display_format='YYYY-MM-DD'
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Customer Segment:", className="form-label"),
                    dcc.Dropdown(
                        id='segment-filter',
                        options=[{'label': seg, 'value': seg} 
                                for seg in self.customers_df['segment'].unique()],
                        value='all',
                        placeholder="Select segment"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Product Category:", className="form-label"),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': cat, 'value': cat} 
                                for cat in self.sales_df['product_category'].unique()],
                        value='all',
                        placeholder="Select category"
                    )
                ], className="col-md-3"),
                
                html.Div([
                    html.Label("Sales Channel:", className="form-label"),
                    dcc.Dropdown(
                        id='channel-filter',
                        options=[{'label': chan, 'value': chan} 
                                for chan in self.sales_df['sales_channel'].unique()],
                        value='all',
                        placeholder="Select channel"
                    )
                ], className="col-md-3")
            ], className="row mb-4"),
            
            # KPI Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Total Revenue", className="card-title"),
                        html.H2(id="total-revenue", className="text-primary"),
                        html.P("All time", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("MRR", className="card-title"),
                        html.H2(id="mrr", className="text-success"),
                        html.P("Monthly Recurring Revenue", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("Avg CLV", className="card-title"),
                        html.H2(id="avg-clv", className="text-info"),
                        html.P("Customer Lifetime Value", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3"),
                
                html.Div([
                    html.Div([
                        html.H4("Churn Rate", className="card-title"),
                        html.H2(id="churn-rate", className="text-warning"),
                        html.P("Monthly churn rate", className="text-muted")
                    ], className="card-body text-center")
                ], className="col-md-3")
            ], className="row mb-4", id="kpi-cards"),
            
            # Charts Row 1
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Revenue Trend", className="card-title"),
                        dcc.Graph(id="revenue-trend")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Customer Segments", className="card-title"),
                        dcc.Graph(id="customer-segments")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Charts Row 2
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("CLV Distribution", className="card-title"),
                        dcc.Graph(id="clv-distribution")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Churn Analysis", className="card-title"),
                        dcc.Graph(id="churn-analysis")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Charts Row 3
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Cohort Retention", className="card-title"),
                        dcc.Graph(id="cohort-retention")
                    ], className="card-body")
                ], className="col-md-12")
            ], className="row mb-4"),
            
            # Charts Row 4
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("Product Performance", className="card-title"),
                        dcc.Graph(id="product-performance")
                    ], className="card-body")
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5("Channel Performance", className="card-title"),
                        dcc.Graph(id="channel-performance")
                    ], className="card-body")
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Footer
            html.Div([
                html.P("SmartRetail Analytics Dashboard", 
                      className="text-center text-muted")
            ], className="container-fluid bg-light py-3")
            
        ], className="container-fluid")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("total-revenue", "children"),
             Output("mrr", "children"),
             Output("avg-clv", "children"),
             Output("churn-rate", "children")],
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("segment-filter", "value"),
             Input("category-filter", "value"),
             Input("channel-filter", "value")]
        )
        def update_kpi_cards(start_date, end_date, segment, category, channel):
            """Update KPI cards based on filters."""
            
            # Apply filters
            filtered_sales = self.apply_filters(start_date, end_date, segment, category, channel)
            filtered_customers = self.apply_customer_filters(segment)
            
            # Calculate KPIs
            total_revenue = filtered_sales['total_amount'].sum()
            mrr = filtered_sales.groupby(filtered_sales['purchase_date'].dt.to_period('M'))['total_amount'].sum().mean()
            avg_clv = self.kpi_results['clv']['predictive_clv'].mean()
            churn_rate = self.kpi_results['churn']['overall_churn_rate'] * 100
            
            return [
                f"${total_revenue:,.0f}",
                f"${mrr:,.0f}",
                f"${avg_clv:.0f}",
                f"{churn_rate:.1f}%"
            ]
        
        @self.app.callback(
            Output("revenue-trend", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("segment-filter", "value"),
             Input("category-filter", "value"),
             Input("channel-filter", "value")]
        )
        def update_revenue_trend(start_date, end_date, segment, category, channel):
            """Update revenue trend chart."""
            
            filtered_sales = self.apply_filters(start_date, end_date, segment, category, channel)
            
            # Group by month
            monthly_revenue = filtered_sales.groupby(
                filtered_sales['purchase_date'].dt.to_period('M')
            )['total_amount'].sum().reset_index()
            
            monthly_revenue['month'] = monthly_revenue['purchase_date'].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_revenue['month'],
                y=monthly_revenue['total_amount'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Monthly Revenue Trend",
                xaxis_title="Month",
                yaxis_title="Revenue ($)",
                hovermode='x unified',
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("customer-segments", "figure"),
            [Input("segment-filter", "value")]
        )
        def update_customer_segments(segment):
            """Update customer segments chart."""
            
            if segment == 'all':
                segment_data = self.customers_df['segment'].value_counts()
            else:
                segment_data = self.customers_df[self.customers_df['segment'] == segment]['segment'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=segment_data.index,
                values=segment_data.values,
                hole=0.4,
                marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )])
            
            fig.update_layout(
                title="Customer Segments Distribution",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("clv-distribution", "figure"),
            [Input("segment-filter", "value")]
        )
        def update_clv_distribution(segment):
            """Update CLV distribution chart."""
            
            clv_data = self.kpi_results['clv']
            
            if segment != 'all':
                clv_data = clv_data[clv_data['segment'] == segment]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=clv_data['predictive_clv'],
                nbinsx=30,
                name='CLV Distribution',
                marker_color='#1f77b4'
            ))
            
            fig.update_layout(
                title="Customer Lifetime Value Distribution",
                xaxis_title="CLV ($)",
                yaxis_title="Number of Customers",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("churn-analysis", "figure"),
            [Input("segment-filter", "value")]
        )
        def update_churn_analysis(segment):
            """Update churn analysis chart."""
            
            churn_data = self.kpi_results['churn']['segment_churn']
            
            if segment != 'all':
                churn_data = churn_data[churn_data['segment'] == segment]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=churn_data['segment'],
                y=churn_data['churn_rate'] * 100,
                name='Churn Rate',
                marker_color='#ff7f0e'
            ))
            
            fig.update_layout(
                title="Churn Rate by Segment",
                xaxis_title="Segment",
                yaxis_title="Churn Rate (%)",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("cohort-retention", "figure"),
            [Input("segment-filter", "value")]
        )
        def update_cohort_retention(segment):
            """Update cohort retention chart."""
            
            cohort_data = self.kpi_results['cohorts']['retention_matrix']
            
            if segment != 'all':
                # Filter cohort data by segment
                cohort_customers = self.kpi_results['cohorts']['cohort_data']
                cohort_customers = cohort_customers[cohort_customers['segment'] == segment]
                cohort_months = cohort_customers['cohort_month'].unique()
                cohort_data = cohort_data[cohort_data['cohort_month'].isin(cohort_months)]
            
            # Create heatmap data
            pivot_data = cohort_data.pivot(
                index='cohort_month', 
                columns='period', 
                values='retention_rate'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Blues',
                text=np.round(pivot_data.values * 100, 1),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Cohort Retention Heatmap",
                xaxis_title="Period (Months)",
                yaxis_title="Cohort Month",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("product-performance", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("segment-filter", "value"),
             Input("category-filter", "value"),
             Input("channel-filter", "value")]
        )
        def update_product_performance(start_date, end_date, segment, category, channel):
            """Update product performance chart."""
            
            filtered_sales = self.apply_filters(start_date, end_date, segment, category, channel)
            
            product_perf = filtered_sales.groupby('product_category').agg({
                'total_amount': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=product_perf['product_category'],
                y=product_perf['total_amount'],
                name='Revenue',
                marker_color='#2ca02c'
            ))
            
            fig.update_layout(
                title="Product Category Performance",
                xaxis_title="Product Category",
                yaxis_title="Revenue ($)",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output("channel-performance", "figure"),
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("segment-filter", "value"),
             Input("category-filter", "value"),
             Input("channel-filter", "value")]
        )
        def update_channel_performance(start_date, end_date, segment, category, channel):
            """Update channel performance chart."""
            
            filtered_sales = self.apply_filters(start_date, end_date, segment, category, channel)
            
            channel_perf = filtered_sales.groupby('sales_channel').agg({
                'total_amount': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=channel_perf['sales_channel'],
                y=channel_perf['total_amount'],
                name='Revenue',
                marker_color='#d62728'
            ))
            
            fig.update_layout(
                title="Sales Channel Performance",
                xaxis_title="Sales Channel",
                yaxis_title="Revenue ($)",
                template="plotly_white"
            )
            
            return fig
    
    def apply_filters(self, start_date, end_date, segment, category, channel):
        """Apply filters to sales data."""
        filtered_sales = self.sales_df.copy()
        
        # Date filter
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            filtered_sales = filtered_sales[
                (filtered_sales['purchase_date'] >= start_date) &
                (filtered_sales['purchase_date'] <= end_date)
            ]
        
        # Segment filter
        if segment != 'all':
            filtered_sales = filtered_sales[filtered_sales['customer_segment'] == segment]
        
        # Category filter
        if category != 'all':
            filtered_sales = filtered_sales[filtered_sales['product_category'] == category]
        
        # Channel filter
        if channel != 'all':
            filtered_sales = filtered_sales[filtered_sales['sales_channel'] == channel]
        
        return filtered_sales
    
    def apply_customer_filters(self, segment):
        """Apply filters to customer data."""
        filtered_customers = self.customers_df.copy()
        
        if segment != 'all':
            filtered_customers = filtered_customers[filtered_customers['segment'] == segment]
        
        return filtered_customers
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=port)

def main():
    """Run the sales analytics dashboard."""
    dashboard = SalesDashboard()
    print("Starting Sales Analytics Dashboard...")
    print("Open http://localhost:8050 in your browser")
    dashboard.run()

if __name__ == "__main__":
    main() 