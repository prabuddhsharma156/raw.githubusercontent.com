import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Sales & Inventory Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(uploaded_file):
    """
    Loads, cleans, and prepares sales data from a user-uploaded file.
    This function is now upgraded to dynamically read product names from the file header.
    """
    try:
        # Intelligently read the multi-level header to get real product names.
        header_df = pd.read_excel(uploaded_file, header=None, nrows=2, engine='openpyxl')
        header_df.iloc[0] = header_df.iloc[0].ffill()
        header_df = header_df.fillna('')
        combined_headers = header_df.iloc[0].astype(str) + ' ' + header_df.iloc[1].astype(str)
        combined_headers = combined_headers.str.strip().str.replace(' ', '_').str.replace('__', '_').str.rstrip('_')
        
        df = pd.read_excel(uploaded_file, header=None, skiprows=2, engine='openpyxl')
        
        num_columns_read = df.shape[1]
        df.columns = combined_headers[:num_columns_read]
        
        df = df.rename(columns={df.columns[0]: 'Date'})

        # --- The rest of the cleaning process continues as before ---
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        if df.empty:
            st.warning("Warning: The uploaded file was loaded, but no valid data rows were found after cleaning. Please check the file's content and format.")
            return None
        
        numeric_cols = [col for col in df.columns if col != 'Date' and 'Total' not in col]

        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in numeric_cols:
            df[col] = df[col].astype(int)

        df.set_index('Date', inplace=True)
        total_cols_to_drop = [col for col in df.columns if 'Total' in col]
        df.drop(columns=total_cols_to_drop, inplace=True)
            
        df = df.loc[:, (df != 0).any(axis=0)]
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Main Application Logic ---
st.title("📈 Dynamic Sales & Inventory Dashboard")
st.markdown("Upload your sales data in Excel format to generate an interactive business analytics report.")

uploaded_file = st.file_uploader(
    "Choose an Excel file (ensure it has a 2-row header for product names)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # --- Sidebar ---
        st.sidebar.title("Dashboard Navigation")
        st.sidebar.markdown("Use the options below to explore the business analytics.")
        # NEW FEATURE: Added "Profit Analysis" to the navigation
        page = st.sidebar.radio("Go to", ("Executive Summary", "What-If Simulation", "Detailed Product Analysis", "Sales Forecasting", "Profit Analysis"))
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This dashboard provides automated analysis for sales and inventory management."
        )

        # --- Pre-computation for Analytics ---
        total_sales = df.sum().sort_values(ascending=False)
        mean_sales = df.mean()
        std_sales = df.std()
        coefficient_of_variation = (std_sales / mean_sales).fillna(0).sort_values(ascending=False)
        
        # --- Page 1: Executive Summary ---
        if page == "Executive Summary":
            st.header("Executive Summary")
            st.markdown("This page provides a high-level overview of product performance and demand stability.")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Top 3 Best-Selling Products")
                st.dataframe(total_sales.head(3).rename("Total Units Sold"))

            with col2:
                st.subheader("❌ Bottom 3 Worst-Selling Products")
                st.dataframe(total_sales.tail(3).rename("Total Units Sold"))
                
            st.markdown("---")
            st.subheader("⚡ Demand Volatility Analysis (Predictability)")
            st.markdown(
                "The **Coefficient of Variation (CV)** measures sales predictability. "
                "A **high CV** indicates volatile, unpredictable demand, which requires higher safety stock."
            )
            
            cv_df = coefficient_of_variation.reset_index()
            cv_df.columns = ['Product', 'Coefficient of Variation']
            
            chart = alt.Chart(cv_df).mark_bar().encode(
                x=alt.X('Product:N', sort='-y', title="Product"),
                y=alt.Y('Coefficient of Variation:Q', title="CV (Higher = Less Predictable)"),
                tooltip=['Product', 'Coefficient of Variation']
            ).properties(
                title='Product Sales Volatility'
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

        # --- Page 2: What-If Simulation ---
        elif page == "What-If Simulation":
            st.header("🔬 'What-If' Inventory Simulation")
            st.markdown("Test different inventory strategies to see how they perform against simulated demand.")

            col1, col2, col3 = st.columns(3)
            with col1:
                product = st.selectbox("Select Product for Simulation", df.columns)
            with col2:
                initial_stock = st.number_input("Initial Stock Level", min_value=0, value=3000, step=100)
            with col3:
                reorder_point = st.number_input("Reorder Point", min_value=0, value=1000, step=50)

            if st.button("▶️ Run Simulation"):
                # Simulation logic... (omitted for brevity)
                pass

        # --- Page 3: Detailed Product Analysis ---
        elif page == "Detailed Product Analysis":
            st.header("📊 Detailed Product Analysis")
            st.markdown("Select any product to view its sales trend and key performance indicators.")

            product_to_view = st.selectbox("Select a Product", df.columns, key="detailed_product")
            
            if product_to_view:
                # Detailed analysis logic... (omitted for brevity)
                pass
        
        # --- Page 4: Sales Forecasting ---
        elif page == "Sales Forecasting":
            st.header("🔮 Sales Forecasting")
            st.markdown("Generate a sales forecast for any product based on its recent performance.")

            col1, col2 = st.columns(2)
            with col1:
                product_to_forecast = st.selectbox("Select Product to Forecast", df.columns)
            with col2:
                forecast_days = st.slider("Number of Days to Forecast", min_value=7, max_value=90, value=30, step=7)

            if st.button("📈 Generate Forecast"):
                # Forecasting logic... (omitted for brevity)
                pass
        
        # --- NEW FEATURE PAGE: Profit Analysis ---
        elif page == "Profit Analysis":
            st.header("💰 Profit Analysis")
            st.markdown("Enter the cost and price for each product to calculate and visualize your daily profit.")
            
            # Create a form for user input
            with st.form(key='profit_form'):
                st.subheader("Enter Product Financials")
                
                # Create a dictionary to hold the financial inputs
                financial_data = {}
                
                # Create two columns for a cleaner layout
                col1, col2 = st.columns(2)
                
                # Dynamically create input fields for each product
                for i, product_name in enumerate(df.columns):
                    # Alternate between columns for the inputs
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"**{product_name}**")
                        unit_cost = st.number_input(f"Unit Cost (₹)", key=f"cost_{product_name}", min_value=0.0, step=0.5, format="%.2f")
                        unit_price = st.number_input(f"Unit Price (₹)", key=f"price_{product_name}", min_value=0.0, step=0.5, format="%.2f")
                        financial_data[product_name] = {'cost': unit_cost, 'price': unit_price}

                submit_button = st.form_submit_button(label='📊 Calculate and Analyze Profit')

            if submit_button:
                st.markdown("---")
                st.subheader("Profit Calculation Results")

                # Create a copy of the original dataframe to avoid modifying it
                profit_df = df.copy()
                
                # Calculate profit for each product
                for product_name, data in financial_data.items():
                    if data['price'] > 0: # Only calculate if price is entered
                        unit_profit = data['price'] - data['cost']
                        profit_df[product_name] = profit_df[product_name] * unit_profit
                    else:
                        profit_df[product_name] = 0 # Ignore products without a price

                # Calculate total daily profit
                daily_profit = profit_df.sum(axis=1)

                if daily_profit.sum() > 0:
                    # Display metrics
                    total_profit = daily_profit.sum()
                    avg_daily_profit = daily_profit.mean()
                    best_profit_day = daily_profit.idxmax()
                    
                    st.subheader("Key Profit Metrics")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric(label="Total Profit", value=f"₹{total_profit:,.2f}")
                    kpi2.metric(label="Average Daily Profit", value=f"₹{avg_daily_profit:,.2f}")
                    kpi3.metric(label="Best Day for Profit", value=f"{best_profit_day.strftime('%b %d, %Y')}")

                    # Display the chart
                    st.subheader("Total Daily Profit Trend")
                    profit_chart_df = daily_profit.reset_index()
                    profit_chart_df.columns = ['Date', 'Profit']
                    
                    chart = alt.Chart(profit_chart_df).mark_line(point=True, color='green').encode(
                        x=alt.X('Date:T', title="Date"),
                        y=alt.Y('Profit:Q', title="Total Profit (₹)"),
                        tooltip=['Date:T', 'Profit:Q']
                    ).properties(
                        title='Daily Profit Over Time'
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No profit data to display. Please enter a unit price for at least one product.")

else:
    st.info("Please upload an Excel file to begin the analysis.")

