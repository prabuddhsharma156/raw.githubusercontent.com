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

# --- Initialize Session State ---
# THE FIX: Initialize session state variables to store results across reruns.
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'profit_results' not in st.session_state:
    st.session_state.profit_results = None

# --- Data Loading and Caching ---
@st.cache_data
def load_data(uploaded_file):
    """
    Loads, cleans, and prepares sales data from a user-uploaded file.
    """
    try
        # Data loading and cleaning logic... (omitted for brevity, as it is correct)
        header_df = pd.read_excel(uploaded_file, header=None, nrows=2, engine='openpyxl')
        header_df.iloc[0] = header_df.iloc[0].ffill()
        header_df = header_df.fillna('')
        combined_headers = header_df.iloc[0].astype(str) + ' ' + header_df.iloc[1].astype(str)
        combined_headers = combined_headers.str.strip().str.replace(' ', '_').str.replace('__', '_').str.rstrip('_')
        df = pd.read_excel(uploaded_file, header=None, skiprows=2, engine='openpyxl')
        num_columns_read = df.shape[1]
        df.columns = combined_headers[:num_columns_read]
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        if df.empty:
            st.warning("Warning: No valid data rows found after cleaning.")
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
        page = st.sidebar.radio("Go to", ("Executive Summary", "What-If Simulation", "Detailed Product Analysis", "Sales Forecasting", "Profit Analysis"))
        st.sidebar.markdown("---")
        st.sidebar.info("This dashboard provides automated analysis for sales and inventory management.")

        # --- Pre-computation for Analytics ---
        total_sales = df.sum().sort_values(ascending=False)
        mean_sales = df.mean()
        std_sales = df.std()
        coefficient_of_variation = (std_sales / mean_sales).fillna(0).sort_values(ascending=False)
        
        # --- Page 1: Executive Summary ---
        if page == "Executive Summary":
            st.header("Executive Summary")
            # ... (Executive Summary code remains the same)

        # --- Page 2: What-If Simulation ---
        elif page == "What-If Simulation":
            st.header("🔬 'What-If' Inventory Simulation")
            st.markdown("Test different inventory strategies to see how they perform against simulated demand.")

            col1, col2, col3 = st.columns(3)
            with col1:
                product = st.selectbox("Select Product for Simulation", df.columns, key="sim_product")
            with col2:
                initial_stock = st.number_input("Initial Stock Level", min_value=0, value=3000, step=100, key="sim_stock")
            with col3:
                reorder_point = st.number_input("Reorder Point", min_value=0, value=1000, step=50, key="sim_reorder")

            if st.button("▶️ Run Simulation"):
                # THE FIX: Calculate and SAVE results to session state on button click.
                avg_sales_sim = mean_sales.get(product, 0)
                std_dev_sales_sim = std_sales.get(product, 0)
                day = 1
                current_stock = initial_stock
                reorder_triggered = False
                simulation_log = []
                while day <= 30:
                    daily_sales = max(0, int(np.random.normal(avg_sales_sim, std_dev_sales_sim)))
                    if daily_sales > current_stock:
                        simulation_log.append({"Day": day, "Activity": f"DEMAND ({daily_sales}) > STOCK ({current_stock}). STOCK OUT!", "Stock Level": 0})
                        break
                    current_stock -= daily_sales
                    activity = f"Sold {daily_sales} units."
                    if current_stock <= reorder_point and not reorder_triggered:
                        activity += " -> Reached reorder point!"
                        reorder_triggered = True
                    simulation_log.append({"Day": day, "Activity": activity, "Stock Level": current_stock})
                    day += 1
                
                st.session_state.simulation_results = {
                    'log': pd.DataFrame(simulation_log).set_index("Day"),
                    'final_stock': current_stock,
                    'product': product
                }

            # THE FIX: Display results from session state on EVERY rerun.
            if st.session_state.simulation_results is not None:
                results = st.session_state.simulation_results
                st.markdown("---")
                st.subheader(f"Simulation Log for '{results['product']}'")
                st.dataframe(results['log'], use_container_width=True)
                st.markdown("---")
                st.subheader("Simulation Summary")
                if results['final_stock'] > 0:
                    st.success(f"The inventory policy was robust. Final stock: {results['final_stock']} units.")
                else:
                    st.error("A STOCK OUT occurred. This inventory policy is risky.")

        # --- Page 3: Detailed Product Analysis ---
        elif page == "Detailed Product Analysis":
            st.header("📊 Detailed Product Analysis")
            # ... (Detailed Product Analysis code remains the same)
        
        # --- Page 4: Sales Forecasting ---
        elif page == "Sales Forecasting":
            st.header("🔮 Sales Forecasting")
            st.markdown("Generate a sales forecast for any product based on its recent performance.")

            col1, col2 = st.columns(2)
            with col1:
                product_to_forecast = st.selectbox("Select Product to Forecast", df.columns, key="forecast_product")
            with col2:
                forecast_days = st.slider("Number of Days to Forecast", min_value=7, max_value=90, value=30, step=7, key="forecast_days")

            if st.button("📈 Generate Forecast"):
                # THE FIX: Calculate and SAVE forecast to session state.
                last_14_days_avg = df[product_to_forecast].tail(14).mean()
                last_date = df.index.max()
                future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                forecast_values = [int(last_14_days_avg)] * forecast_days
                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Sales': forecast_values}).set_index('Date')
                
                historical_data = df[[product_to_forecast]].reset_index()
                historical_data.columns = ['Date', 'Sales']
                historical_data['Type'] = 'Historical'
                forecast_data = forecast_df.reset_index()
                forecast_data.columns = ['Date', 'Sales']
                forecast_data['Type'] = 'Forecast'
                combined_chart_df = pd.concat([historical_data, forecast_data])

                st.session_state.forecast_results = {
                    'log': forecast_df,
                    'chart_data': combined_chart_df,
                    'product': product_to_forecast,
                    'days': forecast_days
                }

            # THE FIX: Display forecast from session state on EVERY rerun.
            if st.session_state.forecast_results is not None:
                results = st.session_state.forecast_results
                st.markdown("---")
                st.subheader(f"Forecast for {results['product']}")
                st.dataframe(results['log'])
                chart = alt.Chart(results['chart_data']).mark_line(point=True).encode(
                    x=alt.X('Date:T', title="Date"),
                    y=alt.Y('Sales:Q', title="Units Sold"),
                    color=alt.Color('Type:N', title="Data Type"),
                    strokeDash=alt.condition(alt.datum.Type == 'Forecast', alt.value([5, 5]), alt.value([0])),
                    tooltip=['Date', 'Sales', 'Type']
                ).properties(title=f"Historical Sales vs. {results['days']}-Day Forecast").interactive()
                st.altair_chart(chart, use_container_width=True)
        
        # --- Page 5: Profit Analysis ---
        elif page == "Profit Analysis":
            st.header("💰 Profit Analysis")
            st.markdown("Enter the cost and price for each product to calculate and visualize your daily profit.")
            
            with st.form(key='profit_form'):
                st.subheader("Enter Product Financials")
                financial_data = {}
                col1, col2 = st.columns(2)
                for i, product_name in enumerate(df.columns):
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.markdown(f"**{product_name}**")
                        unit_cost = st.number_input(f"Unit Cost (₹)", key=f"cost_{product_name}", min_value=0.0, step=0.5, format="%.2f")
                        unit_price = st.number_input(f"Unit Price (₹)", key=f"price_{product_name}", min_value=0.0, step=0.5, format="%.2f")
                        financial_data[product_name] = {'cost': unit_cost, 'price': unit_price}
                submit_button = st.form_submit_button(label='📊 Calculate and Analyze Profit')

            if submit_button:
                # THE FIX: Calculate and SAVE profit analysis to session state.
                profit_df = df.copy()
                for product_name, data in financial_data.items():
                    if data['price'] > 0:
                        unit_profit = data['price'] - data['cost']
                        profit_df[product_name] = profit_df[product_name] * unit_profit
                    else:
                        profit_df[product_name] = 0
                daily_profit = profit_df.sum(axis=1)
                st.session_state.profit_results = daily_profit

            # THE FIX: Display profit results from session state on EVERY rerun.
            if st.session_state.profit_results is not None:
                daily_profit = st.session_state.profit_results
                st.markdown("---")
                st.subheader("Profit Calculation Results")
                if daily_profit.sum() > 0:
                    total_profit = daily_profit.sum()
                    avg_daily_profit = daily_profit.mean()
                    best_profit_day = daily_profit.idxmax()
                    st.subheader("Key Profit Metrics")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric(label="Total Profit", value=f"₹{total_profit:,.2f}")
                    kpi2.metric(label="Average Daily Profit", value=f"₹{avg_daily_profit:,.2f}")
                    kpi3.metric(label="Best Day for Profit", value=f"{best_profit_day.strftime('%b %d, %Y')}")
                    st.subheader("Total Daily Profit Trend")
                    profit_chart_df = daily_profit.reset_index()
                    profit_chart_df.columns = ['Date', 'Profit']
                    chart = alt.Chart(profit_chart_df).mark_line(point=True, color='green').encode(
                        x=alt.X('Date:T', title="Date"),
                        y=alt.Y('Profit:Q', title="Total Profit (₹)"),
                        tooltip=['Date:T', 'Profit:Q']
                    ).properties(title='Daily Profit Over Time').interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("No profit data to display. Please enter a unit price for at least one product.")
else:
    st.info("Please upload an Excel file to begin the analysis.")

