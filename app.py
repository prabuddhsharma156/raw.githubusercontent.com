import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
        # THE FIX: Intelligently read the multi-level header to get real product names.
        # 1. Read the first two rows to capture the header information.
        header_df = pd.read_excel(uploaded_file, header=None, nrows=2, engine='openpyxl')
        
        # 2. Forward-fill the product names on the first row to handle merged cells.
        header_df.iloc[0] = header_df.iloc[0].ffill()
        
        # 3. Combine the two header rows into one. Replace NaN with an empty string.
        header_df = header_df.fillna('')
        combined_headers = header_df.iloc[0].astype(str) + ' ' + header_df.iloc[1].astype(str)
        
        # 4. Clean up the combined headers for better readability.
        combined_headers = combined_headers.str.strip().str.replace(' ', '_').str.replace('__', '_').str.rstrip('_')
        
        # 5. Read the actual data, skipping the header rows we just processed.
        df = pd.read_excel(uploaded_file, header=None, skiprows=2, engine='openpyxl')
        
        # 6. Assign the new, dynamic headers.
        num_columns_read = df.shape[1]
        df.columns = combined_headers[:num_columns_read]
        
        # 7. Rename the first column to 'Date' for consistency.
        df = df.rename(columns={df.columns[0]: 'Date'})

        # --- The rest of the cleaning process continues as before ---
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        if df.empty:
            st.warning("Warning: The uploaded file was loaded, but no valid data rows were found after cleaning. Please check the file's content and format.")
            return None
        
        # Identify numeric columns, safely excluding any 'Total' column
        numeric_cols = [col for col in df.columns if col != 'Date' and 'Total' not in col]

        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in numeric_cols:
            df[col] = df[col].astype(int)

        df.set_index('Date', inplace=True)
        # Drop any column that has 'Total' in its name
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
        page = st.sidebar.radio("Go to", ("Executive Summary", "What-If Simulation", "Detailed Product Analysis"))
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
                st.markdown("---")
                st.subheader(f"Simulation Log for '{product}'")
                
                avg_sales_sim = mean_sales.get(product, 0)
                std_dev_sales_sim = std_sales.get(product, 0)
                
                day = 1
                current_stock = initial_stock
                reorder_triggered = False
                simulation_log = []
                
                while day <= 30:
                    daily_sales = max(0, int(np.random.normal(avg_sales_sim, std_dev_sales_sim)))
                    log_entry = {}
                    if daily_sales > current_stock:
                        log_entry = {"Day": day, "Activity": f"DEMAND ({daily_sales}) > STOCK ({current_stock}). STOCK OUT!", "Stock Level": 0}
                        simulation_log.append(log_entry)
                        break
                    
                    current_stock -= daily_sales
                    activity = f"Sold {daily_sales} units."
                    if current_stock <= reorder_point and not reorder_triggered:
                        activity += " -> Reached reorder point!"
                        reorder_triggered = True
                    
                    log_entry = {"Day": day, "Activity": activity, "Stock Level": current_stock}
                    simulation_log.append(log_entry)
                    day += 1

                log_df = pd.DataFrame(simulation_log).set_index("Day")
                st.dataframe(log_df, use_container_width=True)

                st.markdown("---")
                st.subheader("Simulation Summary")
                if current_stock > 0:
                    st.success(f"The inventory policy was robust. Final stock: {current_stock} units.")
                else:
                    st.error("A STOCK OUT occurred. This inventory policy is risky.")

        # --- Page 3: Detailed Product Analysis ---
        elif page == "Detailed Product Analysis":
            st.header("📊 Detailed Product Analysis")
            st.markdown("Select any product to view its sales trend and key performance indicators.")

            product_to_view = st.selectbox("Select a Product", df.columns, key="detailed_product")
            
            if product_to_view:
                st.markdown("---")
                st.subheader(f"Sales Trend for {product_to_view}")
                
                product_df = df[[product_to_view]].reset_index()
                product_df.columns = ['Date', 'Units Sold']

                line_chart = alt.Chart(product_df).mark_line(point=True).encode(
                    x='Date:T',
                    y='Units Sold:Q',
                    tooltip=['Date:T', 'Units Sold:Q']
                ).interactive()
                
                avg_rule = alt.Chart(product_df).mark_rule(color='red', strokeDash=[3,3]).encode(
                    y=f'mean(\'Units Sold\'):Q'
                )
                
                st.altair_chart((line_chart + avg_rule), use_container_width=True)

                st.markdown("---")
                st.subheader("Key Analytics")
                
                LEAD_TIME_DAYS = 3
                Z_SCORE = 1.65
                reorder_point_val = (mean_sales[product_to_view] * LEAD_TIME_DAYS + (Z_SCORE * np.sqrt(LEAD_TIME_DAYS) * std_sales[product_to_view])).round(0)
                
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(label="Total Units Sold", value=int(total_sales[product_to_view]))
                kpi2.metric(label="Average Daily Sales", value=f"{mean_sales[product_to_view]:.2f}")
                kpi3.metric(label="Sales Volatility (CV)", value=f"{coefficient_of_variation[product_to_view]:.2f}")
                kpi4.metric(label="Recommended Reorder Point", value=int(reorder_point_val))
else:
    st.info("Please upload an Excel file to begin the analysis.")

