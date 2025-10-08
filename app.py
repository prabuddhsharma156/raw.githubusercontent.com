import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="Sampad Chemicals Inventory Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """
    Loads, cleans, and prepares the sales data from a file on GitHub.
    This function is cached to improve performance.
    """
    try:
        # THE FIX: Explicitly add sep=',' to ensure the parser uses the comma as a delimiter.
        df = pd.read_csv(file_path, header=None, skiprows=2, engine='python', encoding='latin1', sep=',')
        product_headers = [
            "Date", "Dr.Phenyle_Total", "Dr.Phenyle_450ML", "Dr.Phenyle_5L", "Dr.Phenyle_200ML",
            "DiamondBall_100PCS", "3DSOL_500ML", "NEEM_1L", "BlackCactus_450ML",
            "Hygiene_500ML", "Hygiene_1L", "Hygiene_5L", "BleachingPowder_500G",
            "DDF_1L", "JADU_1L", "LemonDrop_1L", "Angelo_500ML", "Angelo_1L"
        ]
        num_columns_read = df.shape[1]
        df.columns = product_headers[:num_columns_read]

        # A more robust cleaning sequence for dates.
        # 1. First, ensure the column is a string and strip any leading/trailing whitespace.
        df['Date'] = df['Date'].astype(str).str.strip()
        
        # 2. Now, convert to datetime, letting pandas infer the format.
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # 3. Drop any rows where the date conversion failed.
        df.dropna(subset=['Date'], inplace=True)

        if df.empty:
            st.warning("Warning: The data file was loaded, but no valid data rows were found after cleaning. Please check the file's content and format.")
            return None
        
        # Identify numeric columns, safely excluding 'Dr.Phenyle_Total' if it exists
        numeric_cols = df.columns.drop(['Date'])
        if 'Dr.Phenyle_Total' in numeric_cols:
            numeric_cols = numeric_cols.drop(['Dr.Phenyle_Total'])

        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in numeric_cols:
            df[col] = df[col].astype(int)

        df.set_index('Date', inplace=True)
        if 'Dr.Phenyle_Total' in df.columns:
            df.drop(columns=['Dr.Phenyle_Total'], inplace=True)
            
        df = df.loc[:, (df != 0).any(axis=0)]
        return df
    except FileNotFoundError:
        st.error(f"Error: The data file at the specified URL was not found. Please check the link in the code.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# --- Main Application Logic ---
# IMPORTANT: Replace the URL below with the raw URL of your data file from GitHub
data_file_path = 'https://raw.githubusercontent.com/prabuddhsharma156/raw.githubusercontent.com/main/Business%20Analytics%20CA.xlsx'
df = load_data(data_file_path)

if df is not None:
    # --- Sidebar ---
    st.sidebar.title("Dashboard Navigation")
    st.sidebar.markdown("Use the options below to explore the business analytics.")
    page = st.sidebar.radio("Go to", ("Executive Summary", "What-If Simulation", "Detailed Product Analysis"))
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Project by:**\n"
        "Prabuddh Sharma & Devleen Patnaik\n\n"
        "**Course:** MGNM801 Business Analytics"
    )

    # --- Pre-computation for Analytics ---
    total_sales = df.sum().sort_values(ascending=False)
    mean_sales = df.mean()
    std_sales = df.std()
    coefficient_of_variation = (std_sales / mean_sales).fillna(0).sort_values(ascending=False)
    
    # --- Page 1: Executive Summary ---
    if page == "Executive Summary":
        st.title("🧪 Sampad Chemicals: Executive Summary")
        st.markdown("This dashboard provides a high-level overview of product performance and demand stability.")
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
        st.title("🔬 'What-If' Inventory Simulation")
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
        st.title("📊 Detailed Product Analysis")
        st.markdown("Select any product to view its sales trend and key performance indicators.")

        product_to_view = st.selectbox("Select a Product", df.columns, key="detailed_product")
        
        if product_to_view:
            st.markdown("---")
            st.subheader(f"Sales Trend for {product_to_view}")
            
            # Create a dataframe for the selected product's sales
            product_df = df[[product_to_view]].reset_index()
            product_df.columns = ['Date', 'Units Sold']

            # Create the chart
            line_chart = alt.Chart(product_df).mark_line(point=True).encode(
                x='Date:T',
                y='Units Sold:Q',
                tooltip=['Date:T', 'Units Sold:Q']
            ).interactive()
            
            # Create the average line
            avg_rule = alt.Chart(product_df).mark_rule(color='red', strokeDash=[3,3]).encode(
                y=f'mean(\'Units Sold\'):Q'
            )
            
            st.altair_chart((line_chart + avg_rule), use_container_width=True)

            st.markdown("---")
            st.subheader("Key Analytics")
            
            # Pre-calculate reorder point
            LEAD_TIME_DAYS = 3
            Z_SCORE = 1.65 # 95% service level
            reorder_point_val = (mean_sales[product_to_view] * LEAD_TIME_DAYS + (Z_SCORE * np.sqrt(LEAD_TIME_DAYS) * std_sales[product_to_view])).round(0)
            
            # Display metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric(label="Total Units Sold", value=int(total_sales[product_to_view]))
            kpi2.metric(label="Average Daily Sales", value=f"{mean_sales[product_to_view]:.2f}")
            kpi3.metric(label="Sales Volatility (CV)", value=f"{coefficient_of_variation[product_to_view]:.2f}")
            kpi4.metric(label="Recommended Reorder Point", value=int(reorder_point_val))

