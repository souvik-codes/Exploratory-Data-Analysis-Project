# ---------------------------- LIBRARIES ----------------------------
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import os
import time
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="✨ AI-Powered Sales Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4B0082;'>📊 AI-Powered Sales Dashboard 2019</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;color:gray;'>Dynamic insights with forecasting, clustering, bundling & anomaly detection</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------- CACHING UTILS ----------------------------
@st.cache_data
def safe_read_csv(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    header_row = list(df.columns)
    mask = ~(df.apply(lambda row: all([str(row[c]).strip() == str(c).strip() for c in header_row]), axis=1))
    df = df[mask].copy()
    return df

@st.cache_data
def get_city_coords(cities):
    geolocator = Nominatim(user_agent="geo")
    coords = {}
    for city in cities:
        try:
            loc = geolocator.geocode(city)
            if loc:
                coords[city] = (loc.latitude, loc.longitude)
            else:
                coords[city] = None
        except:
            coords[city] = None
    return coords

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_data():
    if os.path.exists("Sales_2019.csv"):
        raw = safe_read_csv("Sales_2019.csv")
    else:
        months = ['January','February','March','April','May','June',
                  'July','August','September','October','November','December']
        frames = []
        for m in months:
            p = f"datasets/Sales_{m}_2019.csv"
            if os.path.exists(p):
                frames.append(safe_read_csv(p))
        if not frames:
            st.error("❌ No data files found.")
            st.stop()
        raw = pd.concat(frames, ignore_index=True)

    raw.columns = [c.strip() for c in raw.columns]

    colmap = {}
    for c in raw.columns:
        lc = c.lower()
        if 'order id' in lc: colmap[c] = 'Order ID'
        elif 'order date' in lc: colmap[c] = 'Order Date'
        elif 'purchase address' in lc: colmap[c] = 'Purchase Address'
        elif 'quantity' in lc: colmap[c] = 'Quantity Ordered'
        elif 'price' in lc: colmap[c] = 'Price Each'
        elif 'product' in lc: colmap[c] = 'Product'
        elif 'sales' in lc: colmap[c] = 'Sales'
        elif 'month' in lc: colmap[c] = 'Month'
    raw = raw.rename(columns=colmap)

    expected_cols = ['Order ID','Order Date','Purchase Address','Quantity Ordered','Price Each','Product','Sales','Month']
    for c in expected_cols:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw.copy()
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce').fillna(0).astype(int)
    df['Price Each'] = pd.to_numeric(df['Price Each'], errors='coerce').fillna(0.0).astype(float)
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    if df['Sales'].isna().sum():
        df['Sales'] = df['Quantity Ordered'] * df['Price Each']
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Hour'] = df['Order Date'].dt.hour.fillna(-1).astype(int)
    df['Day'] = df['Order Date'].dt.day.fillna(0).astype(int)
    df['DayOfWeek'] = df['Order Date'].dt.day_name().fillna('Unknown')
    df['Week'] = df['Order Date'].dt.isocalendar().week.astype('Int64')

    def extract_city(addr):
        try:
            if pd.isna(addr):
                return "Unknown"
            parts = str(addr).split(',')
            if len(parts) >= 2:
                return parts[1].strip()
            return parts[0].strip()
        except:
            return "Unknown"
    df['City'] = df['Purchase Address'].apply(extract_city)

    if df['Month'].isna().all():
        df['Month'] = df['Order Date'].dt.month_name().fillna("Unknown")

    df['Product'] = df['Product'].fillna("Unknown Product")
    df['Order ID'] = df['Order ID'].fillna(method='ffill').astype(str)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

merged_df = load_data()

# ---------------------------- FILTERS ----------------------------
st.sidebar.header("🔍 Filters")
months = merged_df['Month'].dropna().unique().tolist()
cities = merged_df['City'].dropna().unique().tolist()
month_filter = st.sidebar.multiselect("Select Month(s)", months, default=months)
city_filter = st.sidebar.multiselect("Select City(s)", cities, default=cities)

df_filtered = merged_df.copy()
if month_filter:
    df_filtered = df_filtered[df_filtered['Month'].isin(month_filter)]
if city_filter:
    df_filtered = df_filtered[df_filtered['City'].isin(city_filter)]

if df_filtered.empty:
    st.warning("⚠️ No data after filter selection.")
    st.stop()

# ---------------------------- RFM COMPUTATION ----------------------------
rfm = df_filtered.groupby('Purchase Address').agg({
    'Order ID':'nunique',
    'Sales':'sum',
    'Order Date': lambda x: (df_filtered['Order Date'].max() - x.max()).days
})
rfm.columns = ['Frequency (# Orders)','Monetary ($)','Recency (Days)']

# Stable ranking
rfm['R_rank'] = (4 - (rfm['Recency (Days)'].rank(method='first', pct=True) * 4).apply(np.ceil)).astype(int).clip(1,4)
rfm['F_rank'] = (rfm['Frequency (# Orders)'].rank(method='first', pct=True) * 4).apply(np.ceil).astype(int).clip(1,4)
rfm['M_rank'] = (rfm['Monetary ($)'].rank(method='first', pct=True) * 4).apply(np.ceil).astype(int).clip(1,4)
rfm['RFM Score'] = rfm[['R_rank','F_rank','M_rank']].sum(axis=1)

# ---------------------------- KPI ANIMATION ----------------------------
def animate_metric(container, label, value, prefix="", duration=1.0, steps=25):
    placeholder = container.empty()
    final = int(value)
    for i in range(steps):
        progress = (i + 1) / steps
        current = int(final * progress)
        placeholder.metric(label=label, value=f"{prefix}{current:,}")
        time.sleep(duration / steps)
    placeholder.metric(label=label, value=f"{prefix}{final:,}")

# ---------------------------- KPIs DISPLAY ----------------------------
st.subheader("✨ Key Metrics")
c1, c2, c3 = st.columns(3)
animate_metric(c1, "💰 Total Sales", df_filtered['Sales'].sum(), prefix="$")
animate_metric(c2, "🧾 Total Orders", df_filtered['Order ID'].nunique())
animate_metric(c3, "📦 Total Quantity", df_filtered['Quantity Ordered'].sum())
st.markdown("---")

# ---------------------------- Helper ----------------------------
def plot_bar(x, y, x_title="", y_title=""):
    fig = px.bar(x=x, y=y, labels={'x': x_title, 'y': y_title}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- TABS ----------------------------
tabs = st.tabs([
    "📈 Sales Analysis", 
    "🛍️ Product Analysis", 
    "🏆 Customer & Basket Insight",
    "📦 Bundle Analysis", 
    "🌎 Geo Heatmap", 
    "🔎 Clustering & Elasticity",
    "📑 Summary Insights"
])

# ---------------------------- TAB 1 — SALES ANALYSIS ----------------------------
with tabs[0]:
    st.subheader("💰 Total Sales Per Month")
    sales_per_month = df_filtered.groupby('Month')['Sales'].sum()
    plot_bar(sales_per_month.index, sales_per_month.values, "Month", "Revenue")

    st.subheader("📅 Weekly Revenue Trend")
    weekly = df_filtered.groupby("Week")['Sales'].sum()
    fig = px.line(x=weekly.index, y=weekly.values, template="plotly_white", labels={'x':"Week", 'y':"Revenue"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕒 Hour-wise Purchase Pattern")
    hour_sales = df_filtered.groupby("Hour")['Order ID'].count()
    plot_bar(hour_sales.index, hour_sales.values, "Hour", "Orders")

    st.subheader("📆 Weekend vs Weekday")
    weekend = df_filtered[df_filtered['DayOfWeek'].isin(["Saturday","Sunday"])]
    weekday = df_filtered[~df_filtered['DayOfWeek'].isin(["Saturday","Sunday"])]
    st.write(f"📦 Weekend Total Sales: **${weekend['Sales'].sum():,.2f}**")
    st.write(f"📦 Weekday Total Sales: **${weekday['Sales'].sum():,.2f}**")

# ---------------------------- TAB 2 — PRODUCT ANALYSIS ----------------------------
with tabs[1]:
    st.subheader("📦 Quantity Sold Per Product")
    qty = df_filtered.groupby("Product")['Quantity Ordered'].sum().sort_values()
    plot_bar(qty.index, qty.values, "Product", "Quantity")

    st.subheader("💰 Revenue Per Product")
    prod_rev = df_filtered.groupby("Product")['Sales'].sum().sort_values()
    plot_bar(prod_rev.index, prod_rev.values, "Product", "Revenue")

    st.subheader("📈 Price vs Quantity Scatter + Trendline")
    fig = px.scatter(df_filtered, x="Price Each", y="Quantity Ordered", trendline="ols", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Pair Correlation")
    fig = sns.pairplot(df_filtered[['Quantity Ordered','Price Each','Sales']])
    st.pyplot(fig)

# ---------------------------- TAB 3 — CUSTOMER & BASKET ----------------------------
with tabs[2]:
    st.subheader("🏆 RFM Customer Segmentation")
    st.write("🎯 **Top 15 Loyal & High-Valued Customers**")
    st.dataframe(rfm.sort_values("RFM Score", ascending=False).head(15))

    st.write("⚠️ **At-Risk Customers (Low Score)**")
    st.dataframe(rfm.sort_values("RFM Score", ascending=True).head(15))

    st.markdown("---")
    st.subheader("🧺 Market Basket Association (Most Common Pairs)")
    df_group = df_filtered.groupby("Order ID")['Product'].apply(list)
    pair_count = {}
    for prods in df_group:
        for a, b in combinations(sorted(prods), 2):
            pair_count[(a, b)] = pair_count.get((a, b), 0) + 1

    assoc = pd.DataFrame([{"Pair": k, "Count": v} for k, v in pair_count.items()])
    assoc = assoc.sort_values("Count", ascending=False).head(15)
    st.dataframe(assoc.reset_index(drop=True))

    st.markdown("---")
    st.subheader("⚠️ Quick Anomaly Insight")
    try:
        iso = IsolationForest(contamination=0.01, random_state=0)
        df_filtered['Anomaly'] = iso.fit_predict(df_filtered[['Sales']])
        anomalies = df_filtered[df_filtered['Anomaly'] == -1]
        st.write(f"💡 Found **{len(anomalies)}** unusual purchases (price spikes, returns, fraud, typos)")
        st.dataframe(anomalies[['Order ID','Product','Price Each','Sales']].head(10))
    except:
        st.info("Not enough data for anomaly model.")

# ---------------------------- TAB 4 — PRODUCT BUNDLING ----------------------------
with tabs[3]:
    st.subheader("🧺 Product Bundling (Frequently Bought Together)")
    df_group = df_filtered.groupby("Order ID")['Product'].apply(list)
    combos = {}
    for order_products in df_group:
        for combo in combinations(sorted(order_products), 2):
            combos[combo] = combos.get(combo, 0) + 1

    bundle_df = pd.DataFrame([{"Combo": k, "Frequency": v} for k,v in combos.items()])
    bundle_df = bundle_df.sort_values("Frequency", ascending=False).head(20)
    st.dataframe(bundle_df.reset_index(drop=True))

# ---------------------------- TAB 5 — GEO HEATMAP ----------------------------
with tabs[4]:
    st.subheader("🌍 Geographic Sales Heatmap")
    sales_city = df_filtered.groupby("City")['Sales'].sum()
    m = folium.Map(location=[37, -95], zoom_start=4)
    city_coords = get_city_coords(df_filtered['City'].unique())
    for city, val in sales_city.items():
        loc = city_coords.get(city)
        if loc:
            folium.CircleMarker(location=[loc[0], loc[1]],
                                radius=10, popup=f"{city}: ${val:,.0f}",
                                color='crimson', fill=True).add_to(m)
    st_folium(m, width=700)

# ---------------------------- TAB 6 — CLUSTERING & ELASTICITY ----------------------------
with tabs[5]:
    st.subheader("👥 Customer Persona Segmentation")
    st.write("### 🧮 RFM Base Table (Customer Level)")
    st.dataframe(rfm.sort_values("Monetary ($)", ascending=False))

    scaler = MinMaxScaler()
    scaled_rfm = scaler.fit_transform(rfm[['Recency (Days)','Frequency (# Orders)','Monetary ($)']])
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled_rfm)

    persona_map = {0:"💎 VIP Customers",1:"🛒 Loyal Buyers",2:"🥱 At-Risk Customers",3:"🌱 New Customers"}
    st.write("### 🧭 Persona Interpretation")
    for k,v in persona_map.items():
        st.write(f"**Cluster {k} → {v}**")

    profile = rfm.groupby('Cluster').agg({
        'Recency (Days)':'mean',
        'Frequency (# Orders)':'mean',
        'Monetary ($)':'mean',
        'RFM Score':'mean'
    }).reset_index()
    st.write("### 📊 Cluster Profiles")
    st.dataframe(profile)

    st.subheader("📈 Persona Visualization (Recency vs Monetary)")
    fig = px.scatter(
        rfm,
        x="Recency (Days)",
        y="Monetary ($)",
        color="Cluster",
        hover_data=['Frequency (# Orders)', 'RFM Score'],
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Persona Distribution by Size")
    count_df = rfm['Cluster'].value_counts().reset_index()
    count_df.columns = ['Cluster','Count']
    fig = px.pie(count_df, names='Cluster', values='Count', title="Persona Group Sizes", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Anomaly Detection — High Value / Unusual")
    iso = IsolationForest(contamination=0.04, random_state=42)
    rfm['Anomaly'] = iso.fit_predict(rfm[['Monetary ($)', 'Frequency (# Orders)']])
    flagged = rfm[rfm['Anomaly'] == -1]
    st.write(f"🚨 **Detected {flagged.shape[0]} Suspicious Customers**")
    st.dataframe(flagged[['Monetary ($)', 'Frequency (# Orders)', 'Recency (Days)','RFM Score','Cluster']])

# ---------------------------- TAB 7 — EXECUTIVE SUMMARY ----------------------------
with tabs[6]:
    st.subheader("📑 Executive Summary — Snapshot Insights")

    # --- Core insights ---
    top_month = df_filtered.groupby('Month')['Sales'].sum().idxmax()
    top_city = df_filtered.groupby('City')['Sales'].sum().idxmax()
    top_product = df_filtered.groupby('Product')['Sales'].sum().idxmax()
    best_hour = df_filtered.groupby('Hour')['Order ID'].count().idxmax()

    weekend_rev = df_filtered[df_filtered['DayOfWeek'].isin(["Saturday","Sunday"])]['Sales'].sum()
    weekday_rev = df_filtered[~df_filtered['DayOfWeek'].isin(["Saturday","Sunday"])]['Sales'].sum()
    
    best_combo = pd.DataFrame(bundle_df).iloc[0]["Combo"]
    
    loyal_segment = rfm.sort_values("RFM Score", ascending=False).head(1).index[0]

    st.markdown(f"""
### 🧾 Business Takeaways

- 📆 **Top Revenue Month:** {top_month}  → prioritize campaigns
- 🌆 **Best Performing City:** {top_city} → strong distribution channel
- 📦 **Best Selling Product:** {top_product} → stock priority
- ⏰ **Peak Buying Hour:** {best_hour}:00 → run ads & promos then
- 👥 **Top Loyal Customer Region:** {loyal_segment}
- ➗ **Weekend vs Weekday:**  
  - Weekend Revenue: **${weekend_rev:,.0f}**  
  - Weekday Revenue: **${weekday_rev:,.0f}**
- 🔗 **Most Frequently Purchased Combo:** {best_combo[0]} + {best_combo[1]}

---

### 🏬 What to Do Next?

- 💰 Cross-sell **bundles** around most frequent combos  
- 🔥 Boost stock of **top 3 selling products**  
- 📍 Run city-wise **localized promotions** in {top_city}  
- 📊 Send personalized offers to RFM **premium clusters**  
- ⏳ Target ads at **peak buying hours** ({best_hour}:00)

---

### 🚀 Quick Wins
- Run **flash discounts** right before peak hours
- Push complementary products based on basket pairs
- Target weekend buyers with premium bundles
    """)

    st.success("📌 Use this summary to pitch insights to management or include in reports 📊")    