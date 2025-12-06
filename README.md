# 📊 Sales Analytics Dashboard (Streamlit)

An interactive Sales Analytics dashboard built using **Streamlit, Pandas, Plotly, and Scikit-learn** to generate deep and actionable business insights.

---

## 🚀 Features

### 📈 Exploratory Data Analysis (EDA)
- Monthly sales trends
- Product-level performance insights
- Hourly purchase behavior
- Revenue distribution across states/cities

### 👥 Customer Segmentation
- RFM scoring (Recency, Frequency, Monetary)
- KMeans clustering
- Persona tagging:
  - VIP Customers
  - Loyal Customers
  - Occasional Buyers
  - Lost Customers

### 🛍️ Market Basket Analysis
- Product bundling insights
- Association Rules (support, confidence, lift)
- Cross-selling strategy triggers

### 📦 Business Insights
- Best-selling products and bundles
- Geographic sales concentration
- Seasonal demand analysis

---

## 📁 Dataset Overview

Your CSV dataset should contain:

| Column            | Description                             |
|-------------------|-----------------------------------------|
| Order ID          | Unique transaction ID                   |
| Product           | Product purchased                       |
| Quantity Ordered  | Units ordered                           |
| Price Each        | Price per unit                          |
| Order Date        | Timestamp of purchase                   |
| Purchase Address  | Full customer address                   |
| Month             | Numeric/label month (derived)           |
| Hour              | Hour extracted from timestamp (derived) |
| City              | Parsed from address (derived)           |
| Sales             | Quantity Ordered × Price Each           |


---

## 🛠️ Technologies Used

- **Streamlit**
- **Pandas**
- **Numpy**
- **Plotly**
- **Scikit-Learn**
- **MLxtend**

---

## 📥 Installation Guide

### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
````

### 2️⃣ Run Locally

```bash
streamlit run app.py
```

---

## 📦 Requirements

- **streamlit**
- **pandas**
- **numpy**
- **matplotlib**
- **plotly**
- **scikit-learn**
- **mlxtend**

---

## 🔍 Insights Generated

### 📅 Time Analysis

* Month-wise revenue projection
* Hour-wise demand hotspot detection (for ad campaigns)

### 🏙️ City Analysis

* Best performing cities
* Underperforming regions → optimization opportunities

### 💰 Product Insights

* Margin drivers
* Low-selling inventory alerts
* Brand/category comparison

### 🧠 Customer Behavior

* Identify high-value customers (VIP)
* Re-engage cold/lost customers
* Personalized campaign groups

### 🛍️ Bundling Strategy

* Product combinations bought together
* Upsell/cross-sell recommendations

---

## 📊 Output Visualizations

* Line charts (time trends)
* Bar charts (city-wise performance)
* Pie charts (product distribution)
* Heatmaps (hour vs orders)
* RFM scatter clusters

---

## 📌 Business Applications

✔️ Demand forecasting
✔️ Inventory optimization
✔️ Personalized marketing
✔️ Cross-sell and upsell strategies
✔️ Customer retention planning

---

## 📝 Future Enhancements

* Customer Lifetime Value (CLV)
* Sales forecasting using Prophet
* Recommendation engine v2.0

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork and raise PRs.

---

## 📬 Contact

📧 Email: **[souvikbose337@gmail.com](mailto:souvikbose337@gmail.com)**
🔗 LinkedIn: **[https://www.linkedin.com/in/souvik-thecvguy/](https://www.linkedin.com/in/souvikbose-ai/)**
💻 GitHub: **[https://github.com/souvik-codes/](https://github.com/souvik-codes/)**
