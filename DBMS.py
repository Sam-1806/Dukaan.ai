import streamlit as st

# Set page title and favicon
st.set_page_config(page_title="Dukaan Dashboard", page_icon=":shopping_cart:")

# Define data
data = {
    "Total Profits": 36963,
    "Total Revenue": 437771,
    "Clothing": 3516,
    "Total Quantity Sold": 5615,
    "Popular Products": {
        "Clothing - Saree": 15,
        "Clothing - Hankerchief": 10,
        "Electronics - Printers": 12,
        "Clothing - Stole": 14,
        "Clothing - T-shirt": 11,
        "Electronics - Phones": 18,
        "Furniture - Furnishings": 8,
        "Furniture - Bookcases": 7,
        "Electronics - Electronic Games": 16,
        "Clothing -Shirt": 13,
        "Clothing - Skirt": 12,
        "Furniture - Chairs": 9,
        "Electronics - Accessories": 11,
        "Clothing - Kurti": 14,
        "Clothing - Trousers": 17,
        "Furniture - Tables": 10,
        "Clothing - Leggings": 16,
    },
    "Profit-Sub Category": {
        "Printers": 800,
        "Accessories": 700,
        "Bookcases": 600,
        "Saree": 500,
        "Tables": 400,
        "Trousers": 300,
    },
}

# Display data using Streamlit
st.title("Dukaan Dashboard")

# Create a sidebar navigation
navigation = st.sidebar.radio("Navigation", ["Home", "Retail Optimizer", "Out of Stock Detection", "Seasonal Products", "Apriori - Shelf", "CSV INPUT", "Optimization"])

if navigation == "Home":
    st.subheader("Welcome to Dukaan Dashboard")
elif navigation == "Retail Optimizer":
    st.subheader("Retail Optimizer")
elif navigation == "Out of Stock Detection":
    st.subheader("Out of Stock Detection")
elif navigation == "Seasonal Products":
    st.subheader("Seasonal Products")
elif navigation == "Apriori - Shelf":
    st.subheader("Apriori - Shelf")
elif navigation == "CSV INPUT":
    st.subheader("CSV INPUT")
elif navigation == "Optimization":
    st.subheader("Optimization")

# Display graphs using Streamlit
st.metric(label="Total Profits", value=data["Total Profits"])
st.metric(label="Total Revenue", value=data["Total Revenue"])

st.subheader("Clothing Sales")
st.bar_chart(data["Popular Products"])

st.subheader("Profit by Sub Category")
st.bar_chart(data["Profit-Sub Category"])
