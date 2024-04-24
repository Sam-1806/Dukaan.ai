import pandas as pd
import numpy as np
import streamlit as st
import plotly as px
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Set page title and favicon
st.set_page_config(page_title="Dukaan.ai Dashboard", page_icon="",layout="wide")

excel_file = "Superstore.xlsx"
sheet_name = "Superstore"

# Read the Excel file & Model
df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols='A:U')
with open('model_discount .pkl', 'rb') as model_file:
    model_discount, scaler = pickle.load(model_file)

#<--------------------------------Titile-------------------------------------->
st.markdown("<div style='text-align: center;'><h1 style='color:black;'>Dukaan Dashboard Analytics</h1></div>", unsafe_allow_html=True)







#<------------------------Display Net Profit and Revenue------------------------------->

# Calculate total profits and revenue
total_profit = df['Profit'].sum()
formatted_profit = "{:,.2f}".format(total_profit)

total_sales = df['Sales'].sum()
formatted_sales = "{:,.2f}".format(total_sales)

# Create a 2-column layout for total profit and revenue
col1, col2 = st.columns(2)

# Total profits on the left side
with col1:
    st.subheader("Total Profits")
    st.write(f'<p style="color:green; font-size:35px; border: 2px solid black; padding: 10px;">Rs. {formatted_profit}</p>', unsafe_allow_html=True)

# Total revenue on the right side
with col2:
    st.subheader("Total Revenue")
    st.write(f'<p style="color:grey; font-size:35px; border: 2px solid black; padding: 10px;">Rs. {formatted_sales}</p>', unsafe_allow_html=True)




#<---------------------------Display  Category Breakdown and Monthly Sales------------------------>

# Convert 'Order Date' column to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month from 'Order Date' and create a new column
df['Month'] = df['Order Date'].dt.month
df_2017 = df[df['Order Date'].dt.year == 2017]
# Calculate monthly sales
monthly_sales_2017 = df_2017.groupby(df_2017['Order Date'].dt.month)['Sales'].sum().reset_index()
monthly_sales_2017.rename(columns={'Order Date': 'Month'}, inplace=True)

fig = px.line(monthly_sales_2017, x='Month', y='Sales')
fig.update_xaxes(title='Month', tickvals=list(range(1, 13)))
fig.update_yaxes(title='Sales')

# Create a 2-row layout for category breakdown and monthly sales
row1, row2 = st.columns(2)

# Category breakdown on the left side
with row1:
    st.title("Category Analysis")
    # Define data
    category = df['Category'].unique().tolist()

    # Create pie chart for category breakdown with matching size
    category_pie_chart = px.pie(df, names='Category', title='Category Breakdown', width=450, height=400)
    st.plotly_chart(category_pie_chart)

# Monthly sales on the right side
with row2:
    st.title("Monthly Sales Analysis for 2017")
    st.plotly_chart(fig)






#<---------------------------Display Histogram from net Profit and net Sales----------------------->

df['Order Date'] = pd.to_datetime(df['Order Date']) 
df['Year'] = df['Order Date'].dt.year



sales_histogram = px.histogram(df, x='Year', y='Sales',
                                histfunc='sum')
profit_histogram = px.histogram(df, x='Year', y='Profit',
                                histfunc='sum')

sales_histogram.update_traces(marker_line_width=0.5)
profit_histogram.update_traces(marker_line_width=0.5)

sales_histogram.update_xaxes(tickvals=df['Year'].unique(), tickmode='array')
profit_histogram.update_xaxes(tickvals=df['Year'].unique(), tickmode='array')

# Display histograms in left and right columns
col1, col2 = st.columns(2)

with col1:
    st.title("Yearly Total Sales")
    st.plotly_chart(sales_histogram)

with col2:
    st.title("Yearly Total Profit")
    st.plotly_chart(profit_histogram)




#<---------------------------Display Scatter Plot ofr Profit v Discount & Pie Chart for Sales StateWise---------------------->

scatter_plot = px.scatter(df, x='Discount', y='Profit')
net_sales_by_state = df.groupby('State')['Sales'].sum().reset_index()


pie_chart = px.pie(net_sales_by_state, values='Sales', names='State')
col1, col2 = st.columns(2)

with col1:
    st.title("Discount and Profit Analysis")
    st.plotly_chart(scatter_plot)

# Display pie chart on the right side
with col2:
    st.title("State wise Sales")
    st.plotly_chart(pie_chart)



navigation = st.sidebar.radio("My Dukaan", ["Dashboard Analytics","Predictive Analytics"])


if navigation == "Dashboard Analytics":
    st.title("")
elif navigation == "Predictive Analytics":
#<--------------------Discount Optimization----------->
   
    # Streamlit UI
    st.title("Discount Optimization")

    # User input for features
    sales = st.number_input("Enter Sales:", value=1000)
    quantity = st.number_input("Enter Quantity:", value=10)
    profit = st.number_input("Enter Profit:", value=100)
    user_input_data = pd.DataFrame({'Sales': [sales], 'Quantity': [quantity], 'Profit': [profit]})
    if st.button("Predict Discount Rate"):
        try:

          # Scale the user input using the fitted StandardScaler
   
          user_input_scaled = scaler.transform(user_input_data)

          # Predict discount rate
          predicted_discount = model_discount.predict(user_input_scaled)[0]

          st.success(f"Predicted Discount Rate: {predicted_discount:.2f}")
        except Exception as e:
         st.error(f"Error: {e}")



st.title("Demand Forecasting")
# Load the trained model from the H5 file


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("C:/Users/Samruddhi/Downloads/demand-forecast.h5")
    return model

model = load_model()

categories = ["Shirt", "Shoes", "Groceries", "Pants", "Dresses"]
category_images = {
    "Shirt": "src\chart_1.png" ,
    "Shoes": "src\chart_4.png",
    "Groceries": "src\chart_5.png",
    "Pants": "src\chart_2.png",
    "Dresses": "src\chart_3.png",
}

selected_category = st.selectbox("Select Category", categories)
if selected_category in category_images:
    image_path = category_images[selected_category]
    st.image(image_path, caption=f"Forecasted Demand for {selected_category}", use_column_width=True)
else:
    st.write("No forecast available for the selected category.")


if st.button("Predict Demand"):
    st.write("Predicting demand...")

    
year = st.number_input("Year", min_value=2015, max_value=2025)
month = st.number_input("Month", min_value=1, max_value=12)

# Create a DataFrame with the input data
input_data = pd.DataFrame({'Year': [year], 'Month': [month]})
print("Shape of input data:", input_data.shape)

# Ensure the input data has the correct shape
#input_data = input_data.to_numpy().reshape((1, 12, 1))


# Make predictions
#if st.button("Predict Demand"):
    #prediction = model.predict(input_data)
    #st.write(f"Predicted demand: {prediction[0][0]}")




