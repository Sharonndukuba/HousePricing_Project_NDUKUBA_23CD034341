import streamlit as st
import pickle
import pandas as pd

with open('model/house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("House Price Prediction System")
st.write("Input house details to predict the price.")

overall_qual = st.number_input("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 10000, 1500)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 5000, 800)
garage_cars = st.number_input("Garage Cars Capacity", 0, 5, 2)
full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
year_built = st.number_input("Year Built", 1800, 2025, 2000)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]],
                              columns=['OverallQual','GrLivArea','TotalBsmtSF','GarageCars','FullBath','YearBuilt'])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
