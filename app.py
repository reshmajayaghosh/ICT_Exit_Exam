import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))


df = pd.read_csv("D:\ICT\Exit Exam\Bengaluru_House_Data.csv")


df = df.dropna(subset=['location', 'size'])
df['bhk'] = df['size'].str.split(" ").str[0].astype(int)


st.title("🏠 Bangalore House Price Prediction")


location = st.selectbox("Select Location", df['location'].unique())

bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)

sqft = st.number_input("Total Sqft", min_value=300.0)

bath = st.number_input("Bathrooms", min_value=1.0)

balcony = st.number_input("Balcony", min_value=0.0)


if st.button("Predict Price"):
    
    if sqft <= 0:
        st.error("Sqft must be greater than 0")
    
    else:
       
        input_data = pd.DataFrame({
            'location': [0],   
            'total_sqft': [sqft],
            'bath': [bath],
            'balcony': [balcony],
            'bhk': [bhk]
        })

        try:
            prediction = model.predict(input_data)
            st.success(f"Estimated Price: {round(prediction[0],2)} Lakhs")
        except:
            st.error("Error in prediction ")


st.subheader("Top 5 Expensive Locations")

filtered = df[df['bhk'] == bhk]

top_locations = (
    filtered.groupby('location')['price']
    .mean()
    .sort_values(ascending=False)
    .head(5)
)

st.bar_chart(top_locations)