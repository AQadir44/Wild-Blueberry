import streamlit as st
import pickle
import numpy as np

# import the model
xg = pickle.load(open("model/lreg_bbry_tuned_model.pkl",'rb'))
rf = pickle.load(open("model/rf_bbry_tuned_model.pkl",'rb'))

st.title("Laptop Predictor")


clone_size = st.number_input("enter clone size: " , max_value=40.0 , key = "clone_size")

bumbles =  st.number_input("enter bumbles" , key = "bumbles")

andrena =  st.number_input("enter andrena: ", key = "andrena")

osmia = st.number_input("enter osmia ", key = "osmia")

AverageOfUpperTRange = st.number_input("enter AverageOfUpperTRange ", key = "upper_range")

AverageOfLowerTRange = st.number_input("enter AverageOfLowerTRange: ", key = "lower_range")

AverageRainingDays = st.number_input("enter AverageRainingDays: ", key = "raining_day")


submit = st.button('Predict Yield')
if submit:
    query = np.array([clone_size , bumbles , andrena , osmia , AverageOfUpperTRange , AverageOfLowerTRange , AverageRainingDays])

    query = query.reshape(1,7)

    st.title("The predicted price from xgboost of this configuration is " + str(int(xg.predict(query)[0])))

    st.title("The predicted yield from randomforest value of these parameter is " + str(int((rf.predict(query)[0]))))
