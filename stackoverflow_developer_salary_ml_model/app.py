import streamlit as st
import sklearn as sk
from predict_page import show_predict_page
from explore_page import show_explore_page


pages = st.sidebar.selectbox("Explore Or Predict",("Predict","Explore"))

if pages == "Predict":
    show_predict_page()
else:
    show_explore_page()