import streamlit as st
import sklearn as sk
import pickle
import numpy as np



# loading the model
def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write(""" ### We need some information to predict the Salary """)

    countries = (
        "United States of America",
        "India",
        "United Kingdom of Great Britain and Northern Ireland",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelor's",
        "Bachelor's degree",
        "Master's degree",
        "Post grad",
    )

    country = st.selectbox("Country",countries)
    education = st.selectbox("Education Level",education)
    experience = st.slider("Years of Experience",0,50,3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimates salary is ${salary[0]:.02f}")
 