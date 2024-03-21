import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns


# add title
st.title('Data Analyst Application')
st.subheader('This is a simple Data analysis application create by github.com/uzairafridi00')


# Create a dropdown list to choose a dataset
dataset_option = ['iris','titanic','tips','diamonds']
selected_dataset = st.selectbox('Select a dataset',dataset_option)

# uploaded the selected dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')
    
# button to upload custom dataset
uploaded_file = st.file_uploader('Upload a custom dataset', type=['csv','xlsx'])

if uploaded_file is not None:
    # process the uploaded file
    df = pd.read_csv(uploaded_file) # assuming the file upload is CSV


# display the dataset
st.write(df.head())


# display the number of rows and columns from selected one
st.write('Number of Rows:', df.shape[0])
st.write('Number of Columns:', df.shape[1])

# display the columns of selected dataset with datatypes
st.write('Column Names and Data Types:', df.dtypes)

# print the null values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null Values')
    

# display the summary statistics of the selected one
st.write('Summary Statistics:', df.describe())
    
