# Streamlit cheat sheet
# https://docs.streamlit.io/develop/quick-reference/cheat-sheet

import streamlit as st
import numpy as np
import pandas as pd

# adding title of your web app
st.title('My first app')

# adding simple text
st.write('Sample Text')

# user input
number = st.slider('Pick a number', 0, 100)
st.write('You selected:', number)

# adding a button
if st.button('Greeting'):
    st.write('Hello')
else:
    st.write('Goodbye')

# adding a checkbox
if st.checkbox('Show Data'):
    st.write('Showing Data')
    df = pd.DataFrame({
        '1st column': [1, 2, 3, 4],
        '2nd column': [10, 20, 30, 40]
    })
    st.write(df)

# adding a radio button
genre = st.radio(
    "What's your favorite movie genre",
    ('Comedy', 'Drama', 'Documentary'))

# print the text of genre
st.write(f'You selected: {genre}')

# add a drop down list
# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))
# st.write('You selected:', option)

# add a drop down list on the left sidebar
option = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

# add your whatsapp number
st.sidebar.text_input('Enter your whatsapp number')

# add a file uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# create a line plot
# Plotting
data = pd.DataFrame({
  'first column': list(range(1, 11)),
  'second column': np.arange(number, number + 10)
})
st.line_chart(data)