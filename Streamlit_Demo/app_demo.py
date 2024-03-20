import streamlit as st
import pandas as pd
import numpy as np

# adding title of your app
st.title('My First Streamlit APP')

# adding simple text
st.write('This is Simple Text')

# user input
number = st.slider('Pick a Number',0,100,10)

# print the text of number that is slider
st.write(f'You Selected: {number}')

# adding a button
if st.button('Greetings'):
    st.write('Hi, hello there')
else:
    st.write('Good Bye')
    
    
# add radio button
genre = st.radio(
    "Whats your Favorite Movie genre",
    ('Comedy','Drama','Documentry')
)
# print the genere
st.write(f'You Selected: {genre}')

# add a drop list
# option = st.selectbox(
#     'How would you like to be contacted',
#     ('Email','Home Phone','Mobile Phone')
# )
# print the contact option
#st.write(f'You Selected: {option}')


# add a drop down list to Sidebar
option_sidebar = st.sidebar.selectbox(
    'How would you like to be contacted',
    ('Email','Home Phone','Mobile Phone')
)
st.sidebar.write(f'You Selected: {option_sidebar}')


# add your whatsapp number
st.sidebar.text_input('Enter your Phone Number')

# add a file uploader
uploaded_file = st.sidebar.file_uploader('Choose a CSV File',type='csv')


# create a line plot
# plotting
data = pd.DataFrame({
    'first column':list(range(1,11)),
    'second column':np.arange(number, number + 10),
})
st.line_chart(data)



