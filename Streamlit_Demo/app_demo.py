import streamlit as st

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



