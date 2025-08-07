import streamlit as st
from langgraph_backend import chatbot
from langchain.core import HumanMessage

# ***************************** THREAD CONFIGURATION *****************************
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# ***************************** SESSION SETUP ************************************
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []

# ***************************** MAIN UI ******************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = 'messages'
            )
        )
    
    # add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})