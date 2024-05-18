import streamlit as st
import streamlit_authenticator as stauth

import populate_vectorstore
from rag_pipeline import query_rag

def rag(prompt):
    response = query_rag(prompt)
    return response

## STREAMLIT AND UI ##

st.header("Retreival Augmented Generation (RAG) Demo", divider="orange")

# Create a form
with st.form(key='my_form'):
    # Create a text input widget within the form
    user_input = st.text_input("Enter some text:")

    # Create a submit button within the form
    submit_button = st.form_submit_button(label='Submit')

# Check if the form is submitted
if submit_button:
    if user_input:
        # Call the processing function
        result = rag(user_input)
        # Display the result
        st.write(result)
    else:
        st.write("Please enter some text before submitting.")

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Use Content From',
    ('All Classes', 'ECON 2010', 'CS 3120', 'APMA 3100')
)

# Add a slider to the sidebar:
add_file_uploader = st.sidebar.file_uploader("Upload Files To Vectorstore", accept_multiple_files=True)