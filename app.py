import os
import streamlit as st
import streamlit_authenticator as stauth

import populate_vectorstore
from rag_pipeline import query_rag, final_rag_pipeline
from populate_vectorstore import update_vectorstore

def rag(prompt):
    response = query_rag(prompt)
    return response

# Function to list uploaded files
def list_uploaded_files():
    return [f for f in os.listdir("data") if f.endswith(".pdf")]

def final_rag(prompt):
    if(multi_query_on):
        response = final_rag_pipeline(prompt, context_type='multi-query')
    elif(hyde_on):
        response = final_rag_pipeline(prompt, context_type='hyde')
    else:
        response = final_rag_pipeline(prompt)
    return response
    

## STREAMLIT AND UI ##

st.header("RAG Assistant", divider="orange")

# Create a form
with st.form(key='input_form'):
    # Create a text input widget within the form
    user_input = st.text_input("Enter some text:")
    # Create a submit button within the form
    submit_button = st.form_submit_button(label='Submit')
    # Rag options 
    multi_query_on = st.toggle("Multi-Query(RRF)")
    recursive_decomp_on = st.toggle("Recursive-Decomp")
    hyde_on = st.toggle("HyDE")

# Check if the form is submitted
if submit_button:
    if user_input:
        # Call the processing function
        #result = rag(user_input)
        result = final_rag(user_input)

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
# Create a form
with st.sidebar.form(key='file_form'):
    # Create a text input widget within the form
    uadd_file_uploader = st.file_uploader("Upload File To Vectorstore", accept_multiple_files=True, label_visibility="collapsed")

    # Create a submit button within the form
    upload_button = st.form_submit_button(label='Upload')

if upload_button:
    if uadd_file_uploader is not None:
        # Save the uploaded file to the specified directory
        file_path = os.path.join("data", uadd_file_uploader[0].name)
        
        # Write the uploaded file to the file path
        with open(file_path, "wb") as f:
            f.write(uadd_file_uploader[0].getbuffer())
        
        st.success(f"File '{uadd_file_uploader[0].name}' has been uploaded successfully!")
    update_vectorstore()

st.sidebar.subheader("Uploaded Files")
uploaded_files = list_uploaded_files()
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("data/", file)
        st.sidebar.markdown(f"-[{file}]({file_path})", unsafe_allow_html=True)
else:
    st.sidebar.write("No files uploaded yet.")