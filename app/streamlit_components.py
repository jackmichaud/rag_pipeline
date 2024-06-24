import os
import streamlit as st

def chatbot(chat_function: callable):
    col1, col2 = st.columns([4,2])

    # Title of the Streamlit app
    with col1:
        st.header("RAG Teaching Assistant", divider="orange")
    with col2:
        context_collection = context_selector()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            chat_response = chat_function(prompt, context_collection)
            response = st.write_stream(chat_response["response"])
            st.write("Sources: ", chat_response["sources"])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    

def context_selector():
    # Add a selectbox to the sidebar:
    data = [collection for collection in os.listdir(f"app/data")]
    data.insert(0, "All Indexes")

    selectbox = st.selectbox('Use Content From', data)
    return selectbox

def upload_selector():
    # Add a selectbox to the sidebar:
    data = [collection for collection in os.listdir(f"app/data")]
    data.append("New Collection")

    with st.sidebar:
        selectbox = st.selectbox('Upload Context To', data)

    return selectbox

def file_uploader(post_upload_function: callable):
    # Create a text input widget for file upload
    file = st.sidebar.file_uploader("Upload File To Vectorstore", accept_multiple_files=False)

    # Selector for collection
    collection = upload_selector()

    # If "New Collection" is selected, show the text input for the new collection name
    if collection == "New Collection":
        new_collection_name = st.sidebar.text_input("Enter new collection name")
        if new_collection_name:
            if len(new_collection_name) > 3 and len(new_collection_name) < 16:
                collection = new_collection_name
            else:
                st.sidebar.error("Collection name must be between 4 and 15 characters.")
                return

    upload_button = st.sidebar.button(label="Upload")

    if upload_button:
        if file is not None:
            print(file)
            # Save the uploaded file to the specified directory
            file_path = os.path.join("app/data", collection, file.name)
            
            if not os.path.isdir(os.path.join("app/data", collection)):
                os.makedirs(os.path.join("app/data", collection), exist_ok=True)
            
            # Write the uploaded file to the file path
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            st.sidebar.success(f"File '{file.name}' has been uploaded successfully!")
        if post_upload_function is not None:
            post_upload_function(collection)
    
    # Add a horizontal line below the collection selector
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

def file_explorer():
    st.sidebar.subheader("Uploaded Files")
    uploaded_files = list_uploaded_files()
    if uploaded_files:
        for collection in uploaded_files:
            st.sidebar.text(collection)
            for file in uploaded_files[collection]:
                file_path = os.path.join(f"data/{collection}/", file)
                file_path = file_path.replace(" ", "&nbsp;")
                st.sidebar.markdown(f"- [{file}]({file_path})", unsafe_allow_html=True)
    else:
        st.sidebar.write("No files uploaded yet.")

def list_uploaded_files():
    files = {}
    for collection in os.listdir("app/data"):
        files.update({collection: [f for f in os.listdir(f"app/data/{collection}") if f.endswith(".pdf")]})
    return files