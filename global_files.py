import os
import zipfile
from io import BytesIO
import base64
import requests
import streamlit as st
from PIL import Image
from helpers import add_file_to_global_storage, add_to_chat_knowledge_base

GLOBAL_FILES_DIR = "global_files"

def initialize_global_files():
    if not os.path.exists(GLOBAL_FILES_DIR):
        os.makedirs(GLOBAL_FILES_DIR)

def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

def describe_image(api_key, base64_image):
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 1000
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200 and 'choices' in response.json():
            description = response.json()['choices'][0]['message']['content']
            return description
        else:
            return "Failed to analyze the image."
    except Exception as e:
        return f"An error occurred: {e}"

def analyze_and_store_image(api_key, file_name, file_data):
    base64_image = encode_image(file_data)
    description = describe_image(api_key, base64_image)
    if description:
        add_to_chat_knowledge_base(file_name, description)
        st.success(f"Image {file_name} analyzed and stored in knowledge base.")
    else:
        st.error(f"Failed to analyze and store image {file_name}.")

def add_file_to_global(file_data, filename):
    """
    Adds a file to the global file storage.

    Parameters:
    file_data (BytesIO or str): The file data to be written to the global storage. It can be a BytesIO object or a file path.
    filename (str): The name of the file to be saved.

    Returns:
    str: The path to the saved file.
    """
    destination = os.path.join(GLOBAL_FILES_DIR, filename)
    
    # Check if file_data is a BytesIO object or a file path
    if isinstance(file_data, BytesIO):
        file_data.seek(0)
        file_content = file_data.read()
        with open(destination, 'wb') as dst_file:
            dst_file.write(file_content)
        add_file_to_global_storage(filename, file_content)
        # Analyze and store the image if it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            api_key = st.session_state["api_keys"]["openai"]
            analyze_and_store_image(api_key, filename, file_content)
    else:
        with open(file_data, 'rb') as src_file:
            file_content = src_file.read()
            with open(destination, 'wb') as dst_file:
                dst_file.write(file_content)
        add_file_to_global_storage(filename, file_content)
        # Analyze and store the image if it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            api_key = st.session_state["api_keys"]["openai"]
            analyze_and_store_image(api_key, filename, file_content)
    
    return destination

def list_global_files():
    return os.listdir(GLOBAL_FILES_DIR)

def create_zip_of_global_files():
    """
    Creates a zip file of all files in the global storage.

    Returns:
    BytesIO: A BytesIO object containing the zip file.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name, file_data in st.session_state.get("global_file_storage", {}).items():
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return zip_buffer

def add_to_chat_knowledge_base(file_name, description):
    if "chat_knowledge_base" not in st.session_state:
        st.session_state["chat_knowledge_base"] = {}
    st.session_state["chat_knowledge_base"][file_name] = description

def add_file_to_global_storage(file_name, file_data):
    if "global_file_storage" not in st.session_state:
        st.session_state["global_file_storage"] = {}
    st.session_state["global_file_storage"][file_name] = file_data

