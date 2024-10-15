import streamlit as st
from io import BytesIO
from helpers import add_file_to_global_storage, enhance_content, add_to_chat_knowledge_base, generate_file_with_gpt
from global_files import initialize_global_files, create_zip_of_global_files
import zipfile
import os
import base64
import requests

GLOBAL_FILES_DIR = "global_files"

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

def analyze_and_store_file(file_name, file_data):
    if file_name.lower().endswith('.txt'):
        content = file_data.decode('utf-8')
        analyzed_content = enhance_content(content, file_name)
        add_to_chat_knowledge_base(file_name, analyzed_content)
        st.success(f"Analyzed and stored {file_name} in knowledge base")
    elif file_name.lower().endswith('.zip'):
        with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename.lower().endswith('.txt'):
                    with zip_ref.open(zip_info.filename) as f:
                        content = f.read().decode('utf-8')
                        analyzed_content = enhance_content(content, zip_info.filename)
                        add_to_chat_knowledge_base(zip_info.filename, analyzed_content)
                        st.success(f"Analyzed and stored {zip_info.filename} from {file_name} in knowledge base")
    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        api_keys = st.session_state.get("api_keys", {})
        api_key = api_keys.get("openai")
        if api_key:
            analyze_and_store_image(api_key, file_name, file_data)

def delete_all_files():
    st.session_state["global_file_storage"] = {}
    st.success("All files have been deleted.")

def file_management_tab():
    st.title("File Management")

    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        file_data = uploaded_file.read()
        add_file_to_global_storage(uploaded_file.name, file_data)
        analyze_and_store_file(uploaded_file.name, file_data)
        st.success(f"Uploaded {uploaded_file.name}")

    # Add text field and button for generating files using GPT-4
    st.subheader("Generate File")
    generation_prompt = st.text_input("Enter prompt to generate file:")
    if st.button("Generate File"):
        if generation_prompt:
            with st.spinner("Generating file..."):
                file_name, file_data = generate_file_with_gpt(generation_prompt)
                if file_name and file_data:
                    st.session_state[file_name] = file_data
                    add_file_to_global_storage(file_name, file_data)
                    st.success(f"Generated file: {file_name}")
                    st.download_button(
                        label="Download Generated File",
                        data=file_data,
                        file_name=file_name,
                        mime="text/plain" if file_name.endswith(".txt") else None
                    )

    files = st.session_state.get("global_file_storage", {})
    if files:
        st.subheader("Uploaded Files")

        # Place the Download All as ZIP button above the files
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download All as ZIP"):
                with st.spinner("Creating ZIP file..."):
                    zip_data = create_zip_of_global_files()
                    st.download_button(
                        label="Download ZIP",
                        data=zip_data.getvalue(),
                        file_name="all_files.zip",
                        mime="application/zip"
                    )
        with col2:
            if st.button("Delete All Files"):
                delete_all_files()

        for file_name, file_data in files.items():
            st.write(f"{file_name}: {len(file_data)} bytes")

if __name__ == "__main__":
    initialize_global_files()
    file_management_tab()
