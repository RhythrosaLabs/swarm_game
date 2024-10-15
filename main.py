import streamlit as st
import requests
import json
import os
import zipfile
import time
from io import BytesIO
from PIL import Image, ImageOps
import replicate
import base64
import numpy as np
import pandas as pd
from fpdf import FPDF
from gtts import gTTS
import threading

# Set page configuration
st.set_page_config(
    page_title="B35 - Super-Powered Automation App",
    layout="wide",
    page_icon="🚀"
)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': '',
        'replicate': '',
        'stability': '',
        'luma': '',
        'runway': '',
        'clipdrop': ''
    }

if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

if 'generated_videos' not in st.session_state:
    st.session_state.generated_videos = []

if 'workflow_files' not in st.session_state:
    st.session_state.workflow_files = {}

if 'campaign_plan' not in st.session_state:
    st.session_state.campaign_plan = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'global_file_storage' not in st.session_state:
    st.session_state.global_file_storage = {}

if 'chat_knowledge_base' not in st.session_state:
    st.session_state.chat_knowledge_base = {}

# Constants
GLOBAL_FILES_DIR = "global_files"
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"

# Helper Functions
def load_api_keys():
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", 'r') as file:
            data = json.load(file)
            st.session_state.api_keys.update(data)

def save_api_keys():
    with open("api_keys.json", 'w') as file:
        json.dump(st.session_state.api_keys, file)

def get_headers(api_name):
    api_key = st.session_state.api_keys.get(api_name)
    if not api_key:
        return None
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def get_all_global_files():
    return st.session_state.get("global_file_storage", {})

def add_file_to_global_storage(file_name, file_data):
    st.session_state.global_file_storage[file_name] = file_data

def add_to_chat_knowledge_base(file_name, description):
    st.session_state.chat_knowledge_base[file_name] = description

def display_chat_history():
    st.sidebar.markdown("### Chat History")
    for entry in st.session_state.get("chat_history", []):
        if entry["role"] == "user":
            st.sidebar.markdown(f"**You:** {entry['content']}")
        else:
            st.sidebar.markdown(f"**Assistant:** {entry['content']}")

def generate_content(prompt, role):
    model = st.session_state.get('selected_chat_model', 'gpt-4')
    headers = get_headers('openai')
    if not headers:
        st.error("OpenAI API key is not set.")
        return None

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant specializing in {role}."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        content_text = response_data["choices"][0]["message"]["content"]
        return content_text
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

def generate_image(prompt, size="512x512"):
    model = st.session_state.get('selected_image_model', 'dall-e')
    if model == 'dall-e':
        headers = get_headers('openai')
        if not headers:
            st.error("OpenAI API key is not set.")
            return None
        data = {
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "url"
        }
        try:
            response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            image_url = response_data['data'][0]['url']
            return image_url
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None
    elif model == 'stable-diffusion':
        stability_api_key = st.session_state.api_keys.get('stability')
        if not stability_api_key:
            st.error("Stability AI API key is not set.")
            return None
        return generate_image_with_stability(prompt, size)
    else:
        st.error("Selected image model is not supported yet.")
        return None

def generate_image_with_stability(prompt, size):
    headers = {
        "Authorization": f"Bearer {st.session_state.api_keys['stability']}",
        "Content-Type": "application/json"
    }
    data = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "clip_guidance_preset": "FAST_BLUE",
        "height": int(size.split('x')[1]),
        "width": int(size.split('x')[0]),
        "samples": 1,
        "steps": 30,
    }
    try:
        response = requests.post("https://api.stability.ai/v1/generation/stable-diffusion-xl-beta-v2-2-2/text-to-image", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_base64 = response_data['artifacts'][0]['base64']
        image_data = base64.b64decode(image_base64)
        return image_data
    except Exception as e:
        st.error(f"Error generating image with Stability AI: {e}")
        return None

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

def display_image(image_data, caption):
    try:
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def generate_audio_logo(prompt):
    model = st.session_state.get('selected_music_model', 'meta-musicgen')
    if model == 'meta-musicgen':
        return generate_music_with_replicate(prompt)
    else:
        st.error("Selected music model is not supported yet.")
        return None, None

def generate_video_logo(prompt):
    model = st.session_state.get('selected_video_model', 'luma-ai')
    if model == 'luma-ai':
        luma_api_key = st.session_state.api_keys.get('luma')
        if not luma_api_key:
            st.error("Luma AI API key is not set.")
            return None, None
        # Placeholder for Luma AI video generation
        st.info("Luma AI video generation is not yet implemented.")
        return None, None
    else:
        st.error("Selected video model is not supported yet.")
        return None, None

def create_gif(images, filter_type=None):
    try:
        pil_images = [Image.open(BytesIO(img)) for img in images]
        if filter_type:
            pil_images = [apply_filter(img, filter_type) for img in pil_images]
        gif_buffer = BytesIO()
        pil_images[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=pil_images[1:],
            duration=1000,
            loop=0
        )
        gif_buffer.seek(0)
        return gif_buffer
    except Exception as e:
        st.error(f"Error creating GIF: {e}")
        return None

def apply_filter(image, filter_type):
    try:
        if filter_type == "sepia":
            sepia_image = ImageOps.colorize(ImageOps.grayscale(image), '#704214', '#C0A080')
            return sepia_image
        elif filter_type == "greyscale":
            return ImageOps.grayscale(image).convert("RGB")
        elif filter_type == "negative":
            return ImageOps.invert(image)
        elif filter_type == "solarize":
            return ImageOps.solarize(image, threshold=128)
        elif filter_type == "posterize":
            return ImageOps.posterize(image, bits=2)
        else:
            return image
    except Exception as e:
        st.error(f"Error applying filter: {e}")
        return image

def load_preset_bots():
    if os.path.exists('presetBots.json'):
        with open('presetBots.json') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                st.error("Error decoding presetBots.json.")
                return {}
    else:
        st.warning("presetBots.json not found.")
        return {}

def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

def describe_image(api_key, base64_image):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4-vision",
        "messages": [
            {"role": "user", "content": f"Describe the following image: data:image/jpeg;base64,{base64_image}"}
        ]
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
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

def generate_file_with_gpt(prompt):
    api_keys = st.session_state.api_keys
    openai_api_key = api_keys.get("openai")
    replicate_api_key = api_keys.get("replicate")

    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the sidebar.")
        return None, None

    if prompt.startswith("/music "):
        if not replicate_api_key:
            st.error("Replicate API key is not set. Please add it in the sidebar.")
            return None, None
        specific_prompt = prompt.replace("/music ", "").strip()
        return generate_music_with_replicate(specific_prompt)

    if prompt.startswith("/image "):
        specific_prompt = prompt.replace("/image ", "").strip()
        file_name = specific_prompt.replace(" ", "_") + ".png"
        image_url_or_data = generate_image(specific_prompt)
        if image_url_or_data:
            if isinstance(image_url_or_data, bytes):
                image_data = image_url_or_data
            else:
                image_data = download_image(image_url_or_data)
            if image_data:
                return file_name, image_data
            else:
                return None, None
        else:
            return None, None

    if prompt.startswith("/video "):
        specific_prompt = prompt.replace("/video ", "").strip()
        return generate_video_logo(specific_prompt)

    specific_prompt = f"Please generate the following file content without any explanations or additional text:\n{prompt}"

    model = st.session_state.get('selected_code_model', 'gpt-4')
    headers = get_headers('openai')
    if not headers:
        st.error("OpenAI API key is not set.")
        return None, None

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": specific_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data['choices'][0]['message']['content']

        generated_text = generated_text.strip()
        if prompt.startswith("/python "):
            start_index = generated_text.find("import")
            generated_text = generated_text[start_index:] if start_index != -1 else generated_text
        elif prompt.startswith("/html "):
            start_index = generated_text.find("<!DOCTYPE html>")
            generated_text = generated_text[start_index:] if start_index != -1 else generated_text
        elif prompt.startswith("/js "):
            generated_text = generated_text
        elif prompt.startswith("/md "):
            generated_text = generated_text
        elif prompt.startswith("/pdf "):
            generated_text = generated_text
        elif prompt.startswith("/doc ") or prompt.startswith("/txt "):
            start_index = generated_text.find("\n")
            generated_text = generated_text[start_index + 1:] if start_index != -1 else generated_text

        if generated_text.endswith("'''"):
            generated_text = generated_text[:-3].strip()
        elif generated_text.endswith("```"):
            generated_text = generated_text[:-3].strip()

    except requests.RequestException as e:
        st.error(f"Error generating file: {e}")
        return None, None

    if prompt.startswith("/python "):
        file_extension = ".py"
    elif prompt.startswith("/html "):
        file_extension = ".html"
    elif prompt.startswith("/js "):
        file_extension = ".js"
    elif prompt.startswith("/md "):
        file_extension = ".md"
    elif prompt.startswith("/pdf "):
        file_extension = ".pdf"
    elif prompt.startswith("/doc "):
        file_extension = ".doc"
    elif prompt.startswith("/txt "):
        file_extension = ".txt"
    else:
        file_extension = ".txt"

    # Ensure the file name is safe
    base_name = prompt.split(" ", 1)[1].strip().replace(" ", "_")
    file_name = f"{base_name}{file_extension}"
    file_data = generated_text.encode("utf-8")

    return file_name, file_data

def generate_music_with_replicate(prompt):
    replicate_api_key = st.session_state.api_keys.get("replicate")
    if not replicate_api_key:
        st.error("Replicate API key is not set.")
        return None, None
    model = st.session_state.get('selected_music_model', 'meta-musicgen')
    if model == 'meta-musicgen':
        try:
            client = replicate.Client(api_token=replicate_api_key)
            output_url = client.run(
                "facebook/MusicGen",
                input={"text": prompt}
            )
            music_data = requests.get(output_url).content
            file_name = prompt.replace(" ", "_") + ".mp3"
            return file_name, music_data
        except Exception as e:
            st.error(f"Error generating music: {e}")
            return None, None
    else:
        st.error("Selected music model is not supported yet.")
        return None, None

def chat_with_gpt(prompt, uploaded_files):
    model = st.session_state.get('selected_chat_model', 'gpt-4')
    headers = get_headers('openai')
    openai_api_key = st.session_state.api_keys.get("openai")

    if not openai_api_key:
        return "Error: OpenAI API key is not set."

    if not headers:
        return "Error: OpenAI API headers are not set."

    file_contents = []
    for file in uploaded_files:
        content = st.session_state.global_file_storage.get(file, None)
        if content:
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    content = "Binary file content not displayable."
            file_contents.append(f"File: {file}\nContent:\n{content}\n")
        else:
            file_contents.append(f"Content for {file} not found in global storage.\n")

    knowledge_base_contents = [
        f"File: {k}\nDescription:\n{v}\n" for k, v in st.session_state.get("chat_knowledge_base", {}).items()
    ]

    chat_history = st.session_state.get("chat_history", [])

    data = {
        "model": model,
        "messages": chat_history + [
            {"role": "user", "content": f"{prompt}\n\nFiles:\n{''.join(file_contents)}\n\nKnowledge Base:\n{''.join(knowledge_base_contents)}"}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]

        # Save chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state["chat_history"] = chat_history

        return assistant_reply
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

def enhance_content(content, filename):
    api_key = st.session_state.api_keys.get('openai')
    if not api_key:
        st.warning("OpenAI API Key is required for content enhancement.")
        return content

    model = st.session_state.get('selected_code_model', 'gpt-4')
    headers = get_headers('openai')
    if not headers:
        st.error("OpenAI API headers are not set.")
        return content

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Enhance the following content from {filename}."},
            {"role": "user", "content": content}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        enhanced_content = response_data['choices'][0]['message']['content']
        return enhanced_content
    except Exception as e:
        st.error(f"Error enhancing content: {e}")
        return content

def analyze_and_store_file(file_name, file_data):
    if file_name.lower().endswith('.txt'):
        try:
            content = file_data.decode('utf-8')
            analyzed_content = enhance_content(content, file_name)
            add_to_chat_knowledge_base(file_name, analyzed_content)
            st.success(f"Analyzed and stored {file_name} in knowledge base.")
        except UnicodeDecodeError:
            st.error(f"Failed to decode {file_name}.")
    elif file_name.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.lower().endswith('.txt'):
                        with zip_ref.open(zip_info.filename) as f:
                            content = f.read().decode('utf-8')
                            analyzed_content = enhance_content(content, zip_info.filename)
                            add_to_chat_knowledge_base(zip_info.filename, analyzed_content)
                            st.success(f"Analyzed and stored {zip_info.filename} from {file_name} in knowledge base.")
        except zipfile.BadZipFile:
            st.error(f"{file_name} is not a valid zip file.")
    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        api_key = st.session_state.api_keys.get("openai")
        if api_key:
            analyze_and_store_image(api_key, file_name, file_data)
        else:
            st.error("OpenAI API key is not set for image analysis.")

def delete_all_files():
    st.session_state["global_file_storage"] = {}
    st.session_state["chat_knowledge_base"] = {}
    st.success("All files and knowledge base entries have been deleted.")

def create_zip_of_global_files():
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name, file_data in st.session_state.get("global_file_storage", {}).items():
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return zip_buffer

def initialize_global_files():
    if not os.path.exists(GLOBAL_FILES_DIR):
        os.makedirs(GLOBAL_FILES_DIR)

def file_management_tab():
    st.title("📂 File Management")

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
                    mime_type = "text/plain"
                    if file_name.endswith(".py"):
                        mime_type = "text/x-python"
                    elif file_name.endswith(".html"):
                        mime_type = "text/html"
                    elif file_name.endswith(".js"):
                        mime_type = "application/javascript"
                    elif file_name.endswith(".md"):
                        mime_type = "text/markdown"
                    elif file_name.endswith(".pdf"):
                        mime_type = "application/pdf"
                    elif file_name.endswith(".doc"):
                        mime_type = "application/msword"
                    st.download_button(
                        label="Download Generated File",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type
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
            st.write(f"**{file_name}**: {len(file_data)} bytes")
            # Optionally, provide individual download buttons
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                st.image(file_data, caption=file_name, use_column_width=True)
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="image/png" if file_name.lower().endswith('.png') else "image/jpeg"
                )
            elif file_name.lower().endswith(('.mp3', '.wav')):
                st.audio(file_data, format="audio/mp3" if file_name.lower().endswith('.mp3') else "audio/wav")
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="audio/mp3" if file_name.lower().endswith('.mp3') else "audio/wav"
                )
            elif file_name.lower().endswith(('.pdf', '.doc', '.txt', '.py', '.html', '.js', '.md')):
                mime_type = "application/octet-stream"
                if file_name.endswith(".pdf"):
                    mime_type = "application/pdf"
                elif file_name.endswith(".doc"):
                    mime_type = "application/msword"
                elif file_name.endswith(".txt"):
                    mime_type = "text/plain"
                elif file_name.endswith(".py"):
                    mime_type = "text/x-python"
                elif file_name.endswith(".html"):
                    mime_type = "text/html"
                elif file_name.endswith(".js"):
                    mime_type = "application/javascript"
                elif file_name.endswith(".md"):
                    mime_type = "text/markdown"
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime=mime_type
                )
            else:
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_data,
                    file_name=file_name,
                    mime="application/octet-stream"
                )

# Sidebar with Tabs: API Keys and Chat
def sidebar():
    with st.sidebar:
        tab = st.radio("Sidebar", ["🔑 API Keys", "💬 Chat"], key="sidebar_tab")

        if tab == "🔑 API Keys":
            st.header("🔑 API Keys")
            st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_keys['openai'],
                type="password",
                key="openai_api_key"
            )
            st.text_input(
                "Replicate API Key",
                value=st.session_state.api_keys['replicate'],
                type="password",
                key="replicate_api_key"
            )
            st.text_input(
                "Stability AI API Key",
                value=st.session_state.api_keys['stability'],
                type="password",
                key="stability_api_key"
            )
            st.text_input(
                "Luma AI API Key",
                value=st.session_state.api_keys['luma'],
                type="password",
                key="luma_api_key"
            )
            st.text_input(
                "RunwayML API Key",
                value=st.session_state.api_keys['runway'],
                type="password",
                key="runway_api_key"
            )
            st.text_input(
                "Clipdrop API Key",
                value=st.session_state.api_keys['clipdrop'],
                type="password",
                key="clipdrop_api_key"
            )
            if st.button("💾 Save API Keys"):
                st.session_state.api_keys['openai'] = st.session_state.openai_api_key
                st.session_state.api_keys['replicate'] = st.session_state.replicate_api_key
                st.session_state.api_keys['stability'] = st.session_state.stability_api_key
                st.session_state.api_keys['luma'] = st.session_state.luma_api_key
                st.session_state.api_keys['runway'] = st.session_state.runway_api_key
                st.session_state.api_keys['clipdrop'] = st.session_state.clipdrop_api_key
                save_api_keys()
                st.success("API Keys saved successfully!")

        elif tab == "💬 Chat":
            st.header("💬 Chat Assistant")
            # Model selection
            st.subheader("Model Selection")
            st.session_state['selected_chat_model'] = st.selectbox(
                "Chat Model",
                ["gpt-4", "gpt-3.5-turbo", "gpt-4-32k"],
                key="selected_chat_model"
            )
            st.session_state['selected_image_model'] = st.selectbox(
                "Image Model",
                ["dall-e", "stable-diffusion"],
                key="selected_image_model"
            )
            st.session_state['selected_video_model'] = st.selectbox(
                "Video Model",
                ["luma-ai"],
                key="selected_video_model"
            )
            st.session_state['selected_music_model'] = st.selectbox(
                "Music Model",
                ["meta-musicgen"],
                key="selected_music_model"
            )
            st.session_state['selected_code_model'] = st.selectbox(
                "Code Model",
                ["gpt-4", "gpt-3.5-turbo"],
                key="selected_code_model"
            )

            # Chat functionality in sidebar
            use_personal_assistants = st.checkbox("Use Personal Assistants", key="use_personal_assistants")

            preset_bots = load_preset_bots() if use_personal_assistants else {}

            selected_bot = None
            if use_personal_assistants and preset_bots:
                categories = list(preset_bots.keys())
                if categories:
                    selected_category = st.selectbox("Choose a category:", categories, key="category_select")
                    bots = preset_bots[selected_category]
                    bot_names = [bot['name'] for bot in bots]
                    selected_bot_name = st.selectbox("Choose a bot:", bot_names, key="bot_select")

                    selected_bot = next((bot for bot in bots if bot['name'] == selected_bot_name), None)
                    if selected_bot:
                        bot_description = selected_bot.get('description', '')
                        bot_instructions = selected_bot.get('instructions', '')

                        st.write(f"**{selected_bot_name}**: {bot_description}")
                        st.write(f"*Instructions*: {bot_instructions}")

            prompt = st.text_area("Enter your prompt here...", key="chat_prompt")

            if st.button("Send", key="send_button"):
                if prompt.strip() == "":
                    st.warning("Please enter a prompt.")
                else:
                    with st.spinner("Fetching response..."):
                        all_files = get_all_global_files()

                        # Limit the number of files and their size
                        max_files = 5
                        max_file_size = 1024 * 1024  # 1 MB
                        relevant_files = {k: v for k, v in all_files.items() if len(v) <= max_file_size}
                        selected_files = list(relevant_files.keys())[:max_files]

                        # Ensure all files in selected_files exist in session state
                        for file in selected_files:
                            if file not in st.session_state:
                                st.session_state[file] = all_files[file]

                        # Include bot instructions in the prompt if a bot is selected
                        if selected_bot:
                            full_prompt = f"{selected_bot['instructions']}\n\n{prompt}"
                        else:
                            full_prompt = prompt

                        response = chat_with_gpt(full_prompt, selected_files)
                        if response:
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            display_chat_history()

            # Display chat history
            display_chat_history()

# Main Tabs
def main_tabs():
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 AI Content Generation",
        "🎬 Media Generation",
        "📂 Custom Workflows",
        "📁 File Management"
    ])

    # Tab 1: AI Content Generation
    with tab1:
        st.header("🧠 AI Content Generation")
        st.write("Generate marketing campaigns, game plans, comic books, and more using AI.")
        action = st.selectbox("Choose an action", ["Select an action", "Marketing Campaign", "Game Plan", "Comic Book"])
        prompt = st.text_area("Enter your topic/keywords:")
        if st.button("Generate", key="generate_content"):
            if action == "Select an action":
                st.warning("Please select an action.")
            elif not prompt.strip():
                st.warning("Please enter a topic or keywords.")
            else:
                if action == "Marketing Campaign":
                    generate_marketing_campaign(prompt)
                elif action == "Game Plan":
                    st.write("Game Plan generation coming soon!")
                elif action == "Comic Book":
                    st.write("Comic Book generation coming soon!")

    # Tab 2: Media Generation
    with tab2:
        st.header("🎬 Media Generation")
        st.write("Generate images and videos using AI models.")
        media_type = st.selectbox("Select Media Type", ["Select", "Image Generation", "Video Generation", "Music Generation"])
        if media_type == "Image Generation":
            image_prompt = st.text_area("Enter an image prompt:")
            if st.button("Generate Image"):
                if image_prompt.strip() == "":
                    st.warning("Please enter an image prompt.")
                else:
                    with st.spinner("Generating image..."):
                        file_name = image_prompt.replace(" ", "_") + ".png"
                        image_url_or_data = generate_image(image_prompt)
                        if image_url_or_data:
                            if isinstance(image_url_or_data, bytes):
                                image_data = image_url_or_data
                            else:
                                image_data = download_image(image_url_or_data)
                            if image_data:
                                st.session_state.generated_images.append(image_data)
                                display_image(image_data, "Generated Image")
                                add_file_to_global_storage(file_name, image_data)
                                api_key = st.session_state.api_keys.get("openai")
                                if api_key:
                                    analyze_and_store_image(api_key, file_name, image_data)
        elif media_type == "Video Generation":
            video_prompt = st.text_area("Enter a video prompt:")
            if st.button("Generate Video"):
                if video_prompt.strip() == "":
                    st.warning("Please enter a video prompt.")
                else:
                    with st.spinner("Generating video..."):
                        file_name, video_data = generate_video_logo(video_prompt)
                        if video_data:
                            st.session_state.generated_videos.append(video_data)
                            st.video(video_data)
                            add_file_to_global_storage(file_name, video_data)
        elif media_type == "Music Generation":
            music_prompt = st.text_area("Enter a music prompt:")
            if st.button("Generate Music"):
                if music_prompt.strip() == "":
                    st.warning("Please enter a music prompt.")
                else:
                    with st.spinner("Generating music..."):
                        file_name, music_data = generate_music_with_replicate(music_prompt)
                        if music_data:
                            st.audio(music_data, format="audio/mp3")
                            add_file_to_global_storage(file_name, music_data)

    # Tab 3: Custom Workflows
    with tab3:
        st.header("📂 Custom Workflows")
        st.write("Create custom automated workflows.")
        if "workflow_steps" not in st.session_state:
            st.session_state["workflow_steps"] = []

        def add_step():
            st.session_state["workflow_steps"].append({"prompt": "", "file_name": "", "file_data": None})

        if st.button("Add Step"):
            add_step()

        for i, step in enumerate(st.session_state["workflow_steps"]):
            st.write(f"### Step {i + 1}")
            step["prompt"] = st.text_input(f"Prompt for step {i + 1}", value=step["prompt"], key=f"prompt_{i}")
            if st.button("Remove Step", key=f"remove_step_{i}"):
                st.session_state["workflow_steps"].pop(i)
                st.experimental_rerun()

        if st.button("Generate All Files"):
            for i, step in enumerate(st.session_state["workflow_steps"]):
                if step["prompt"].strip():
                    with st.spinner(f"Generating file for step {i + 1}..."):
                        file_name, file_data = generate_file_with_gpt(step["prompt"])
                        if file_name and file_data:
                            step["file_name"] = file_name
                            step["file_data"] = file_data
                            st.success(f"File for step {i + 1} generated: {file_name}")
                            add_file_to_global_storage(file_name, file_data)
                else:
                    st.warning(f"Prompt for step {i + 1} is empty.")

        if st.button("Download Workflow Files as ZIP"):
            with st.spinner("Creating ZIP file..."):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for step in st.session_state["workflow_steps"]:
                        if step["file_data"]:
                            zipf.writestr(step["file_name"], step["file_data"])
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="workflow_files.zip",
                    mime="application/zip"
                )

    # Tab 4: File Management
    with tab4:
        file_management_tab()

# Generate Marketing Campaign Function
def generate_marketing_campaign(prompt):
    st.info("Generating campaign concept...")
    campaign_concept = generate_content(
        f"Create a detailed marketing campaign concept based on the following prompt: {prompt}.",
        "marketing"
    )
    if campaign_concept:
        st.session_state.campaign_plan['campaign_concept'] = campaign_concept
        add_file_to_global_storage("campaign_concept.txt", campaign_concept.encode("utf-8"))

    st.info("Generating marketing plan...")
    marketing_plan = generate_content(
        f"Create a detailed marketing plan for the campaign: {campaign_concept}",
        "marketing"
    )
    if marketing_plan:
        st.session_state.campaign_plan['marketing_plan'] = marketing_plan
        add_file_to_global_storage("marketing_plan.txt", marketing_plan.encode("utf-8"))

    st.info("Generating images...")
    images = {}
    descriptions = {
        "banner": "Wide banner image in a modern and appealing style, with absolutely no text, matching the theme of: " + campaign_concept,
        "instagram_background": "Tall background image suitable for Instagram video, with absolutely no text, matching the theme of: " + campaign_concept,
        "square_post": "Square background image for social media post, with absolutely no text, matching the theme of: " + campaign_concept,
    }
    sizes = {
        "banner": "1792x1024",
        "instagram_background": "1024x1792",
        "square_post": "1024x1024",
    }
    for key, desc in descriptions.items():
        image_data = generate_image(desc, sizes[key])
        if image_data:
            if isinstance(image_data, bytes):
                images[f"{key}.png"] = image_data
            else:
                image_content = download_image(image_data)
                if image_content:
                    images[f"{key}.png"] = image_content
            add_file_to_global_storage(f"{key}.png", images[f"{key}.png"])
    st.session_state.campaign_plan['images'] = images

    st.info("Generating resources and tips...")
    resources_tips = generate_content(
        f"List resources and tips for executing the marketing campaign: {campaign_concept}",
        "marketing"
    )
    if resources_tips:
        st.session_state.campaign_plan['resources_tips'] = resources_tips
        add_file_to_global_storage("resources_tips.txt", resources_tips.encode("utf-8"))

    st.info("Generating recap...")
    recap = generate_content(
        f"Recap the marketing campaign: {campaign_concept}",
        "marketing"
    )
    if recap:
        st.session_state.campaign_plan['recap'] = recap
        add_file_to_global_storage("recap.txt", recap.encode("utf-8"))

    st.info("Generating master document...")
    master_doc = create_master_document(st.session_state.campaign_plan)
    st.session_state.campaign_plan['master_document'] = master_doc
    add_file_to_global_storage("master_document.txt", master_doc.encode("utf-8"))

    st.success("Marketing Campaign Generated!")
    with st.spinner("Creating ZIP file..."):
        zip_buffer = create_zip(st.session_state.campaign_plan)
        st.download_button(
            label="Download Campaign ZIP",
            data=zip_buffer.getvalue(),
            file_name="marketing_campaign.zip",
            mime="application/zip"
        )

def create_master_document(content_dict):
    master_doc = ""
    for key, value in content_dict.items():
        if key == "images":
            continue
        if isinstance(value, dict):
            continue  # Skip dictionaries like 'images'
        master_doc += f"{key.replace('_', ' ').capitalize()}:\n{value}\n\n"
    return master_doc

def create_zip(content_dict):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, value in content_dict.items():
            if isinstance(value, str):
                zip_file.writestr(f"{key}.txt", value)
            elif isinstance(value, bytes):
                zip_file.writestr(key, value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        zip_file.writestr(f"{key}/{sub_key}.txt", sub_value)
                    elif isinstance(sub_value, bytes):
                        zip_file.writestr(f"{key}/{sub_key}", sub_value)
    zip_buffer.seek(0)
    return zip_buffer

# Main function
def main():
    load_api_keys()
    initialize_global_files()
    sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
