import streamlit as st
import requests
import json
import os
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from gtts import gTTS
import replicate
import time
from fpdf import FPDF
import threading
import base64
from PyPDF2 import PdfReader
import cv2
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.playback import play
import ffmpeg
import plotly.express as px
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import pytesseract
import pyttsx3
import opencv_python_headless
from pedalboard import Pedalboard, Reverb, Compressor
import docx
from reportlab.pdfgen import canvas
import xlsxwriter
from tqdm import tqdm
from midiutil import MIDIFile
import pygame
import streamlit_quill
import streamlit_editorjs
import streamlit_aggrid
import altair as alt

# Set page configuration
st.set_page_config(page_title="B35 - Super-Powered Automation App", layout="wide", page_icon="🚀")

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

if 'model_selections' not in st.session_state:
    st.session_state.model_selections = {
        'image_generation': 'DALL·E',
        'video_generation': 'RunwayML',
        'music_generation': 'MusicGen'
    }

if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []

if 'generated_videos' not in st.session_state:
    st.session_state.generated_videos = []

if 'workflow_files' not in st.session_state:
    st.session_state.workflow_files = {}

if 'campaign_plan' not in st.session_state:
    st.session_state.campaign_plan = {}

if 'game_plan' not in st.session_state:
    st.session_state.game_plan = {}

if 'business_plan' not in st.session_state:
    st.session_state.business_plan = {}

if 'comic_book' not in st.session_state:
    st.session_state.comic_book = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'global_file_storage' not in st.session_state:
    st.session_state.global_file_storage = {}

if 'chat_knowledge_base' not in st.session_state:
    st.session_state.chat_knowledge_base = {}

if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Constants
GLOBAL_FILES_DIR = "global_files"
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"

# Helper Functions

def load_api_keys():
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", 'r') as file:
            data = json.load(file)
            st.session_state.api_keys.update(data)

def save_api_keys():
    with open("api_keys.json", 'w') as file:
        json.dump(st.session_state.api_keys, file)

def get_headers():
    return {
        "Authorization": f"Bearer {st.session_state.api_keys['openai']}",
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

def delete_all_files():
    st.session_state["global_file_storage"] = {}
    st.session_state["chat_knowledge_base"] = {}
    st.success("All files and knowledge base entries have been deleted.")

def initialize_global_files():
    if not os.path.exists(GLOBAL_FILES_DIR):
        os.makedirs(GLOBAL_FILES_DIR)

def create_zip_of_global_files():
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name, file_data in st.session_state["global_file_storage"].items():
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return zip_buffer

def get_model_selections():
    return st.session_state.model_selections

def save_model_selections():
    st.session_state.model_selections['image_generation'] = st.session_state.image_generation_model
    st.session_state.model_selections['video_generation'] = st.session_state.video_generation_model
    st.session_state.model_selections['music_generation'] = st.session_state.music_generation_model

def generate_image_with_dalle(prompt, size="1024x1024"):
    api_key = st.session_state.api_keys['openai']
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "n": 1,
        "size": size
    }
    try:
        response = requests.post(DALLE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        image_data = requests.get(image_url).content
        return image_data
    except Exception as e:
        st.error(f"Error generating image with DALL·E: {e}")
        return None

def generate_image_with_stable_diffusion(prompt, size="512x512"):
    replicate_api_key = st.session_state.api_keys.get("replicate")
    if not replicate_api_key:
        st.error("Replicate API Key is required for Stable Diffusion.")
        return None
    model = replicate.models.get("stability-ai/stable-diffusion")
    try:
        image_url = model.predict(prompt=prompt)[0]
        image_data = requests.get(image_url).content
        return image_data
    except Exception as e:
        st.error(f"Error generating image with Stable Diffusion: {e}")
        return None

def generate_image(prompt, size="1024x1024"):
    model_selections = get_model_selections()
    if model_selections['image_generation'] == 'DALL·E':
        return generate_image_with_dalle(prompt, size)
    elif model_selections['image_generation'] == 'Stable Diffusion':
        return generate_image_with_stable_diffusion(prompt, size)
    else:
        st.error("Invalid image generation model selected.")
        return None

def generate_music(prompt):
    replicate_api_key = st.session_state.api_keys.get("replicate")
    if not replicate_api_key:
        st.error("Replicate API Key is required for music generation.")
        return None
    model = replicate.models.get("facebook/musicgen")
    try:
        music_url = model.predict(text=prompt)[0]
        music_data = requests.get(music_url).content
        return music_data
    except Exception as e:
        st.error(f"Error generating music: {e}")
        return None

def generate_video(prompt):
    runway_api_key = st.session_state.api_keys.get("runway")
    if not runway_api_key:
        st.error("RunwayML API Key is required for video generation.")
        return None
    # Implement RunwayML API calls here
    st.error("RunwayML integration is not implemented in this example.")
    return None

def generate_business_plan(prompt):
    api_key = st.session_state.api_keys['openai']
    headers = get_headers()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": f"Generate a detailed business plan for the following idea:\n\n{prompt}"}
        ],
        "max_tokens": 3000
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        plan_text = response.json()['choices'][0]['message']['content']
        st.session_state.business_plan['business_plan.txt'] = plan_text
        add_file_to_global_storage('business_plan.txt', plan_text)
        return plan_text
    except Exception as e:
        st.error(f"Error generating business plan: {e}")
        return None

def generate_game_plan(prompt):
    api_key = st.session_state.api_keys['openai']
    headers = get_headers()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": f"Generate a detailed game development plan for the following idea:\n\n{prompt}"}
        ],
        "max_tokens": 3000
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        game_plan_text = response.json()['choices'][0]['message']['content']
        st.session_state.game_plan['game_plan.txt'] = game_plan_text
        add_file_to_global_storage('game_plan.txt', game_plan_text)
        return game_plan_text
    except Exception as e:
        st.error(f"Error generating game plan: {e}")
        return None

def generate_comic_book(prompt):
    api_key = st.session_state.api_keys['openai']
    headers = get_headers()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": f"Generate a script for a comic book based on the following idea:\n\n{prompt}"}
        ],
        "max_tokens": 3000
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        script_text = response.json()['choices'][0]['message']['content']
        st.session_state.comic_book['comic_script.txt'] = script_text
        add_file_to_global_storage('comic_script.txt', script_text)
        return script_text
    except Exception as e:
        st.error(f"Error generating comic book script: {e}")
        return None

# Sidebar Functions

def sidebar():
    with st.sidebar:
        tab = st.radio("Sidebar", ["🔑 API Keys", "⚙️ Model Selections", "💬 Chat"], key="sidebar_tab")

        if tab == "🔑 API Keys":
            st.header("🔑 API Keys")
            st.text_input("OpenAI API Key", value=st.session_state.api_keys['openai'], type="password", key="openai_api_key")
            st.text_input("Replicate API Key", value=st.session_state.api_keys['replicate'], type="password", key="replicate_api_key")
            st.text_input("Stability AI API Key", value=st.session_state.api_keys['stability'], type="password", key="stability_api_key")
            st.text_input("Luma AI API Key", value=st.session_state.api_keys['luma'], type="password", key="luma_api_key")
            st.text_input("RunwayML API Key", value=st.session_state.api_keys['runway'], type="password", key="runway_api_key")
            st.text_input("Clipdrop API Key", value=st.session_state.api_keys['clipdrop'], type="password", key="clipdrop_api_key")
            if st.button("💾 Save API Keys"):
                st.session_state.api_keys['openai'] = st.session_state.openai_api_key
                st.session_state.api_keys['replicate'] = st.session_state.replicate_api_key
                st.session_state.api_keys['stability'] = st.session_state.stability_api_key
                st.session_state.api_keys['luma'] = st.session_state.luma_api_key
                st.session_state.api_keys['runway'] = st.session_state.runway_api_key
                st.session_state.api_keys['clipdrop'] = st.session_state.clipdrop_api_key
                save_api_keys()
                st.success("API Keys saved successfully!")

        elif tab == "⚙️ Model Selections":
            st.header("⚙️ Model Selections")
            st.selectbox("Image Generation Model", ["DALL·E", "Stable Diffusion"], key="image_generation_model")
            st.selectbox("Video Generation Model", ["RunwayML", "Luma AI"], key="video_generation_model")
            st.selectbox("Music Generation Model", ["MusicGen", "Jukebox"], key="music_generation_model")
            if st.button("💾 Save Model Selections"):
                save_model_selections()
                st.success("Model selections saved successfully!")

        elif tab == "💬 Chat":
            st.header("💬 Chat Assistant")
            use_personal_assistants = st.checkbox("Use Personal Assistants", key="use_personal_assistants")
            preset_bots = load_preset_bots() if use_personal_assistants else None
            selected_bot = None
            if use_personal_assistants and preset_bots:
                categories = list(preset_bots.keys())
                selected_category = st.selectbox("Choose a category:", categories, key="category_select")
                bots = preset_bots[selected_category]
                bot_names = [bot['name'] for bot in bots]
                selected_bot_name = st.selectbox("Choose a bot:", bot_names, key="bot_select")
                selected_bot = next(bot for bot in bots if bot['name'] == selected_bot_name)
                bot_description = selected_bot.get('description', '')
                bot_instructions = selected_bot.get('instructions', '')
                st.write(f"**{selected_bot_name}**: {bot_description}")
                st.write(f"*Instructions*: {bot_instructions}")

            prompt = st.text_area("Enter your prompt here...", key="chat_prompt")
            if st.button("Send", key="send_button"):
                with st.spinner("Fetching response..."):
                    all_files = get_all_global_files()
                    max_files = 5
                    max_file_size = 1024 * 1024
                    relevant_files = {k: v for k, v in all_files.items() if len(v) <= max_file_size}
                    selected_files = list(relevant_files.keys())[:max_files]
                    for file in selected_files:
                        if file not in st.session_state:
                            st.session_state[file] = all_files[file]
                    if selected_bot:
                        full_prompt = f"{selected_bot['instructions']}\n\n{prompt}"
                    else:
                        full_prompt = prompt
                    response = chat_with_gpt(full_prompt, selected_files)
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    display_chat_history()
            display_chat_history()

def chat_with_gpt(prompt, uploaded_files):
    model = 'gpt-4'
    headers = get_headers()
    openai_api_key = st.session_state.api_keys.get('openai')

    if not openai_api_key:
        return "Error: OpenAI API key is not set."

    file_contents = []
    for file in uploaded_files:
        if file in st.session_state:
            content = st.session_state[file]
            if isinstance(content, bytes):
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    content = "Binary file content not displayable."
            file_contents.append(f"File: {file}\nContent:\n{content}\n")
        else:
            file_contents.append(f"Content for {file} not found in session state.")

    knowledge_base_contents = [f"File: {k}\nDescription:\n{v}\n" for k, v in st.session_state.get("chat_knowledge_base", {}).items()]

    chat_history = st.session_state.get("chat_history", [])

    data = {
        "model": model,
        "messages": chat_history + [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]

        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state["chat_history"] = chat_history

        return assistant_reply.strip()
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

def load_preset_bots():
    if os.path.exists('presetBots.json'):
        with open('presetBots.json') as f:
            return json.load(f)
    else:
        return {}

# Main Tabs

def main_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🧠 AI Content Generation",
        "🎬 Media Generation",
        "📂 Custom Workflows",
        "📁 File Management",
        "🎞️ Edit Videos",
        "🖼️ Multimedia Analysis",
        "📖 Comic Book Generation",
        "🎮 Game Development"
    ])

    with tab1:
        generate_content_tab()

    with tab2:
        media_generation_tab()

    with tab3:
        custom_workflows_tab()

    with tab4:
        file_management_tab()

    with tab5:
        edit_videos_tab()

    with tab6:
        multimedia_analysis_tab()

    with tab7:
        comic_book_generation_tab()

    with tab8:
        game_development_tab()

# Implement each tab function accordingly, similar to previous examples.

def generate_content_tab():
    st.title("🧠 AI Content Generation")
    st.header("Business Plan Automation")
    prompt = st.text_area("Enter your business idea:")
    if st.button("Generate Business Plan"):
        with st.spinner("Generating business plan..."):
            plan_text = generate_business_plan(prompt)
            if plan_text:
                st.success("Business plan generated!")
                st.download_button("Download Business Plan", plan_text, file_name="business_plan.txt")

def media_generation_tab():
    st.title("🎬 Media Generation")
    media_type = st.selectbox("Select Media Type", ["Image", "Video", "Music"])
    if media_type == "Image":
        st.header("Image Generation")
        prompt = st.text_area("Enter an image prompt:")
        size = st.selectbox("Select Image Size", ["256x256", "512x512", "1024x1024"])
        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                image_data = generate_image(prompt, size)
                if image_data:
                    st.image(image_data, caption="Generated Image")
                    add_file_to_global_storage(f"{prompt.replace(' ', '_')}.png", image_data)
    elif media_type == "Music":
        st.header("Music Generation")
        prompt = st.text_area("Enter a music prompt:")
        if st.button("Generate Music"):
            with st.spinner("Generating music..."):
                music_data = generate_music(prompt)
                if music_data:
                    st.audio(music_data)
                    add_file_to_global_storage(f"{prompt.replace(' ', '_')}.mp3", music_data)
    elif media_type == "Video":
        st.header("Video Generation")
        prompt = st.text_area("Enter a video prompt:")
        if st.button("Generate Video"):
            with st.spinner("Generating video..."):
                video_data = generate_video(prompt)
                if video_data:
                    st.video(video_data)
                    add_file_to_global_storage(f"{prompt.replace(' ', '_')}.mp4", video_data)

def custom_workflows_tab():
    st.title("📂 Custom Workflows")
    st.write("Create custom automated workflows.")
    # Implement custom workflows functionality here

def file_management_tab():
    st.title("📁 File Management")
    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file:
        file_data = uploaded_file.read()
        add_file_to_global_storage(uploaded_file.name, file_data)
        st.success(f"Uploaded {uploaded_file.name}")
    files = st.session_state.get("global_file_storage", {})
    if files:
        st.subheader("Uploaded Files")
        for file_name, file_data in files.items():
            st.write(f"{file_name}: {len(file_data)} bytes")
        if st.button("Download All as ZIP"):
            with st.spinner("Creating ZIP file..."):
                zip_data = create_zip_of_global_files()
                st.download_button(
                    label="Download ZIP",
                    data=zip_data.getvalue(),
                    file_name="all_files.zip",
                    mime="application/zip"
                )
        if st.button("Delete All Files"):
            delete_all_files()

def edit_videos_tab():
    st.title("🎞️ Edit Videos")
    # Implement video editing functionality here

def multimedia_analysis_tab():
    st.title("🖼️ Multimedia Analysis")
    # Implement multimedia analysis functionality here

def comic_book_generation_tab():
    st.title("📖 Comic Book Generation")
    prompt = st.text_area("Enter your comic book idea:")
    if st.button("Generate Comic Book Script"):
        with st.spinner("Generating comic book script..."):
            script_text = generate_comic_book(prompt)
            if script_text:
                st.success("Comic book script generated!")
                st.download_button("Download Comic Book Script", script_text, file_name="comic_script.txt")

def game_development_tab():
    st.title("🎮 Game Development")
    prompt = st.text_area("Enter your game idea:")
    if st.button("Generate Game Plan"):
        with st.spinner("Generating game development plan..."):
            game_plan_text = generate_game_plan(prompt)
            if game_plan_text:
                st.success("Game development plan generated!")
                st.download_button("Download Game Plan", game_plan_text, file_name="game_plan.txt")

def main():
    load_api_keys()
    initialize_global_files()
    sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
