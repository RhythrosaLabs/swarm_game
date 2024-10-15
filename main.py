import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip, CompositeVideoClip, vfx
import replicate
import time
import base64
import numpy as np
import pandas as pd
import traceback
from fpdf import FPDF
from gtts import gTTS

# Set page configuration
st.set_page_config(page_title="Super-Powered Automation App", layout="wide", page_icon="🚀")

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'openai': '', 'replicate': '', 'stability': '', 'luma': '', 'runway': '', 'clipdrop': ''}

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
    return {
        "Authorization": f"Bearer {st.session_state.api_keys[api_name]}",
        "Content-Type": "application/json"
    }

def generate_content(prompt, role):
    headers = get_headers('openai')
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant specializing in {role}."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        content_text = response_data["choices"][0]["message"]["content"]
        return content_text
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

def generate_image(prompt, size="1024x1024"):
    headers = get_headers('openai')
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

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

def add_file_to_global_storage(file_name, file_data):
    st.session_state.global_file_storage[file_name] = file_data

def get_all_global_files():
    return st.session_state.global_file_storage

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

def display_image(image_data, caption):
    image = Image.open(BytesIO(image_data))
    st.image(image, caption=caption, use_column_width=True)

def generate_audio_logo(prompt):
    replicate_api_key = st.session_state.api_keys['replicate']
    if not replicate_api_key:
        st.warning("Replicate API Key is required for audio generation.")
        return None, None
    try:
        client = replicate.Client(api_token=replicate_api_key)
        output_url = client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={"prompt": prompt}
        )
        audio_data = requests.get(output_url).content
        file_name = f"{prompt.replace(' ', '_')}.mp3"
        return file_name, audio_data
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None, None

def generate_video_logo(prompt):
    image_url = generate_image(prompt)
    if image_url:
        image_data = download_image(image_url)
        if image_data:
            # Here you can integrate video generation using the image
            return "video_logo.mp4", image_data  # Placeholder
    return None, None

def create_gif(images, filter_type=None):
    try:
        pil_images = [Image.open(BytesIO(img)) for img in images]
        if filter_type:
            pil_images = [apply_filter(img, filter_type) for img in pil_images]
        gif_buffer = BytesIO()
        pil_images[0].save(gif_buffer, format='GIF', save_all=True, append_images=pil_images[1:], duration=1000, loop=0)
        gif_buffer.seek(0)
        return gif_buffer
    except Exception as e:
        st.error(f"Error creating GIF: {e}")
        return None

def apply_filter(image, filter_type):
    if filter_type == "sepia":
        return image.convert("L").convert("RGB")
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

# Sidebar for API Keys
def api_keys_sidebar():
    st.sidebar.header("🔑 API Keys")
    st.sidebar.text_input("OpenAI API Key", value=st.session_state.api_keys['openai'], type="password", key="openai_api_key")
    st.sidebar.text_input("Replicate API Key", value=st.session_state.api_keys['replicate'], type="password", key="replicate_api_key")
    st.sidebar.text_input("Stability AI API Key", value=st.session_state.api_keys['stability'], type="password", key="stability_api_key")
    st.sidebar.text_input("Luma AI API Key", value=st.session_state.api_keys['luma'], type="password", key="luma_api_key")
    st.sidebar.text_input("RunwayML API Key", value=st.session_state.api_keys['runway'], type="password", key="runway_api_key")
    st.sidebar.text_input("Clipdrop API Key", value=st.session_state.api_keys['clipdrop'], type="password", key="clipdrop_api_key")
    if st.sidebar.button("💾 Save API Keys"):
        st.session_state.api_keys['openai'] = st.session_state.openai_api_key
        st.session_state.api_keys['replicate'] = st.session_state.replicate_api_key
        st.session_state.api_keys['stability'] = st.session_state.stability_api_key
        st.session_state.api_keys['luma'] = st.session_state.luma_api_key
        st.session_state.api_keys['runway'] = st.session_state.runway_api_key
        st.session_state.api_keys['clipdrop'] = st.session_state.clipdrop_api_key
        save_api_keys()
        st.sidebar.success("API Keys saved successfully!")

# Main Tabs
def main_tabs():
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 AI Content Generation", "🎬 Media Generation", "📂 Custom Workflows", "💬 Chat Assistant"])

    # Tab 1: AI Content Generation
    with tab1:
        st.header("🧠 AI Content Generation")
        st.write("Generate marketing campaigns, game plans, comic books, and more using AI.")
        action = st.selectbox("Choose an action", ["Select an action", "Marketing Campaign", "Game Plan", "Comic Book"])
        prompt = st.text_area("Enter your topic/keywords:")
        if st.button("Generate"):
            if action == "Select an action":
                st.warning("Please select an action.")
            elif not prompt:
                st.warning("Please enter a topic or keywords.")
            else:
                if action == "Marketing Campaign":
                    generate_marketing_campaign(prompt)
                elif action == "Game Plan":
                    generate_game_plan(prompt)
                elif action == "Comic Book":
                    generate_comic_book(prompt)

    # Tab 2: Media Generation
    with tab2:
        st.header("🎬 Media Generation")
        st.write("Generate images and videos using AI models.")
        media_type = st.selectbox("Select Media Type", ["Select", "Image Generation", "Video Generation"])
        if media_type == "Image Generation":
            image_prompt = st.text_area("Enter an image prompt:")
            if st.button("Generate Image"):
                image_url = generate_image(image_prompt)
                if image_url:
                    image_data = download_image(image_url)
                    if image_data:
                        st.session_state.generated_images.append(image_data)
                        display_image(image_data, "Generated Image")
        elif media_type == "Video Generation":
            video_prompt = st.text_area("Enter a video prompt:")
            if st.button("Generate Video"):
                # Placeholder for video generation
                st.write("Video generation feature coming soon!")

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
            st.write(f"Step {i + 1}")
            step["prompt"] = st.text_input(f"Prompt for step {i + 1}", key=f"prompt_{i}")
            if st.button("Remove Step", key=f"remove_step_{i}"):
                st.session_state["workflow_steps"].pop(i)
                st.experimental_rerun()

        if st.button("Generate All Files"):
            for i, step in enumerate(st.session_state["workflow_steps"]):
                if step["prompt"]:
                    file_name, file_data = generate_file_with_gpt(step["prompt"])
                    if file_name and file_data:
                        step["file_name"] = file_name
                        step["file_data"] = file_data
                        st.success(f"File for step {i + 1} generated: {file_name}")
                    else:
                        st.error(f"Failed to generate file for step {i + 1}")

        if st.button("Download Workflow Files as ZIP"):
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
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

    # Tab 4: Chat Assistant
    with tab4:
        st.header("💬 Chat Assistant")
        st.write("Interact with the assistant for personalized help.")
        user_input = st.text_input("You:", key="chat_input")
        if st.button("Send", key="chat_send"):
            if user_input:
                response = chat_with_gpt(user_input, [])
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                display_chat_history()

# Generate Marketing Campaign Function
def generate_marketing_campaign(prompt):
    st.info("Generating campaign concept...")
    campaign_concept = generate_content(f"Create a detailed marketing campaign concept based on the following prompt: {prompt}.", "marketing")
    st.session_state.campaign_plan['campaign_concept'] = campaign_concept

    st.info("Generating marketing plan...")
    marketing_plan = generate_content(f"Create a detailed marketing plan for the campaign: {campaign_concept}", "marketing")
    st.session_state.campaign_plan['marketing_plan'] = marketing_plan

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
        image_url = generate_image(desc, sizes[key])
        if image_url:
            image_data = download_image(image_url)
            if image_data:
                images[f"{key}.png"] = image_data
    st.session_state.campaign_plan['images'] = images

    st.info("Generating resources and tips...")
    resources_tips = generate_content(f"List resources and tips for executing the marketing campaign: {campaign_concept}", "marketing")
    st.session_state.campaign_plan['resources_tips'] = resources_tips

    st.info("Generating recap...")
    recap = generate_content(f"Recap the marketing campaign: {campaign_concept}", "marketing")
    st.session_state.campaign_plan['recap'] = recap

    st.info("Generating master document...")
    master_doc = create_master_document(st.session_state.campaign_plan)
    st.session_state.campaign_plan['master_document'] = master_doc

    st.success("Marketing Campaign Generated!")
    st.download_button(
        label="Download Campaign ZIP",
        data=create_zip(st.session_state.campaign_plan).getvalue(),
        file_name="marketing_campaign.zip",
        mime="application/zip"
    )

def create_master_document(content_dict):
    master_doc = ""
    for key in content_dict.keys():
        if key == "images":
            continue
        master_doc += f"{key.replace('_', ' ').capitalize()}:\n{content_dict[key]}\n\n"
    return master_doc

def generate_file_with_gpt(prompt):
    headers = get_headers('openai')
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        file_name = "generated_file.txt"
        return file_name, content.encode('utf-8')
    except Exception as e:
        st.error(f"Error generating file: {e}")
        return None, None

def chat_with_gpt(prompt, uploaded_files):
    headers = get_headers('openai')
    data = {
        "model": "gpt-4",
        "messages": st.session_state.chat_history + [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]
        return assistant_reply
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

def display_chat_history():
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**You:** {entry['content']}")
        else:
            st.markdown(f"**Assistant:** {entry['content']}")

# Main function
def main():
    load_api_keys()
    api_keys_sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
