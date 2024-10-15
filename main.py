import streamlit as st
import requests
import zipfile
import os
import json
import time
from io import BytesIO
from PIL import Image, ImageOps
from fpdf import FPDF
from gtts import gTTS
import replicate
import pandas as pd
import base64
from streamlit_option_menu import option_menu

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="B35 - Super-Powered Automation App",
    layout="wide",
    page_icon="🚀"
)

# --------------------------
# Initialize Session State
# --------------------------
def initialize_session_state():
    """Initialize all necessary session state keys."""
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

    # Initialize selected models if not present
    if 'selected_code_model' not in st.session_state:
        st.session_state.selected_code_model = 'gpt-4o'

    if 'selected_image_model' not in st.session_state:
        st.session_state.selected_image_model = 'dalle3'

    if 'selected_video_model' not in st.session_state:
        st.session_state.selected_video_model = 'stable diffusion'

    if 'selected_audio_model' not in st.session_state:
        st.session_state.selected_audio_model = 'music gen'

initialize_session_state()

# --------------------------
# Constants
# --------------------------
GLOBAL_FILES_DIR = "global_files"
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
STABILITY_API_URL = "https://api.stability.ai/v2beta/image-to-video"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

# --------------------------
# Helper Functions
# --------------------------

def load_api_keys():
    """Load API keys from a JSON file if it exists."""
    if os.path.exists("api_keys.json"):
        with open("api_keys.json", 'r') as file:
            data = json.load(file)
            st.session_state.api_keys.update(data)

def save_api_keys():
    """Save API keys to a JSON file."""
    with open("api_keys.json", 'w') as file:
        json.dump(st.session_state.api_keys, file)

def get_headers(api_name):
    """Generate headers for API requests based on the API name."""
    api_key = st.session_state.api_keys.get(api_name)
    if not api_key:
        return None
    if api_name == 'replicate':
        return {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def add_file_to_global_storage(file_name, file_data):
    """Add a file to the global file storage."""
    st.session_state.global_file_storage[file_name] = file_data

def add_to_chat_knowledge_base(file_name, description):
    """Add analysis of a file to the chat knowledge base."""
    st.session_state.chat_knowledge_base[file_name] = description

def display_chat_history():
    """Display the chat history."""
    st.sidebar.markdown("### Chat History")
    for entry in reversed(st.session_state.get("chat_history", [])):
        if entry["role"] == "user":
            st.sidebar.markdown(f"**You:** {entry['content']}")
        else:
            st.sidebar.markdown(f"**Assistant:** {entry['content']}")

def generate_content(prompt, role):
    """Generate content using the selected chat model."""
    model = 'gpt-4o'  # Fixed model for content generation
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
    """Generate an image using the selected image model."""
    model = st.session_state.get('selected_image_model', 'dalle3')
    if model == 'dalle3':
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
            response = requests.post(DALLE_API_URL, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            image_url = response_data['data'][0]['url']
            return image_url
        except Exception as e:
            st.error(f"Error generating image: {e}")
            return None
    elif model == 'stable diffusion':
        stability_api_key = st.session_state.api_keys.get('stability')
        if not stability_api_key:
            st.error("Stability AI API key is not set.")
            return None
        return generate_image_with_stability(prompt, size)
    elif model == 'flux':
        # Placeholder for Flux model integration
        st.error("Flux model integration is not implemented yet.")
        return None
    else:
        st.error("Selected image model is not supported yet.")
        return None

def generate_image_with_stability(prompt, size):
    """Generate image using Stability AI's API."""
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
    """Download image data from a URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

def display_image(image_data, caption):
    """Display an image in the app."""
    try:
        image = Image.open(BytesIO(image_data))
        st.image(image, caption=caption, use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def generate_audio_logo(prompt, api_key):
    """Generate audio logo using Replicate API."""
    input_data = {
        "prompt": prompt,
        "model_version": "stereo-large",
        "output_format": "mp3",
        "normalization_strategy": "peak"
    }

    try:
        replicate_client = replicate.Client(api_token=api_key)
        output_url = replicate_client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input=input_data
        )
        audio_data = requests.get(output_url).content
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error generating audio: {str(e)}")
        return None, None
    except requests.RequestException as e:
        st.error(f"Error downloading audio: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".mp3"
    file_data = audio_data

    return file_name, file_data

def generate_video_logo(prompt, api_key):
    """Generate video logo using OpenAI's image generation and Stability AI's video API."""
    # Generate image first
    file_name, image_data = generate_image_with_dalle(prompt, api_key)
    if image_data:
        # Animate image to video using Stability AI
        generation_id = animate_image_to_video(image_data, prompt)
        if generation_id:
            video_data = fetch_generated_video(generation_id)
            if video_data:
                file_name = prompt.replace(" ", "_") + ".mp4"
                return file_name, video_data
    return None, None

def animate_image_to_video(image_data, prompt):
    """Animate image to video using Stability AI's API."""
    stability_api_key = st.session_state.api_keys.get("stability")
    url = "https://api.stability.ai/v2beta/image-to-video"

    # Open and resize the image
    image = Image.open(BytesIO(image_data))
    image = image.resize((768, 768))

    # Save the resized image to a buffer
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    # POST request to start video generation
    response = requests.post(
        url,
        headers={
            "authorization": f"Bearer {stability_api_key}"
        },
        files={
            "image": buffered
        },
        data={
            "seed": 0,
            "cfg_scale": 1.8,
            "motion_bucket_id": 127
        }
    )

    if response.status_code == 200:
        generation_id = response.json().get('id')
        return generation_id
    else:
        st.error(f"Error initiating video generation: {response.text}")
        return None

def fetch_generated_video(generation_id):
    """Fetch the generated video from Stability AI's API."""
    stability_api_key = st.session_state.api_keys.get("stability")
    url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"

    while True:
        response = requests.get(
            url,
            headers={
                'accept': "video/*",  # Use 'application/json' to receive base64 encoded JSON
                'authorization': f"Bearer {stability_api_key}"
            }
        )

        if response.status_code == 202:
            st.info("Video generation in progress, please wait...")
            time.sleep(10)
        elif response.status_code == 200:
            st.success("Video generation complete.")
            return response.content
        else:
            st.error(f"Error fetching video: {response.text}")
            return None

def generate_music_with_replicate(prompt, api_key):
    """Generate music using Replicate API."""
    input_data = {
        "prompt": prompt,
        "model_version": "stereo-large",
        "output_format": "mp3",
        "normalization_strategy": "peak"
    }

    try:
        replicate_client = replicate.Client(api_token=api_key)
        output_url = replicate_client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input=input_data
        )
        music_data = requests.get(output_url).content
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error generating music: {str(e)}")
        return None, None
    except requests.RequestException as e:
        st.error(f"Error downloading music: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".mp3"
    file_data = music_data

    return file_name, file_data

def generate_tts(text, lang="en"):
    """Generate text-to-speech audio using gTTS."""
    try:
        tts = gTTS(text=text, lang=lang)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def generate_file_with_gpt(prompt):
    """Generate a file based on user prompt using GPT-4o."""
    api_keys = st.session_state.api_keys
    openai_api_key = api_keys.get("openai")
    replicate_api_key = api_keys.get("replicate")

    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the sidebar.")
        return None, None

    # Handle specific commands
    if prompt.startswith("/music "):
        if not replicate_api_key:
            st.error("Replicate API key is not set. Please add it in the sidebar.")
            return None, None
        specific_prompt = prompt.replace("/music ", "").strip()
        return generate_music_with_replicate(specific_prompt, replicate_api_key)

    if prompt.startswith("/image "):
        specific_prompt = prompt.replace("/image ", "").strip()
        return generate_image_with_dalle(specific_prompt, openai_api_key)

    if prompt.startswith("/video "):
        specific_prompt = prompt.replace("/video ", "").strip()
        return generate_video_logo(specific_prompt, openai_api_key)

    if prompt.startswith("/speak "):
        specific_prompt = prompt.replace("/speak ", "").strip()
        audio_data = generate_tts(specific_prompt)
        if audio_data:
            file_name = specific_prompt.replace(" ", "_") + ".mp3"
            return file_name, audio_data
        else:
            return None, None

    # Default: Generate text-based file
    specific_prompt = f"Please generate the following file content without any explanations or additional text:\n{prompt}"

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
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

        # Clean up the generated text based on prompt
        generated_text = generated_text.strip()
        if any(prompt.startswith(prefix) for prefix in ["/python ", "/html ", "/js ", "/md ", "/doc ", "/txt ", "/docx ", "/rtf ", "/csv ", "/json ", "/xml ", "/yaml ", "/ini ", "/log ", "/c ", "/cpp ", "/java ", "/xls ", "/xlsx ", "/ppt ", "/pptx ", "/bat ", "/sh ", "/ps1 "]):
            if prompt.startswith("/python "):
                start_index = generated_text.find("import")
                generated_text = generated_text[start_index:] if start_index != -1 else generated_text
            elif prompt.startswith("/html "):
                start_index = generated_text.find("<!DOCTYPE html>")
                generated_text = generated_text[start_index:] if start_index != -1 else generated_text
            elif prompt.startswith("/js "):
                start_index = 0
                generated_text = generated_text[start_index:]
            elif prompt.startswith("/md "):
                start_index = 0
                generated_text = generated_text[start_index:]
            elif prompt.startswith("/doc ") or prompt.startswith("/txt "):
                start_index = generated_text.find("\n") + 1
                generated_text = generated_text[start_index:] if start_index != -1 else generated_text

        if generated_text.endswith("'''"):
            generated_text = generated_text[:-3].strip()
        elif generated_text.endswith("```"):
            generated_text = generated_text[:-3].strip()

    except requests.RequestException as e:
        st.error(f"Error generating file: {e}")
        return None, None

    # Determine file extension
    if prompt.startswith("/python "):
        file_extension = ".py"
    elif prompt.startswith("/html "):
        file_extension = ".html"
    elif prompt.startswith("/js "):
        file_extension = ".js"
    elif prompt.startswith("/md "):
        file_extension = ".md"
    elif prompt.startswith("/doc "):
        file_extension = ".doc"
    elif prompt.startswith("/docx "):
        file_extension = ".docx"
    elif prompt.startswith("/txt "):
        file_extension = ".txt"
    elif prompt.startswith("/rtf "):
        file_extension = ".rtf"
    elif prompt.startswith("/csv "):
        file_extension = ".csv"
    elif prompt.startswith("/json "):
        file_extension = ".json"
    elif prompt.startswith("/xml "):
        file_extension = ".xml"
    elif prompt.startswith("/yaml "):
        file_extension = ".yaml"
    elif prompt.startswith("/ini "):
        file_extension = ".ini"
    elif prompt.startswith("/log "):
        file_extension = ".log"
    elif prompt.startswith("/c "):
        file_extension = ".c"
    elif prompt.startswith("/cpp "):
        file_extension = ".cpp"
    elif prompt.startswith("/java "):
        file_extension = ".java"
    elif prompt.startswith("/xls "):
        file_extension = ".xls"
    elif prompt.startswith("/xlsx "):
        file_extension = ".xlsx"
    elif prompt.startswith("/ppt "):
        file_extension = ".ppt"
    elif prompt.startswith("/pptx "):
        file_extension = ".pptx"
    elif prompt.startswith("/bat "):
        file_extension = ".bat"
    elif prompt.startswith("/sh "):
        file_extension = ".sh"
    elif prompt.startswith("/ps1 "):
        file_extension = ".ps1"
    else:
        file_extension = ".txt"

    # Create file name and data
    base_name = prompt.split(" ", 1)[1].strip().replace(" ", "_")
    file_name = f"{base_name}{file_extension}"
    file_data = generated_text.encode("utf-8")

    return file_name, file_data

def generate_image_with_dalle(prompt, api_key):
    """Generate image using DALL·E 3."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    try:
        response = requests.post(DALLE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        image_data = requests.get(image_url).content
    except requests.RequestException as e:
        st.error(f"Error generating image: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".png"
    file_data = image_data

    return file_name, file_data

def create_master_document(content_dict):
    """Create a master document summarizing the campaign plan."""
    master_doc = "Marketing Campaign Master Document\n\n"
    for key, value in content_dict.items():
        if key == "images":
            master_doc += f"{key.capitalize()}:\n"
            for img_key in value:
                master_doc += f" - {img_key}: See attached image.\n"
        else:
            master_doc += f"{key.replace('_', ' ').capitalize()}:\n{value}\n\n"
    return master_doc

def create_zip(content_dict):
    """Create a ZIP file containing all campaign files."""
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

def enhance_content(content, filename):
    """Enhance content using GPT-4o."""
    api_key = st.session_state.api_keys.get('openai')
    if not api_key:
        st.warning("OpenAI API Key is required for content enhancement.")
        return content

    headers = get_headers('openai')
    if not headers:
        st.error("OpenAI API headers are not set.")
        return content

    if isinstance(content, bytes):
        try:
            content = content.decode('utf-8')
        except UnicodeDecodeError:
            st.warning(f"Content of {filename} is not text. Skipping enhancement.")
            return content

    data = {
        "model": "gpt-4o",
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
    """Analyze and store a file's content into the knowledge base."""
    if file_name.lower().endswith('.txt'):
        content = file_data.decode('utf-8')
        analyzed_content = enhance_content(content, file_name)
        add_to_chat_knowledge_base(file_name, analyzed_content)
        add_file_to_global_storage("analyzed_" + file_name, analyzed_content.encode('utf-8'))
    elif file_name.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename.lower().endswith('.txt'):
                        with zip_ref.open(zip_info.filename) as f:
                            content = f.read().decode('utf-8')
                            analyzed_content = enhance_content(content, zip_info.filename)
                            add_to_chat_knowledge_base(zip_info.filename, analyzed_content)
                            add_file_to_global_storage("analyzed_" + zip_info.filename, analyzed_content.encode('utf-8'))
        except zipfile.BadZipFile:
            st.error(f"{file_name} is not a valid zip file.")
    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        api_key = st.session_state.api_keys.get("openai")
        if api_key:
            analyze_and_store_image(api_key, file_name, file_data)
        else:
            st.error("OpenAI API key is not set for image analysis.")

def analyze_and_store_image(api_key, file_name, file_data):
    """Analyze and store image description in knowledge base."""
    base64_image = encode_image(file_data)
    description = describe_image(api_key, base64_image)
    if description:
        add_to_chat_knowledge_base(file_name, description)
        st.success(f"Image {file_name} analyzed and stored in knowledge base.")
    else:
        st.error(f"Failed to analyze and store image {file_name}.")

def encode_image(image_data):
    """Encode image data to base64."""
    return base64.b64encode(image_data).decode('utf-8')

def describe_image(api_key, base64_image):
    """Describe an image using GPT-4o."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": f"What's in this image?\n![image](data:image/jpeg;base64,{base64_image})"
            }
        ],
        "max_tokens": 1000
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=payload)
        if response.status_code == 200 and 'choices' in response.json():
            description = response.json()['choices'][0]['message']['content']
            return description
        else:
            return "Failed to analyze the image."
    except Exception as e:
        return f"An error occurred: {e}"

def delete_all_files():
    """Delete all files from the global file storage and knowledge base."""
    st.session_state["global_file_storage"] = {}
    st.session_state["chat_knowledge_base"] = {}
    st.session_state["chat_history"] = []
    st.success("All files and knowledge base entries have been deleted.")

# --------------------------
# Sidebar with Four Tabs
# --------------------------
def sidebar_menu():
    """Configure the sidebar with four tabs: Keys, Models, About, Chat."""
    
    # Ensure session state keys are initialized before using them
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'openai': '',
            'replicate': '',
            'stability': '',
            'luma': '',
            'runway': '',
            'clipdrop': ''
        }
    
    if 'selected_code_model' not in st.session_state:
        st.session_state['selected_code_model'] = 'gpt-4o'
    
    if 'selected_image_model' not in st.session_state:
        st.session_state['selected_image_model'] = 'dalle3'
    
    if 'selected_video_model' not in st.session_state:
        st.session_state['selected_video_model'] = 'stable diffusion'
    
    if 'selected_audio_model' not in st.session_state:
        st.session_state['selected_audio_model'] = 'music gen'
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Sidebar layout
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["🔑 Keys", "🛠️ Models", "ℹ️ About", "💬 Chat"],
            icons=["key", "tools", "info-circle", "chat-dots"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical"
        )

        if selected == "🔑 Keys":
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

        elif selected == "🛠️ Models":
            st.header("🛠️ Models Selection")
            st.subheader("Code Models")
            st.session_state['selected_code_model'] = st.selectbox(
                "Select Code Model",
                ["gpt-4o", "gpt-4", "llama"],
                index=["gpt-4o", "gpt-4", "llama"].index(st.session_state['selected_code_model']),
                key="selected_code_model"
            )

            st.subheader("Image Models")
            st.session_state['selected_image_model'] = st.selectbox(
                "Select Image Model",
                ["dalle3", "stable diffusion", "flux"],
                index=["dalle3", "stable diffusion", "flux"].index(st.session_state['selected_image_model']),
                key="selected_image_model"
            )

            st.subheader("Video Models")
            st.session_state['selected_video_model'] = st.selectbox(
                "Select Video Model",
                ["stable diffusion", "luma"],
                index=["stable diffusion", "luma"].index(st.session_state['selected_video_model']),
                key="selected_video_model"
            )

            st.subheader("Audio Models")
            st.session_state['selected_audio_model'] = st.selectbox(
                "Select Audio Model",
                ["music gen"],
                index=["music gen"].index(st.session_state['selected_audio_model']),
                key="selected_audio_model"
            )

            st.success("Model selections updated.")

        elif selected == "ℹ️ About":
            st.header("ℹ️ About This App")
            st.write("""
                **B35 - Super-Powered Automation App** is designed to streamline your content generation, media creation, and workflow automation using cutting-edge AI models.
                
                **Features:**
                - **AI Content Generation**: Create marketing campaigns, game plans, and more.
                - **Media Generation**: Generate images, videos, and audio content.
                - **Custom Workflows**: Automate complex tasks with customizable workflows.
                - **File Management**: Upload, generate, and manage your files seamlessly.
                - **Chat Assistant**: Interact with GPT-4o for live knowledge and assistance.
                
                **Supported Models:**
                - **Code**: GPT-4o, GPT-4, Llama
                - **Image**: DALL·E 3, Stable Diffusion, Flux
                - **Video**: Stable Diffusion, Luma
                - **Audio**: Music Gen
            """)

        elif selected == "💬 Chat":
            st.header("💬 Chat Assistant")
            st.subheader("GPT-4o Chat")
            prompt = st.text_area("Enter your prompt here...", key="chat_prompt_sidebar")
            if st.button("Send", key="send_button_sidebar"):
                if prompt.strip() == "":
                    st.warning("Please enter a prompt.")
                else:
                    with st.spinner("Fetching response..."):
                        response = chat_with_gpt(prompt)
                        if response:
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            display_chat_history()

            st.markdown("### Chat History")
            display_chat_history()


def chat_with_gpt(prompt):
    """Handle chat interactions with GPT-4o."""
    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        return "Error: OpenAI API key is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare messages including knowledge base and global files
    file_contents = []
    for file in list(st.session_state.global_file_storage.keys())[:5]:
        content = st.session_state.global_file_storage.get(file, "")
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = "Binary file content not displayable."
        file_contents.append(f"File: {file}\nContent:\n{content}\n")

    knowledge_base_contents = [
        f"File: {k}\nDescription:\n{v}\n" for k, v in st.session_state.chat_knowledge_base.items()
    ]

    # Log the action taken
    action_log = f"Action Taken: User initiated a chat with prompt: '{prompt}'"

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant integrated with the application's knowledge base and file management system."},
            {"role": "system", "content": action_log},
            {"role": "user", "content": f"{prompt}\n\nFiles:\n{''.join(file_contents)}\n\nKnowledge Base:\n{''.join(knowledge_base_contents)}"}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]
        return assistant_reply
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

# --------------------------
# Main Tabs
# --------------------------
def main_tabs():
    """Configure the main tabs: AI Content Generation, Media Generation, Custom Workflows, File Management."""
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
        action = st.selectbox("Choose an action", ["Select an action", "Marketing Campaign", "Game Plan", "Comic Book"], key="content_generation_action")
        prompt = st.text_area("Enter your topic/keywords:", key="content_generation_prompt")
        if st.button("Generate", key="generate_content_button"):
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
        st.write("Generate images, videos, and audio using AI models.")
        media_type = st.selectbox("Select Media Type", ["Select", "Image Generation", "Video Generation", "Audio Generation"], key="media_generation_type")
        if media_type == "Image Generation":
            image_prompt = st.text_area("Enter an image prompt:", key="image_generation_prompt")
            size = st.selectbox("Select Image Size", ["512x512", "1024x1024", "1792x1024", "1024x1792"], key="image_generation_size")
            if st.button("Generate Image", key="generate_image_button"):
                if image_prompt.strip() == "":
                    st.warning("Please enter an image prompt.")
                else:
                    with st.spinner("Generating image..."):
                        image_url = generate_image(image_prompt, size)
                        if image_url:
                            image_data = download_image(image_url)
                            if image_data:
                                add_file_to_global_storage(f"generated_image_{len(st.session_state.generated_images)+1}.png", image_data)
                                st.session_state.generated_images.append(image_data)
                                display_image(image_data, "Generated Image")
                                analyze_and_store_file(f"generated_image_{len(st.session_state.generated_images)}.png", image_data)
        elif media_type == "Video Generation":
            video_prompt = st.text_area("Enter a video prompt:", key="video_generation_prompt")
            if st.button("Generate Video", key="generate_video_button"):
                if video_prompt.strip() == "":
                    st.warning("Please enter a video prompt.")
                else:
                    with st.spinner("Generating video..."):
                        file_name, video_data = generate_video_logo(video_prompt, st.session_state.api_keys.get("openai"))
                        if video_data:
                            add_file_to_global_storage(file_name, video_data)
                            st.session_state.generated_videos.append(video_data)
                            st.video(video_data)
        elif media_type == "Audio Generation":
            audio_prompt = st.text_area("Enter an audio prompt:", key="audio_generation_prompt")
            if st.button("Generate Audio", key="generate_audio_button"):
                if audio_prompt.strip() == "":
                    st.warning("Please enter an audio prompt.")
                else:
                    with st.spinner("Generating audio..."):
                        file_name, audio_data = generate_audio_logo(audio_prompt, st.session_state.api_keys.get("replicate"))
                        if audio_data:
                            add_file_to_global_storage(file_name, audio_data)
                            st.audio(audio_data, format="audio/mp3")
                            st.success(f"Generated audio: {file_name}")

    # Tab 3: Custom Workflows
    with tab3:
        st.header("📂 Custom Workflows")
        st.write("Create custom automated workflows.")
        if "workflow_steps" not in st.session_state:
            st.session_state["workflow_steps"] = []

        def add_step():
            st.session_state["workflow_steps"].append({"prompt": "", "file_name": "", "file_data": None})

        if st.button("➕ Add Step", key="add_workflow_step_button"):
            add_step()

        for i, step in enumerate(st.session_state["workflow_steps"]):
            st.write(f"### Step {i + 1}")
            step["prompt"] = st.text_input(f"Prompt for step {i + 1}", value=step["prompt"], key=f"workflow_prompt_{i}")
            if st.button("➖ Remove Step", key=f"remove_workflow_step_{i}"):
                st.session_state["workflow_steps"].pop(i)
                st.experimental_rerun()

        if st.button("Generate All Files", key="generate_all_workflow_files_button"):
            for i, step in enumerate(st.session_state["workflow_steps"]):
                if step["prompt"].strip():
                    with st.spinner(f"Generating file for step {i + 1}..."):
                        file_name, file_data = generate_file_with_gpt(step["prompt"])
                        if file_name and file_data:
                            step["file_name"] = file_name
                            step["file_data"] = file_data
                            add_file_to_global_storage(file_name, file_data)
                            st.success(f"File for step {i + 1} generated: {file_name}")
                else:
                    st.warning(f"Prompt for step {i + 1} is empty.")

        if st.button("Download Workflow Files as ZIP", key="download_workflow_zip_button"):
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
        st.header("📁 File Management")

        uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "gif", "mp3", "mp4", "txt", "zip"])
        if uploaded_file is not None:
            file_data = uploaded_file.read()
            add_file_to_global_storage(uploaded_file.name, file_data)
            analyze_and_store_file(uploaded_file.name, file_data)
            st.success(f"Uploaded {uploaded_file.name}")

        # Generate File using GPT-4o
        st.subheader("Generate File with GPT-4o")
        generation_prompt = st.text_input("Enter prompt to generate file:", key="generation_prompt_main")
        if st.button("Generate File", key="generate_file_main_button"):
            if generation_prompt.strip():
                with st.spinner("Generating file..."):
                    file_name, file_data = generate_file_with_gpt(generation_prompt)
                    if file_name and file_data:
                        add_file_to_global_storage(file_name, file_data)
                        st.success(f"Generated file: {file_name}")
                        st.download_button(
                            label="Download Generated File",
                            data=file_data,
                            file_name=file_name,
                            mime="application/octet-stream"
                        )
            else:
                st.warning("Please enter a prompt to generate a file.")

        # Display Uploaded Files
        files = st.session_state.get("global_file_storage", {})
        if files:
            st.subheader("Uploaded Files")

            # Download All as ZIP and Delete All Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download All as ZIP", key="download_all_zip_main_button"):
                    with st.spinner("Creating ZIP file..."):
                        zip_data = create_zip(st.session_state.global_file_storage)
                        st.download_button(
                            label="Download ZIP",
                            data=zip_data.getvalue(),
                            file_name="all_files.zip",
                            mime="application/zip"
                        )
            with col2:
                if st.button("🗑️ Delete All Files", key="delete_all_files_main_button"):
                    delete_all_files()

            # List Files with Download Buttons
            for file_name, file_data in files.items():
                st.write(f"**{file_name}**: {len(file_data)} bytes")
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    st.image(file_data, caption=file_name, use_column_width=True)
                    mime_type = "image/png" if file_name.lower().endswith('.png') else "image/jpeg"
                    st.download_button(
                        label=f"📥 Download {file_name}",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type
                    )
                elif file_name.lower().endswith(('.mp3', '.wav')):
                    st.audio(file_data, format="audio/mp3" if file_name.lower().endswith('.mp3') else "audio/wav")
                    mime_type = "audio/mp3" if file_name.lower().endswith('.mp3') else "audio/wav"
                    st.download_button(
                        label=f"📥 Download {file_name}",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type
                    )
                elif file_name.lower().endswith(('.doc', '.docx', '.txt', '.py', '.html', '.js', '.md')):
                    mime_type = "application/octet-stream"
                    if file_name.endswith(".doc"):
                        mime_type = "application/msword"
                    elif file_name.endswith(".docx"):
                        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
                        label=f"📥 Download {file_name}",
                        data=file_data,
                        file_name=file_name,
                        mime=mime_type
                    )
                else:
                    st.download_button(
                        label=f"📥 Download {file_name}",
                        data=file_data,
                        file_name=file_name,
                        mime="application/octet-stream"
                    )

# --------------------------
# Chat Functionality
# --------------------------

def chat_with_gpt(prompt):
    """Handle chat interactions with GPT-4o."""
    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        return "Error: OpenAI API key is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Prepare messages including knowledge base and global files
    file_contents = []
    for file in list(st.session_state.global_file_storage.keys())[:5]:
        content = st.session_state.global_file_storage.get(file, "")
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = "Binary file content not displayable."
        file_contents.append(f"File: {file}\nContent:\n{content}\n")

    knowledge_base_contents = [
        f"File: {k}\nDescription:\n{v}\n" for k, v in st.session_state.chat_knowledge_base.items()
    ]

    # Log the action taken
    action_log = f"Action Taken: User initiated a chat with prompt: '{prompt}'"

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant integrated with the application's knowledge base and file management system."},
            {"role": "system", "content": action_log},
            {"role": "user", "content": f"{prompt}\n\nFiles:\n{''.join(file_contents)}\n\nKnowledge Base:\n{''.join(knowledge_base_contents)}"}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]
        return assistant_reply
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

# --------------------------
# Generate Marketing Campaign
# --------------------------

def generate_marketing_campaign(prompt):
    """Generate a comprehensive marketing campaign based on the prompt."""
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
        image_url = generate_image(desc, sizes[key])
        if image_url:
            if isinstance(image_url, str):
                image_data = download_image(image_url)
                if image_data:
                    images[f"{key}.png"] = image_data
                    add_file_to_global_storage(f"{key}.png", image_data)
            else:
                images[f"{key}.png"] = image_url
                add_file_to_global_storage(f"{key}.png", image_url)
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

# --------------------------
# Initialize Global Files Directory
# --------------------------
def initialize_global_files():
    """Initialize the global files directory."""
    if not os.path.exists(GLOBAL_FILES_DIR):
        os.makedirs(GLOBAL_FILES_DIR)

# --------------------------
# Main Function
# --------------------------
def main():
    """Main function to run the Streamlit app."""
    load_api_keys()
    initialize_global_files()
    sidebar_menu()
    main_tabs()

if __name__ == "__main__":
    main()
