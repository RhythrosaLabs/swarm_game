import streamlit as st
import requests
import json
import os
import zipfile
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
from gtts import gTTS
import replicate
import time
from fpdf import FPDF
import threading
import base64
import cv2
from stability_sdk import client
from stability_sdk.animation import AnimationArgs, Animator
from stability_sdk.utils import create_video_from_frames
from tqdm import tqdm

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
        'video_generation': 'Stability AI',
        'music_generation': 'Replicate'
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
STABILITY_API_URL = "https://api.stability.ai/v2beta/image-to-video"

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

def generate_content(action, prompt, budget, platforms, api_key):
    headers = get_headers()
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": f"You are a creative assistant specializing in {action}."},
            {"role": "user", "content": f"{prompt}"},
            {"role": "user", "content": f"Budget: {budget}"},
            {"role": "user", "content": f"Platforms: {', '.join([k for k, v in platforms.items() if v])}"}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        content_text = response_data["choices"][0]["message"]["content"]
        return content_text.strip()
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

def generate_budget_spreadsheet(budget):
    try:
        budget_value = float(budget)
    except ValueError:
        budget_value = 1000

    budget_allocation = {
        "Advertising": 0.5,
        "Content Creation": 0.2,
        "Social Media": 0.2,
        "Miscellaneous": 0.1
    }

    budget_data = [
        {"Category": category, "Amount": amount * budget_value}
        for category, amount in budget_allocation.items()
    ]

    budget_data.append({"Category": "Total", "Amount": budget_value})

    df = pd.DataFrame(budget_data)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Budget')

    return excel_buffer.getvalue()

def truncate_post_content(post_content):
    max_length = 100
    if len(post_content) > max_length:
        return post_content[:max_length] + "..."
    return post_content

def extract_hashtags(post_content):
    return " ".join([word for word in post_content.split() if word.startswith("#")])

def generate_social_media_schedule(campaign_concept, platforms):
    optimal_times = {
        "facebook": "12:00 PM",
        "twitter": "10:00 AM",
        "instagram": "3:00 PM",
        "linkedin": "11:00 AM"
    }

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    schedule_data = []

    for day in days_of_week:
        for platform, selected in platforms.items():
            if selected:
                post_content = f"Post about {campaign_concept} on {platform.capitalize()}"
                truncated_content = truncate_post_content(post_content)
                hashtags = extract_hashtags(post_content)
                schedule_data.append({
                    "Day": day,
                    "Platform": platform.capitalize(),
                    "Time": optimal_times.get(platform, "12:00 PM"),
                    "Post": truncated_content,
                    "Hashtags": hashtags
                })

    df = pd.DataFrame(schedule_data)
    pivot_table = df.pivot(index="Platform", columns="Day", values=["Time", "Post", "Hashtags"]).swaplevel(axis=1).sort_index(axis=1, level=0)

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        pivot_table.to_excel(writer, sheet_name='Social Media Schedule')

    return excel_buffer.getvalue()

def generate_images(api_key, image_prompts, sizes, model_name, hd=False):
    images = {}

    for i, (desc, size) in enumerate(zip(image_prompts, sizes)):
        st.info(f"Generating image {i+1} with {model_name}...")
        image_data = None
        if model_name == 'DALL·E':
            image_url = generate_image_dalle(api_key, desc, size, hd)
            if image_url:
                image_data = download_image(image_url)
        elif model_name == 'Stable Diffusion':
            stability_api_key = st.session_state.api_keys.get('stability')
            if stability_api_key:
                image_data = generate_image_stability(stability_api_key, desc, size)
            else:
                st.error("Stability AI API Key is required for Stable Diffusion.")
        elif model_name == 'Replicate':
            replicate_api_key = st.session_state.api_keys.get('replicate')
            if replicate_api_key:
                image_data = generate_image_replicate(replicate_api_key, desc)
            else:
                st.error("Replicate API Key is required for this model.")

        if image_data:
            images[f"image_{i+1}.png"] = image_data
        else:
            images[f"image_{i+1}.png"] = b""

    return images

def generate_image_dalle(api_key, prompt, size="1024x1024", hd=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
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
    except requests.RequestException as e:
        st.error(f"Error generating image with DALL·E: {e}")
        return None

def generate_image_stability(api_key, prompt, size="512x512"):
    stability_api = client.StabilityInference(
        key=api_key,
        verbose=True,
    )
    answers = stability_api.generate(
        prompt=prompt,
        width=int(size.split('x')[0]),
        height=int(size.split('x')[1]),
    )
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == client.FILTER:
                st.warning("Your request activated the API's safety filters and could not be processed.")
                return None
            if artifact.type == client.ARTIFACT_IMAGE:
                image_data = artifact.binary
                return image_data
    return None

def generate_image_replicate(api_key, prompt):
    try:
        replicate_client = replicate.Client(api_token=api_key)
        output_url = replicate_client.run(
            "stability-ai/stable-diffusion",
            input={"prompt": prompt}
        )
        image_data = requests.get(output_url[0]).content
        return image_data
    except Exception as e:
        st.error(f"Error generating image with Replicate: {e}")
        return None

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Error downloading image: {e}")
        return None

def create_gif(images, filter_type=None):
    st.info("Creating GIF...")
    try:
        pil_images = [Image.open(BytesIO(img)) for img in images.values() if img]
        if filter_type:
            pil_images = [apply_filter(img, filter_type) for img in pil_images]
        gif_buffer = BytesIO()
        pil_images[0].save(gif_buffer, format='GIF', save_all=True, append_images=pil_images[1:], duration=1000, loop=0)
        gif_buffer.seek(0)
        return gif_buffer
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return None

def apply_filter(image, filter_type):
    if filter_type == "sepia":
        sepia_image = ImageOps.colorize(image.convert("L"), "#704214", "#C0A080")
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

def generate_audio_logo(prompt, api_key):
    input_data = {
        "prompt": prompt,
        "model_version": "stereo-large",
        "output_format": "mp3",
        "normalization_strategy": "peak"
    }

    try:
        replicate_client = replicate.Client(api_token=api_key)
        output_url = replicate_client.run(
            "meta/musicgen",
            input=input_data
        )
        audio_data = requests.get(output_url).content
    except Exception as e:
        st.error(f"Error generating audio logo: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".mp3"
    file_data = audio_data

    return file_name, file_data

def generate_video_logo(prompt, api_key):
    headers = get_headers()
    data = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "url"
    }

    try:
        response = requests.post(DALLE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        image_data = requests.get(image_url).content
    except Exception as e:
        st.error(f"Error generating video logo image: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".png"
    file_data = image_data

    return file_name, file_data

def animate_image_to_video(image_file, prompt):
    stability_api_key = st.session_state.api_keys.get("stability")
    if not stability_api_key:
        st.error("Stability AI API Key is required for animating images to video.")
        return None

    url = STABILITY_API_URL

    image = Image.open(image_file)
    image = image.resize((768, 768))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

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
    stability_api_key = st.session_state.api_keys.get("stability")
    if not stability_api_key:
        st.error("Stability AI API Key is required to fetch generated video.")
        return None

    url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"

    while True:
        response = requests.get(
            url,
            headers={
                'accept': "video/*",
                'authorization': f"Bearer {stability_api_key}"
            }
        )

        if response.status_code == 202:
            st.write("Generation in-progress, trying again in 10 seconds...")
            time.sleep(10)
        elif response.status_code == 200:
            st.write("Generation complete!")
            return response.content
        else:
            st.error(f"Error fetching video: {response.text}")
            return None

def create_master_document(campaign_plan):
    master_doc = "Marketing Campaign Master Document\n\n"
    for key, value in campaign_plan.items():
        if key == "images":
            master_doc += f"{key.capitalize()}:\n"
            for img_key in value:
                master_doc += f" - {img_key}: See attached image.\n"
        else:
            master_doc += f"{key.replace('_', ' ').capitalize()}:\n\n{value}\n\n{'='*50}\n\n"
    return master_doc

def create_zip(content_dict):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, value in content_dict.items():
            if key == 'images' and isinstance(value, dict):
                for img_name, img_data in value.items():
                    zip_file.writestr(img_name, img_data)
            elif isinstance(value, bytes):
                zip_file.writestr(f"{key}", value)
            elif isinstance(value, str):
                zip_file.writestr(f"{key}.txt", value)
    zip_buffer.seek(0)
    return zip_buffer

def enhance_content(content, filename):
    api_key = st.session_state.api_keys.get('openai')
    if not api_key:
        st.warning("OpenAI API Key is required for content enhancement.")
        return content

    headers = get_headers()
    content_summary = ""

    if isinstance(content, bytes):
        try:
            df = pd.read_excel(BytesIO(content))
            content_summary = df.to_string()
        except ValueError:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                return "Skipped image file."
            else:
                try:
                    content_summary = content.decode('utf-8')
                except UnicodeDecodeError:
                    return "Error: Unable to read the provided file as text or Excel file."
    else:
        content_summary = content

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": f"Enhance and summarize the following content from {filename}."},
            {"role": "user", "content": content_summary}
        ],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        enhanced_content = response_data["choices"][0]["message"]["content"]
        return enhanced_content.strip()
    except Exception as e:
        st.error(f"Error enhancing content: {e}")
        return content

def delete_all_files():
    st.session_state["global_file_storage"] = {}
    st.session_state["chat_knowledge_base"] = {}
    st.success("All files and knowledge base entries have been deleted.")

def initialize_global_files():
    if not os.path.exists(GLOBAL_FILES_DIR):
        os.makedirs(GLOBAL_FILES_DIR)

# New Helper Functions from your code snippets

# Function to display image or video
def display_media(file_data, col):
    if file_data:
        try:
            if file_data.name.endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(file_data)
                col.image(image, use_column_width=True, caption=file_data.name)
            elif file_data.name.endswith('.mp4'):
                col.video(file_data, format="video/mp4")
        except Exception as e:
            col.error(f"Error displaying file: {e}")

# Function to encode image to base64
def encode_image(image_data):
    return base64.b64encode(image_data).decode('utf-8')

# Function to describe an image using GPT-4
def describe_image(api_key, image_data):
    encoded_image = encode_image(image_data)
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": f"What's in this image?\n![image](data:image/jpeg;base64,{encoded_image})"
            }
        ],
        "max_tokens": 1000
    }
    response = requests.post(CHAT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Failed to analyze the image: {response.text}")
        return None

# Function to analyze text content
def analyze_text(api_key, text_content):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": f"Analyze the following text:\n\n{text_content}"}
        ],
        "max_tokens": 1000
    }
    response = requests.post(CHAT_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Failed to analyze the text: {response.text}")
        return None

# Function to extract and analyze content from a zip file
def analyze_zip(api_key, zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        analysis_results = {}
        for file_info in z.infolist():
            with z.open(file_info) as f:
                if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_data = f.read()
                    description = describe_image(api_key, image_data)
                    analysis_results[file_info.filename] = description
                elif file_info.filename.lower().endswith('.txt'):
                    text_content = f.read().decode('utf-8')
                    analysis_results[file_info.filename] = analyze_text(api_key, text_content)
                elif file_info.filename.lower().endswith('.csv'):
                    df = pd.read_csv(f)
                    analysis_results[file_info.filename] = df.to_string()
                elif file_info.filename.lower().endswith('.pdf'):
                    reader = PdfReader(f)
                    text_content = ""
                    for page in reader.pages:
                        text_content += page.extract_text()
                    analysis_results[file_info.filename] = analyze_text(api_key, text_content)
    return analysis_results

# Function to stitch frames to video
def stitch_frames_to_video(frames_directory, output_filename, fps):
    frames = []
    frame_files = os.listdir(frames_directory)

    def try_parse_int(s, base=10, val=None):
        try:
            return int(s, base)
        except ValueError:
            return val

    frame_files.sort(key=lambda x: try_parse_int(x.split('_')[1].split('.')[0], val=float('inf')))

    for filename in frame_files:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            frames.append(cv2.imread(os.path.join(frames_directory, filename)))
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

# Function to generate video from text
def generate_video_from_text(api_key, prompts, settings, output_dir):
    context = client.StabilityInference(
        key=api_key,
        verbose=True,
    )

    args = AnimationArgs()
    args.interpolate_prompts = True
    args.locked_seed = settings.get('locked_seed', True)
    args.max_frames = settings.get('animation_length', 120)
    args.seed = settings.get('seed', 42)
    args.strength_curve = settings.get('strength_curve', "0:(0)")
    args.diffusion_cadence_curve = settings.get('diffusion_cadence_curve', "0:(4)")
    args.cadence_interp = settings.get('cadence_interp', "film")

    animation_prompts = prompts
    negative_prompt = settings.get('negative_prompt', "")

    animator = Animator(
        api_context=context,
        animation_prompts=animation_prompts,
        negative_prompt=negative_prompt,
        args=args,
        out_dir=output_dir
    )

    for _ in tqdm(animator.render(), total=args.max_frames):
        pass

    create_video_from_frames(animator.out_dir, os.path.join(output_dir, "video.mp4"), fps=settings.get('fps', 12))

    return os.path.join(output_dir, "video.mp4")

# Function to generate video in a separate thread
def threaded_generate_video(prompts, settings, video_label):
    st.session_state["is_generating"] = True
    api_key = st.session_state.api_keys.get("stability")
    if not api_key:
        st.error("Stability AI API Key is required for video generation.")
        return

    output_dir = "generated_videos"
    os.makedirs(output_dir, exist_ok=True)

    try:
        video_path = generate_video_from_text(api_key, prompts, settings, output_dir)
        st.session_state["generated_video"] = video_path
        video_label.video(video_path)
    except Exception as e:
        st.error(f"Error generating video: {e}")
    finally:
        st.session_state["is_generating"] = False

# Function to initiate video generation
def generate_video(prompts, settings, video_label):
    thread = threading.Thread(target=threaded_generate_video, args=(prompts, settings, video_label))
    thread.start()

# Tabs

def edit_videos_tab():
    st.title("🎞️ Edit Videos")

    # Upload section
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="upload_image_file")
    if uploaded_file:
        add_file_to_global_storage(uploaded_file.name, uploaded_file)

    # Text input for the prompt
    prompt = st.text_area("Enter prompt for video generation:", key="image_prompt_unique")

    # Display section
    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file:
            display_media(uploaded_file, col1)
            if uploaded_file.name.endswith(('.png', '.jpg', '.jpeg')):
                animate_button = st.button("Animate", key="animate_button")
                if animate_button:
                    with st.spinner("Animating..."):
                        generation_id = animate_image_to_video(uploaded_file, prompt)
                        if generation_id:
                            video_data = fetch_generated_video(generation_id)
                            if video_data:
                                st.session_state["generated_video"] = BytesIO(video_data)

    with col2:
        if "generated_video" in st.session_state:
            generated_file = st.session_state["generated_video"]
            col2.video(generated_file, format="video/mp4")

def multimedia_analysis_tab():
    st.title("🖼️ Multimedia Analysis with GPT-4")

    # API Key Input
    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        st.error("OpenAI API key is required. Please set it in the sidebar.")
        return

    # File Upload
    uploaded_file = st.file_uploader("Upload a file (Image, Text, Zip)", type=["png", "jpg", "jpeg", "txt", "csv", "pdf", "zip"])

    if uploaded_file:
        file_type = uploaded_file.type
        file_name = uploaded_file.name

        if file_type in ["image/png", "image/jpeg"]:
            description = describe_image(api_key, uploaded_file.read())
            st.write(f"Description for {file_name}:")
            st.write(description)
        elif file_type == "text/plain":
            text_content = uploaded_file.read().decode('utf-8')
            analysis = analyze_text(api_key, text_content)
            st.write(f"Analysis for {file_name}:")
            st.write(analysis)
        elif file_type == "application/zip":
            analysis_results = analyze_zip(api_key, uploaded_file)
            for file, analysis in analysis_results.items():
                st.write(f"Analysis for {file}:")
                st.write(analysis)
        else:
            st.error("Unsupported file type.")

def text_to_video_tab():
    st.title("🎥 Text to Video")

    prompt_fields = {
        "start": st.text_input("Start Animation Prompt", key="start_animation_prompt"),
        "mid1": st.text_input("Mid Animation Prompt 1", key="mid_animation_prompt1"),
        "mid2": st.text_input("Mid Animation Prompt 2", key="mid_animation_prompt2"),
        "mid3": st.text_input("Mid Animation Prompt 3", key="mid_animation_prompt3"),
        "end": st.text_input("End Animation Prompt", key="end_animation_prompt")
    }

    frame_fields = {
        "start_frame": st.number_input("Start Frame", min_value=0, value=0, key="start_frame"),
        "mid_frame1": st.number_input("Mid Frame 1", min_value=0, value=30, key="mid_frame1"),
        "mid_frame2": st.number_input("Mid Frame 2", min_value=0, value=60, key="mid_frame2"),
        "mid_frame3": st.number_input("Mid Frame 3", min_value=0, value=90, key="mid_frame3"),
        "end_frame": st.number_input("End Frame", min_value=0, value=120, key="end_frame")
    }

    settings_fields = {
        "animation_length": st.number_input("Animation Length (frames)", min_value=1, value=120, key="animation_length"),
        "fps": st.number_input("Frames per Second", min_value=1, value=12, key="fps"),
        "seed": st.number_input("Seed", value=42, key="seed"),
        "translation_x": st.text_input("Translation X", value="0:(0)", key="translation_x"),
        "translation_y": st.text_input("Translation Y", value="0:(0)", key="translation_y"),
        "translation_z": st.text_input("Translation Z", value="0:(0)", key="translation_z"),
        "preset": st.selectbox("Preset", ["default", "cinematic", "artistic"], key="preset"),
        "negative_prompt": st.text_input("Negative Prompt", value="nudity, naked, violence, blood, horror, watermark, logo, sex, guns", key="negative_prompt")
    }

    if st.button("Generate Video"):
        prompts = {
            frame_fields["start_frame"]: prompt_fields["start"],
            frame_fields["mid_frame1"]: prompt_fields["mid1"],
            frame_fields["mid_frame2"]: prompt_fields["mid2"],
            frame_fields["mid_frame3"]: prompt_fields["mid3"],
            frame_fields["end_frame"]: prompt_fields["end"]
        }
        settings = {key: settings_fields[key] for key in settings_fields}
        video_label = st.empty()
        with st.spinner("Generating video. Video in process..."):
            generate_video(prompts, settings, video_label)

def comic_book_generation_tab():
    st.title("📚 Comic Book Generation")

    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        st.error("OpenAI API key is required. Please set it in the sidebar.")
        return

    prompt = st.text_area("Enter the story or idea for your comic book:")

    if st.button("Generate Comic Book"):
        with st.spinner("Generating comic book..."):
            comic_content = generate_content("comic book script", prompt, "", {}, api_key)
            st.write("Comic Book Content:")
            st.write(comic_content)
            add_file_to_global_storage("comic_book.txt", comic_content)

def game_development_tab():
    st.title("🎮 Game Development Automation")

    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        st.error("OpenAI API key is required. Please set it in the sidebar.")
        return

    prompt = st.text_area("Enter your game idea or concept:")

    if st.button("Generate Game Plan"):
        with st.spinner("Generating game plan..."):
            game_plan = generate_content("game development plan", prompt, "", {}, api_key)
            st.write("Game Development Plan:")
            st.write(game_plan)
            add_file_to_global_storage("game_plan.txt", game_plan)

def business_plan_tab():
    st.title("📈 Business Plan Automation")

    api_key = st.session_state.api_keys.get("openai")
    if not api_key:
        st.error("OpenAI API key is required. Please set it in the sidebar.")
        return

    prompt = st.text_area("Describe your business idea:")

    if st.button("Generate Business Plan"):
        with st.spinner("Generating business plan..."):
            business_plan = generate_content("business plan", prompt, "", {}, api_key)
            st.write("Business Plan:")
            st.write(business_plan)
            add_file_to_global_storage("business_plan.txt", business_plan)

def media_generation_tab():
    st.header("🎬 Media Generation")
    st.write("Generate images, videos, and music using AI models.")

    media_type = st.selectbox("Select Media Type", ["Select", "Image Generation", "Video Generation", "Music Generation"])

    if media_type == "Image Generation":
        st.subheader("Image Generation")

        image_prompt = st.text_area("Enter an image prompt:")
        image_model = st.session_state.model_selections.get('image_generation', 'DALL·E')
        image_size = st.selectbox("Select Image Size", ["256x256", "512x512", "1024x1024"])
        hd_images = st.checkbox("Generate HD images")

        if st.button("Generate Image"):
            image_data = generate_images(st.session_state.api_keys['openai'], [image_prompt], [image_size], image_model, hd_images)
            if image_data:
                image_bytes = list(image_data.values())[0]
                if image_bytes:
                    st.image(image_bytes, caption="Generated Image")
                    add_file_to_global_storage(f"{image_prompt.replace(' ', '_')}.png", image_bytes)

    elif media_type == "Video Generation":
        st.subheader("Video Generation")

        video_prompt = st.text_area("Enter a video prompt:")
        video_model = st.session_state.model_selections.get('video_generation', 'Stability AI')
        if video_model == "Stability AI":
            stability_api_key = st.session_state.api_keys.get("stability")
            if not stability_api_key:
                st.warning("Stability AI API Key is required for video generation.")
            else:
                if st.button("Generate Video"):
                    image_url = generate_image_dalle(st.session_state.api_keys['openai'], video_prompt)
                    if image_url:
                        image_data = download_image(image_url)
                        if image_data:
                            generation_id = animate_image_to_video(BytesIO(image_data), video_prompt)
                            if generation_id:
                                video_data = fetch_generated_video(generation_id)
                                if video_data:
                                    st.video(video_data)
                                    add_file_to_global_storage(f"{video_prompt.replace(' ', '_')}.mp4", video_data)
        elif video_model == "RunwayML":
            st.info("RunwayML integration is coming soon.")

    elif media_type == "Music Generation":
        st.subheader("Music Generation")

        music_prompt = st.text_area("Enter a music prompt:")
        music_model = st.session_state.model_selections.get('music_generation', 'Replicate')
        if st.button("Generate Music"):
            replicate_api_key = st.session_state.api_keys.get("replicate")
            if not replicate_api_key:
                st.warning("Replicate API Key is required for music generation.")
            else:
                file_name, music_data = generate_audio_logo(music_prompt, replicate_api_key)
                if music_data:
                    st.audio(music_data)
                    add_file_to_global_storage(file_name, music_data)

def custom_workflows_tab():
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
                    add_file_to_global_storage(file_name, file_data)
            else:
                st.warning(f"Prompt for step {i + 1} is empty.")

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

def file_management_tab():
    st.title("📁 File Management")

    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        file_data = uploaded_file.read()
        add_file_to_global_storage(uploaded_file.name, file_data)
        st.success(f"Uploaded {uploaded_file.name}")

    st.subheader("Generate File")
    generation_prompt = st.text_input("Enter prompt to generate file (e.g., '/python Generate a script that says hello world'):")
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

def create_zip_of_global_files():
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name, file_data in st.session_state["global_file_storage"].items():
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return zip_buffer

def generate_file_with_gpt(prompt):
    api_keys = st.session_state.api_keys
    openai_api_key = api_keys.get("openai")

    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the sidebar.")
        return None, None

    specific_prompt = f"Please generate the following file content without any explanations or additional text:\n{prompt}"

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",
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

    except requests.RequestException as e:
        st.error(f"Error generating file: {e}")
        return None, None

    file_extension = ".txt"
    file_name = prompt.replace(" ", "_") + file_extension
    file_data = generated_text.encode("utf-8")

    return file_name, file_data

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
            st.selectbox("Image Generation Model", ["DALL·E", "Stable Diffusion", "Replicate"], key="image_generation_model")
            st.selectbox("Video Generation Model", ["Stability AI", "RunwayML"], key="video_generation_model")
            st.selectbox("Music Generation Model", ["Replicate"], key="music_generation_model")
            if st.button("💾 Save Model Selections"):
                st.session_state.model_selections['image_generation'] = st.session_state.image_generation_model
                st.session_state.model_selections['video_generation'] = st.session_state.video_generation_model
                st.session_state.model_selections['music_generation'] = st.session_state.music_generation_model
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

def main_tabs():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🧠 AI Content Generation",
        "🎬 Media Generation",
        "📂 Custom Workflows",
        "📁 File Management",
        "🎞️ Edit Videos",
        "🖼️ Multimedia Analysis",
        "📚 Comic Book Generation",
        "🎮 Game Development",
        "📈 Business Plan Automation"
    ])

    with tab1:
        generate_marketing_campaign_tab()

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

    with tab9:
        business_plan_tab()

def main():
    load_api_keys()
    initialize_global_files()
    sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
