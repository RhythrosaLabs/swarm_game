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
        "model": "gpt-4o",
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

def generate_images(api_key, image_prompts, sizes, hd=False):
    images = {}

    for i, (desc, size) in enumerate(zip(image_prompts, sizes)):
        st.info(f"Generating image {i+1}...")
        image_url = generate_image(api_key, desc, size, hd)
        if image_url:
            try:
                image_data = download_image(image_url)
                if image_data:
                    images[f"image_{i+1}.png"] = image_data
                else:
                    images[f"image_{i+1}.png"] = b""
            except Exception as e:
                images[f"image_{i+1}.png"] = b""
                st.error(f"Error downloading image {i+1}: {str(e)}")
        else:
            images[f"image_{i+1}.png"] = b""
    return images

def generate_image(api_key, prompt, size="1024x1024", hd=False):
    quality = "hd" if hd else "standard"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "dall-e",
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
        st.error(f"Error generating image: {e}")
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
        pil_images = [Image.open(BytesIO(img)) for img in images.values()]
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
    headers = get_headers()
    data = {
        "model": "dall-e",
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
    except requests.RequestException as e:
        st.error(f"Error generating image: {e}")
        return None, None

    file_name = prompt.replace(" ", "_") + ".png"
    file_data = image_data

    return file_name, file_data

def animate_image_to_video(image_data, prompt):
    stability_api_key = st.session_state.api_keys["stability"]
    url = STABILITY_API_URL

    image = Image.open(BytesIO(image_data))
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
    stability_api_key = st.session_state.api_keys["stability"]
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
            st.session_state["generation_status"] = "Generation in-progress, trying again in 10 seconds..."
            time.sleep(10)
        elif response.status_code == 200:
            st.session_state["generation_status"] = "Generation complete!"
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
        "model": "gpt-4o",
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

def file_management_tab():
    st.title("📂 File Management")

    uploaded_file = st.file_uploader("Upload a file")
    if uploaded_file is not None:
        file_data = uploaded_file.read()
        add_file_to_global_storage(uploaded_file.name, file_data)
        analyze_and_store_file(uploaded_file.name, file_data)
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

def generate_marketing_campaign_tab():
    st.title("🧠 Generate Marketing Campaign")

    api_key = st.session_state.api_keys.get("openai")
    replicate_api_key = st.session_state.api_keys.get("replicate", None)

    if not api_key:
        st.warning("Please provide a valid OpenAI API Key.")
        return

    prompt = st.text_area("Prompt", "Describe your product or campaign...")

    budget = st.text_input("Budget", "1000")

    with st.expander("Advanced Options"):
        st.subheader("Social Media Platforms")
        platforms = {
            "facebook": st.checkbox("Facebook"),
            "twitter": st.checkbox("Twitter", value=True),
            "instagram": st.checkbox("Instagram"),
            "linkedin": st.checkbox("LinkedIn")
        }

        st.subheader("Image Tools")
        bypass_images = st.checkbox("Bypass image generation", value=True)

        image_size_options = {
            "Wide": "1792x1024",
            "Tall": "1024x1792",
            "Square": "1024x1024"
        }

        if not bypass_images:
            if "image_prompts" not in st.session_state:
                st.session_state["image_prompts"] = [""]
                st.session_state["image_sizes"] = ["Square"]

            for i in range(len(st.session_state["image_prompts"])):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.session_state["image_prompts"][i] = st.text_input(f"Image {i+1} Prompt:", st.session_state["image_prompts"][i])
                with cols[1]:
                    st.session_state["image_sizes"][i] = st.selectbox(f"Size {i+1}:", options=list(image_size_options.keys()), index=["Wide", "Tall", "Square"].index(st.session_state["image_sizes"][i]))
                with cols[2]:
                    if st.button("➖", key=f"remove_image_{i}"):
                        st.session_state["image_prompts"].pop(i)
                        st.session_state["image_sizes"].pop(i)
                        st.experimental_rerun()

            if len(st.session_state["image_prompts"]) < 5:
                if st.button("➕ Add Image"):
                    st.session_state["image_prompts"].append("")
                    st.session_state["image_sizes"].append("Square")

            hd_images = st.checkbox("Generate HD images")

            create_gif_checkbox = st.checkbox("Create GIF from images", value=False)
            filter_type = st.selectbox("Select GIF Filter:", ["None", "Sepia", "Greyscale", "Negative", "Solarize", "Posterize"])
            filter_type = filter_type.lower() if filter_type != "None" else None

        st.subheader("Other Settings")
        add_audio_logo = st.checkbox("Add audio logo")
        add_video_logo = st.checkbox("Add video logo")

    if st.button("Generate Marketing Campaign"):
        with st.spinner("Generating..."):
            campaign_plan = {}

            # Generate and analyze campaign concept
            st.info("Generating campaign concept...")
            campaign_concept = generate_content("campaign concept", prompt, budget, platforms, api_key)
            campaign_plan['campaign_concept'] = campaign_concept
            add_file_to_global_storage("campaign_concept.txt", campaign_concept)

            st.info("Analyzing campaign concept...")
            analyzed_concept = enhance_content(campaign_concept, "Campaign Concept")
            add_to_chat_knowledge_base("Campaign Concept", analyzed_concept)
            add_file_to_global_storage("analyzed_campaign_concept.txt", analyzed_concept)

            # Generate and analyze marketing plan
            st.info("Generating marketing plan...")
            marketing_plan = generate_content("marketing plan", prompt, budget, platforms, api_key)
            campaign_plan['marketing_plan'] = marketing_plan
            add_file_to_global_storage("marketing_plan.txt", marketing_plan)

            st.info("Analyzing marketing plan...")
            analyzed_plan = enhance_content(marketing_plan, "Marketing Plan")
            add_to_chat_knowledge_base("Marketing Plan", analyzed_plan)
            add_file_to_global_storage("analyzed_marketing_plan.txt", analyzed_plan)

            # Generate and analyze budget spreadsheet
            st.info("Generating budget spreadsheet...")
            budget_spreadsheet = generate_budget_spreadsheet(budget)
            campaign_plan['budget_spreadsheet'] = budget_spreadsheet
            add_file_to_global_storage("budget_spreadsheet.xlsx", budget_spreadsheet)

            st.info("Analyzing budget spreadsheet...")
            analyzed_budget = enhance_content(budget_spreadsheet, "Budget Spreadsheet")
            add_to_chat_knowledge_base("Budget Spreadsheet", analyzed_budget)
            add_file_to_global_storage("analyzed_budget_spreadsheet.txt", analyzed_budget)

            # Generate and analyze social media schedule
            st.info("Generating social media schedule...")
            social_media_schedule = generate_social_media_schedule(campaign_concept, platforms)
            campaign_plan['social_media_schedule'] = social_media_schedule
            add_file_to_global_storage("social_media_schedule.xlsx", social_media_schedule)

            st.info("Analyzing social media schedule...")
            analyzed_schedule = enhance_content(social_media_schedule, "Social Media Schedule")
            add_to_chat_knowledge_base("Social Media Schedule", analyzed_schedule)
            add_file_to_global_storage("analyzed_social_media_schedule.txt", analyzed_schedule)

            # Generate images if not bypassed
            if not bypass_images:
                st.info("Generating images...")
                custom_prompts = st.session_state["image_prompts"]
                image_sizes = [image_size_options[size] for size in st.session_state["image_sizes"]]
                images = generate_images(api_key, custom_prompts, image_sizes, hd_images)
                campaign_plan['images'] = images

                for image_key, image_data in images.items():
                    st.info(f"Analyzing {image_key}...")
                    analyzed_image = enhance_content(image_data, image_key)
                    add_to_chat_knowledge_base(image_key, analyzed_image)
                    add_file_to_global_storage(image_key, image_data)

                if create_gif_checkbox and images:
                    st.info("Creating GIF...")
                    gif_data = create_gif(images, filter_type)
                    campaign_plan['images']['animated_gif.gif'] = gif_data.getvalue()
                    add_file_to_global_storage("animated_gif.gif", gif_data.getvalue())

            # Generate audio logo if selected and replicate API key is provided
            if add_audio_logo:
                if replicate_api_key:
                    st.info("Generating audio logo...")
                    audio_prompt = f"Generate an audio logo for the following campaign concept: {campaign_concept}"
                    file_name, audio_data = generate_audio_logo(audio_prompt, replicate_api_key)
                    if audio_data:
                        campaign_plan['audio_logo'] = audio_data
                        add_file_to_global_storage(file_name, audio_data)
                else:
                    st.warning("Replicate API Key is required to generate an audio logo.")

            # Generate video logo if selected
            if add_video_logo:
                st.info("Generating video logo...")
                video_prompt = f"Generate a video logo for the following campaign concept: {campaign_concept}"
                file_name, video_logo_data = generate_video_logo(video_prompt, api_key)
                if video_logo_data:
                    st.info("Animating video logo...")
                    generation_id = animate_image_to_video(video_logo_data, video_prompt)
                    if generation_id:
                        video_data = fetch_generated_video(generation_id)
                        if video_data:
                            campaign_plan['video_logo'] = video_data
                            add_file_to_global_storage("video_logo.mp4", video_data)

            # Generate and analyze resources and tips
            st.info("Generating resources and tips...")
            resources_tips = generate_content("resources and tips", prompt, budget, platforms, api_key)
            campaign_plan['resources_tips'] = resources_tips
            add_file_to_global_storage("resources_tips.txt", resources_tips)

            st.info("Analyzing resources and tips...")
            analyzed_resources = enhance_content(resources_tips, "Resources and Tips")
            add_to_chat_knowledge_base("Resources and Tips", analyzed_resources)
            add_file_to_global_storage("analyzed_resources_tips.txt", analyzed_resources)

            # Generate and analyze recap
            st.info("Generating recap...")
            recap = generate_content("recap", prompt, budget, platforms, api_key)
            campaign_plan['recap'] = recap
            add_file_to_global_storage("recap.txt", recap)

            st.info("Analyzing recap...")
            analyzed_recap = enhance_content(recap, "Recap")
            add_to_chat_knowledge_base("Recap", analyzed_recap)
            add_file_to_global_storage("analyzed_recap.txt", analyzed_recap)

            st.info("Generating master document...")
            master_document = create_master_document(campaign_plan)
            campaign_plan['master_document'] = master_document
            add_file_to_global_storage("master_document.txt", master_document)

            st.info("Packaging into ZIP...")
            zip_data = create_zip(campaign_plan)

            st.session_state.campaign_plan = campaign_plan
            st.success("Marketing Campaign Generated")
            st.download_button(label="Download ZIP", data=zip_data.getvalue(), file_name="marketing_campaign.zip", key="download_campaign_zip")

def sidebar():
    with st.sidebar:
        tab = st.radio("Sidebar", ["🔑 API Keys", "💬 Chat"], key="sidebar_tab")

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
    model = 'gpt-4o'
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 AI Content Generation",
        "🎬 Media Generation",
        "📂 Custom Workflows",
        "📁 File Management"
    ])

    with tab1:
        generate_marketing_campaign_tab()

    with tab2:
        st.header("🎬 Media Generation")
        st.write("Generate images and videos using AI models.")
        media_type = st.selectbox("Select Media Type", ["Select", "Image Generation", "Video Generation", "Music Generation"])
        if media_type == "Image Generation":
            image_prompt = st.text_area("Enter an image prompt:")
            if st.button("Generate Image"):
                file_name = image_prompt.replace(" ", "_") + ".png"
                image_url_or_data = generate_image(st.session_state.api_keys['openai'], image_prompt)
                if image_url_or_data:
                    image_data = download_image(image_url_or_data)
                    if image_data:
                        st.session_state.generated_images.append(image_data)
                        display_image(image_data, "Generated Image")
                        add_file_to_global_storage(file_name, image_data)
                        analyze_and_store_image(file_name, image_data)
        elif media_type == "Video Generation":
            video_prompt = st.text_area("Enter a video prompt:")
            if st.button("Generate Video"):
                file_name, video_data = generate_video_with_replicate(video_prompt, st.session_state.api_keys.get("replicate"))
                if video_data:
                    st.session_state.generated_videos.append(video_data)
                    st.video(video_data)
                    add_file_to_global_storage(file_name, video_data)
        elif media_type == "Music Generation":
            music_prompt = st.text_area("Enter a music prompt:")
            if st.button("Generate Music"):
                file_name, music_data = generate_music_with_replicate(music_prompt, st.session_state.api_keys.get("replicate"))
                if music_data:
                    st.audio(music_data)
                    add_file_to_global_storage(file_name, music_data)

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

    with tab4:
        file_management_tab()

def main():
    load_api_keys()
    initialize_global_files()
    sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
