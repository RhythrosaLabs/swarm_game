import streamlit as st
import requests
import zipfile
import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
from gtts import gTTS
import replicate
import time
from fpdf import FPDF
import json

# API setup
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
STABILITY_API_URL = "https://api.stability.ai/v2beta/image-to-video"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

# Function to get API keys from session state
def get_api_keys():
    return st.session_state.get("api_keys", {})

def add_file_to_global_storage(file_name, file_data):
    if "global_file_storage" not in st.session_state:
        st.session_state["global_file_storage"] = {}
    st.session_state["global_file_storage"][file_name] = file_data

def get_all_global_files():
    if "global_file_storage" in st.session_state:
        return st.session_state["global_file_storage"]
    return {}

def execute_workflow(workflow_name):
    workflow = st.session_state["workflow_files"].get(workflow_name)
    if not workflow:
        st.error(f"No workflow found with the name '{workflow_name}'.")
        return None, None

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for i, step in enumerate(workflow):
            file_name, file_data = generate_file_with_gpt(step["prompt"])
            if file_name and file_data:
                zipf.writestr(file_name, file_data)
            else:
                st.error(f"Failed to generate file for step {i + 1} of workflow '{workflow_name}'.")
                return None, None
    zip_buffer.seek(0)
    return f"{workflow_name}_workflow.zip", zip_buffer.getvalue()
    
def generate_schedule(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.RequestException as e:
        st.error(f"Error generating schedule: {e}")
        return None
        

def generate_document(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data['choices'][0]['message']['content']
        return generated_text
    except requests.RequestException as e:
        st.error(f"Error generating document: {e}")
        return None

# Function to add a file to the chat knowledge base
def add_to_chat_knowledge_base(file_name, file_data):
    if "chat_knowledge_base" not in st.session_state:
        st.session_state["chat_knowledge_base"] = {}
    st.session_state["chat_knowledge_base"][file_name] = file_data

# Other helper functions...

def generate_content(action, prompt, budget, platforms, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant specializing in {action}."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Budget: {budget}"},
            {"role": "user", "content": f"Platforms: {', '.join([k for k, v in platforms.items() if v])}"}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        if "choices" not in response_data or not response_data["choices"]:
            return "Error: No response from the API"
        return response_data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def generate_budget_spreadsheet(budget):
    try:
        budget_value = float(budget)
    except ValueError:
        budget_value = 100

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
    max_length = 100  # Example maximum length, adjust as needed
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

    days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
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

# Example usage
social_media_spreadsheet = generate_social_media_schedule("Campaign Concept", {
    "facebook": True,
    "twitter": True,
    "instagram": True,
    "linkedin": True
})

# Saving the spreadsheet to a file for demonstration
with open('social_media_schedule.xlsx', 'wb') as f:
    f.write(social_media_spreadsheet)
# Function to generate audio logo using Replicate API
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }

    try:
        response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
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
    stability_api_key = st.session_state["api_keys"]["stability"]
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
    stability_api_key = st.session_state["api_keys"]["stability"]
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
            st.session_state["generation_status"] = "Generation in-progress, try again in 10 seconds."
            time.sleep(10)
        elif response.status_code == 200:
            st.session_state["generation_status"] = "Generation complete!"
            return response.content
        else:
            st.error(f"Error fetching video: {response.text}")
            return None

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
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "style": "vivid",
        "response_format": "url"
    }
    try:
        response = requests.post(DALLE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        return image_url
    except requests.RequestException as e:
        st.error(f"RequestException generating image: {e}")
        return None

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"RequestException downloading image: {e}")
        return None

def resize_image(image_data, target_size):
    image = Image.open(BytesIO(image_data))
    resized_image = image.resize(target_size)
    image_buffer = BytesIO()
    resized_image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    return image_buffer.getvalue()

def create_gif(images, filter_type=None):
    st.info("Creating GIF...")
    try:
        pil_images = [Image.open(BytesIO(img)) for img in images]
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

def create_master_document(campaign_plan):
    master_doc = "Marketing Campaign Master Document\n\n"
    for key, value in campaign_plan.items():
        if key == "images":
            master_doc += f"{key.capitalize()}:\n"
            for img_key in value:
                master_doc += f" - {img_key}: See attached image.\n"
        else:
            master_doc += f"{key.replace('_', ' ').capitalize()}: See attached document.\n"
    return master_doc

def create_zip(content_dict):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, value in content_dict.items():
            if isinstance(value, str):
                zip_file.writestr(f"{key}.txt", value)
            elif isinstance(value, bytes):
                if key.endswith('.mp3') or key.endswith('.mp4'):
                    zip_file.writestr(key, value)
                else:
                    zip_file.writestr(f"{key}", value)
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        zip_file.writestr(f"{key}/{sub_key}.txt", sub_value)
                    elif isinstance(sub_value, bytes):
                        if sub_key.endswith('.mp3') or sub_key.endswith('.mp4'):
                            zip_file.writestr(f"{key}/{sub_key}", sub_value)
                        else:
                            zip_file.writestr(f"{key}/{sub_key}", sub_value)
    zip_buffer.seek(0)
    return zip_buffer

def enhance_content(content, filename):
    api_keys = get_api_keys()
    headers = {
        "Authorization": f"Bearer {api_keys['openai']}",
        "Content-Type": "application/json"
    }

    # Handle different types of content
    content_summary = ""
    if isinstance(content, bytes):
        try:
            # Attempt to read as Excel
            df = pd.read_excel(BytesIO(content))
            content_summary = df.to_string()
        except ValueError:
            # Skip image files by checking the file extension
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                return "Skipped image file."
            else:
                try:
                    # Attempt to read as text
                    content_summary = content.decode('utf-8')
                except UnicodeDecodeError:
                    return "Error: Unable to read the provided file as text or Excel file."
    else:
        content_summary = content

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": f"Enhance the following {filename} content:"},
            {"role": "user", "content": content_summary}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def analyze_and_enhance(uploaded_file):
    api_keys = get_api_keys()
    if not uploaded_file:
        st.warning("Please upload a marketing zip file first.")
        return

    enhanced_dir = "enhanced_files"
    os.makedirs(enhanced_dir, exist_ok=True)
    st.info("Starting analysis and enhancement process...")

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(enhanced_dir)

    for root, _, files in os.walk(enhanced_dir):
        for file in files:
            file_path = os.path.join(root, file)
            st.info(f"Analyzing {file}...")
            with open(file_path, 'rb') as f:
                content = f.read()

            enhanced_content = enhance_content(content, file)
            st.info(f"Enhanced {file}.")

            enhanced_file_path = os.path.join(enhanced_dir, file)
            with open(enhanced_file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)

    enhanced_zip_path = "enhanced_marketing_campaign.zip"
    with zipfile.ZipFile(enhanced_zip_path, 'w') as zipf:
        for root, _, files in os.walk(enhanced_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, enhanced_dir))

    st.success(f"Enhanced files saved and zipped at {enhanced_zip_path}")
    st.download_button(label="Download Enhanced ZIP", data=open(enhanced_zip_path, 'rb').read(), file_name="enhanced_marketing_campaign.zip")

def clipdrop_remove_background(api_key, image_data):
    url = f"https://clipdrop-api.co/remove-background/v1"
    headers = {
        "x-api-key": api_key,
    }
    files = {
        "image_file": ("image.png", image_data, "image/png"),
    }
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.content

def clipdrop_reimagine(api_key, image_data):
    url = f"https://clipdrop-api.co/reimagine/v1/reimagine"
    headers = {
        "x-api-key": api_key,
    }
    files = {
        "image_file": ("image.png", image_data, "image/png"),
    }
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.content

def clipdrop_remove_text(api_key, image_data):
    url = f"https://clipdrop-api.co/remove-text/v1"
    headers = {
        "x-api-key": api_key,
    }
    files = {
        "image_file": ("image.png", image_data, "image/png"),
    }
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.content

# Generate video from image using Stability AI
def generate_video_from_image(image_data):
    api_keys = get_api_keys()
    headers = {
        "Authorization": f"Bearer {api_keys['stability']}"
    }
    files = {
        "image": ("image.png", image_data, "image/png"),
    }
    data = {
        "cfg_scale": 1.8,
        "motion_bucket_id": 127
    }

    try:
        response = requests.post(STABILITY_API_URL, headers=headers, files=files, data=data)
        response.raise_for_status()
        response_data = response.json()
        generation_id = response_data.get('id')

        if not generation_id:
            st.error("Error: No generation ID received from Stability AI API")
            return None

        result_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
        while True:
            result_response = requests.get(result_url, headers=headers)
            if result_response.status_code == 202:
                st.info("Video generation in progress, please wait...")
                time.sleep(10)
            elif result_response.status_code == 200:
                st.info("Video generation complete.")
                return result_response.content
            else:
                st.error(f"Error fetching video: {result_response.status_code}")
                return None

    except requests.RequestException as e:
        st.error(f"Error generating video: {str(e)}")
        return None

# Generate audio using Replicate API
def generate_audio(prompt):
    api_keys = get_api_keys()
    headers = {
        "Authorization": f"Bearer {api_keys['replicate']}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "model_version": "stereo-large",
        "output_format": "mp3",
        "normalization_strategy": "peak"
    }
    try:
        response = requests.post(REPLICATE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        audio_url = response_data["output"][0]
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        return audio_response.content
    except requests.RequestException as e:
        st.error(f"RequestException generating audio: {e}")
        return None

# Generate text-to-speech audio
def generate_tts(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()

# Chat-related functions
def chat_with_gpt(prompt, uploaded_files):
    api_keys = get_api_keys()
    openai_api_key = api_keys.get("openai")

    if not openai_api_key:
        return "Error: OpenAI API key is not set."

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    file_contents = []
    for file in uploaded_files:
        if file in st.session_state:
            file_contents.append(st.session_state[file])
        else:
            file_contents.append(f"Content for {file} not found in session state.")

    knowledge_base_contents = [content for content in st.session_state.get("chat_knowledge_base", {}).values()]

    chat_history = st.session_state.get("chat_history", [])

    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}\n\nFiles:\n{file_contents}\n\nKnowledge Base:\n{knowledge_base_contents}"}
        ]
    }

    # Add chat history to the conversation
    for entry in chat_history:
        data["messages"].append(entry)

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        # Save chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response_data["choices"][0]["message"]["content"]})
        st.session_state["chat_history"] = chat_history

        return response_data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def display_chat_history():
    if "chat_history" in st.session_state:
        for entry in reversed(st.session_state["chat_history"]):
            if entry["role"] == "user":
                st.markdown(f"**You:** {entry['content']}")
            else:
                st.markdown(f"**Assistant:** {entry['content']}")

# Function to process text with GPT based on user action
def process_text_with_gpt(action, text, api_key):
    if not api_key:
        return {"result": "Error: OpenAI API key is not set."}

    action_prompts = {
        "Summarize": f"Summarize the following text:\n\n{text}",
        "Expand upon": f"Expand upon the following text:\n\n{text}",
        "Simplify": f"Simplify the following text:\n\n{text}",
        "Make casual": f"Make the following text casual:\n\n{text}",
        "Make formal": f"Make the following text formal:\n\n{text}",
        "Remove extra characters": f"Remove extra characters from the following text:\n\n{text}"
    }

    prompt = action_prompts.get(action, "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        return {"result": response_data['choices'][0]['message']['content']}
    except requests.RequestException as e:
        st.error(f"Error processing text with GPT: {e}")
        return {"result": text}

def get_api_keys():
    return st.session_state.get("api_keys", {})

def generate_file_with_gpt(prompt):
    api_keys = get_api_keys()
    openai_api_key = api_keys.get("openai")
    stability_api_key = api_keys.get("stability")
    replicate_api_key = api_keys.get("replicate")


    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the sidebar.")
        return None, None

    if prompt.startswith("//"):
        workflow_name = prompt[2:]
        if workflow_name in st.session_state.get("workflow_files", {}):
            return execute_workflow(workflow_name)
        else:
            st.error(f"Workflow '{workflow_name}' not found.")
            return None, None
    
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
        image_name, image_data = generate_image_with_dalle(specific_prompt, openai_api_key)
        if image_data:
            st.session_state["generated_image"] = BytesIO(image_data)
            generation_id = animate_image_to_video(image_data, specific_prompt)
            if generation_id:
                video_data = fetch_generated_video(generation_id)
                if video_data:
                    file_name = specific_prompt.replace(" ", "_") + ".mp4"
                    return file_name, video_data
        return None, None
    
    if prompt.startswith("/speak "):
        specific_prompt = prompt.replace("/speak ", "").strip()
        return generate_speech_with_gtts(specific_prompt)
    
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
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data['choices'][0]['message']['content']
        
        generated_text = generated_text.strip()
        if prompt.startswith("/python "):
            start_index = generated_text.find("import")
            generated_text = generated_text[start_index:]
        elif prompt.startswith("/html "):
            start_index = generated_text.find("<!DOCTYPE html>")
            generated_text = generated_text[start_index:]
        elif prompt.startswith("/js "):
            start_index = 0
            generated_text = generated_text[start_index:]
        elif prompt.startswith("/md "):
            start_index = 0
            generated_text = generated_text[start_index:]
        elif prompt.startswith("/pdf "):
            start_index = 0
            generated_text = generated_text[start_index:]
        elif prompt.startswith("/doc ") or prompt.startswith("/txt "):
            start_index = generated_text.find("\n") + 1
            generated_text = generated_text[start_index:]
        elif any(prompt.startswith(prefix) for prefix in ["/docx ", "/rtf ", "/csv ", "/json ", "/xml ", "/yaml ", "/ini ", "/log ", "/c ", "/cpp ", "/java ", "/xls ", "/xlsx ", "/ppt ", "/pptx ", "/bat ", "/sh ", "/ps1 "]):
            start_index = 0
            generated_text = generated_text[start_index:]
        
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
    
    file_name = prompt.split(" ", 1)[1].replace(" ", "_") + file_extension
    file_data = generated_text.encode("utf-8")
    
    return file_name, file_data


def generate_image_with_dalle(prompt, api_key):
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
        response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
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

def generate_music_with_replicate(prompt, api_key):
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

def animate_image_to_video(image_data, prompt):
    stability_api_key = st.session_state["api_keys"]["stability"]
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

# Function to fetch the generated video
def fetch_generated_video(generation_id):
    stability_api_key = st.session_state["api_keys"]["stability"]
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
            st.session_state["generation_status"] = "Generation in-progress, try again in 10 seconds."
            time.sleep(10)
        elif response.status_code == 200:
            st.session_state["generation_status"] = "Generation complete!"
            return response.content
        else:
            st.error(f"Error fetching video: {response.text}")
            return None

def create_zip_of_global_files():
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for file_name, file_data in st.session_state["global_file_storage"].items():
            zipf.writestr(file_name, file_data)
    zip_buffer.seek(0)
    return zip_buffer

def compile_to_pdf(campaign_plan):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for key, value in campaign_plan.items():
        if isinstance(value, str):
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, value)
        elif isinstance(value, bytes):
            try:
                image = Image.open(BytesIO(value))
                pdf.add_page()
                pdf.image(image, x=10, y=8, w=190)
            except Exception as e:
                st.error(f"Error adding image to PDF: {str(e)}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str):
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, sub_value)
                elif isinstance(sub_value, bytes):
                    try:
                        image = Image.open(BytesIO(sub_value))
                        pdf.add_page()
                        pdf.image(image, x=10, y=8, w=190)
                    except Exception as e:
                        st.error(f"Error adding image to PDF: {str(e)}")

    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
