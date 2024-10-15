import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import base64

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

def generate_content(prompt, role):
    model = 'gpt-4o'  # Using GPT-4o
    headers = get_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a creative assistant specializing in {role}."},
            {"role": "user", "content": prompt}
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

def generate_image(prompt, size="512x512"):
    model = st.session_state.get('selected_image_model', 'dall-e')
    if model == 'dall-e':
        headers = get_headers()
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
            st.error(f"Error generating image with DALL·E: {e}")
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

def generate_video(prompt):
    st.info("Video generation is not yet implemented.")
    return None, None

def display_image(image_data, caption):
    image = Image.open(BytesIO(image_data))
    st.image(image, caption=caption, use_column_width=True)

def generate_audio(prompt):
    st.info("Music generation is not yet implemented.")
    return None, None

def describe_image(file_data):
    # Using GPT-4o with image understanding capabilities
    model = 'gpt-4o'
    headers = get_headers()
    base64_image = base64.b64encode(file_data).decode('utf-8')
    image_data_url = f"data:image/png;base64,{base64_image}"

    # Prepare the message content with the image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the content of this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            ]
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.5
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        description = response_data['choices'][0]['message']['content']
        return description.strip()
    except Exception as e:
        st.error(f"Error describing image: {e}")
        return None

def analyze_and_store_image(file_name, file_data):
    description = describe_image(file_data)
    if description:
        add_to_chat_knowledge_base(file_name, description)
        st.success(f"Image {file_name} analyzed and stored in knowledge base.")
    else:
        st.error(f"Failed to analyze and store image {file_name}.")

def generate_file_with_gpt(prompt):
    openai_api_key = st.session_state.api_keys.get("openai")
    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the API Keys tab.")
        return None, None

    specific_prompt = prompt.strip()

    # Identify file type based on prompt
    if prompt.startswith("/python "):
        file_extension = ".py"
        specific_prompt = prompt.replace("/python ", "").strip()
    elif prompt.startswith("/html "):
        file_extension = ".html"
        specific_prompt = prompt.replace("/html ", "").strip()
    elif prompt.startswith("/js "):
        file_extension = ".js"
        specific_prompt = prompt.replace("/js ", "").strip()
    elif prompt.startswith("/md "):
        file_extension = ".md"
        specific_prompt = prompt.replace("/md ", "").strip()
    elif prompt.startswith("/txt "):
        file_extension = ".txt"
        specific_prompt = prompt.replace("/txt ", "").strip()
    elif prompt.startswith("/pdf "):
        file_extension = ".pdf"
        specific_prompt = prompt.replace("/pdf ", "").strip()
    else:
        file_extension = ".txt"

    # Prepare the prompt for the model
    model_prompt = f"Generate a {file_extension} file with the following content:\n{specific_prompt}"

    model = 'gpt-4o'
    headers = get_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": model_prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        generated_content = response_data['choices'][0]['message']['content']

        # Remove any code blocks or markdown formatting
        if generated_content.startswith("```"):
            generated_content = generated_content.strip("```").strip()

        file_name = specific_prompt[:50].replace(" ", "_") + file_extension
        file_data = generated_content.encode("utf-8")

        return file_name, file_data
    except Exception as e:
        st.error(f"Error generating file: {e}")
        return None, None

def analyze_and_store_file(file_name, file_data):
    if file_name.lower().endswith(('.txt', '.py', '.html', '.md')):
        content = file_data.decode('utf-8')
        analyzed_content = enhance_content(content, file_name)
        add_to_chat_knowledge_base(file_name, analyzed_content)
        st.success(f"Analyzed and stored {file_name} in knowledge base.")
    elif file_name.lower().endswith('.zip'):
        with zipfile.ZipFile(BytesIO(file_data), 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename.lower().endswith(('.txt', '.py', '.html', '.md')):
                    with zip_ref.open(zip_info.filename) as f:
                        content = f.read().decode('utf-8')
                        analyzed_content = enhance_content(content, zip_info.filename)
                        add_to_chat_knowledge_base(zip_info.filename, analyzed_content)
                        st.success(f"Analyzed and stored {zip_info.filename} from {file_name} in knowledge base.")
    elif file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        analyze_and_store_image(file_name, file_data)

def enhance_content(content, filename):
    api_key = st.session_state.api_keys.get('openai')
    if not api_key:
        st.warning("OpenAI API Key is required for content enhancement.")
        return content

    model = 'gpt-4o'
    headers = get_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"Enhance and summarize the following content from {filename}."},
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
        return enhanced_content.strip()
    except Exception as e:
        st.error(f"Error enhancing content: {e}")
        return content

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

    # Add text field and button for generating files using GPT-4o
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

# Sidebar with Tabs: API Keys and Chat
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

            # Chat functionality in sidebar
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
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    display_chat_history()

            # Display chat history
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

    # Handle image files in the knowledge base
    images_in_kb = []
    for file_name, description in st.session_state.chat_knowledge_base.items():
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_data = st.session_state.global_file_storage.get(file_name)
            if file_data:
                base64_image = base64.b64encode(file_data).decode('utf-8')
                image_data_url = f"data:image/png;base64,{base64_image}"
                images_in_kb.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                })

    # Include images and files in the prompt
    if images_in_kb:
        data["messages"][-1]["content"] = [
            {"type": "text", "text": prompt},
            *images_in_kb
        ]

    try:
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        assistant_reply = response_data["choices"][0]["message"]["content"]

        # Save chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state["chat_history"] = chat_history

        return assistant_reply.strip()
    except Exception as e:
        st.error(f"Error in chat: {e}")
        return "I'm sorry, I couldn't process your request."

# Load preset bots
def load_preset_bots():
    if os.path.exists('presetBots.json'):
        with open('presetBots.json') as f:
            return json.load(f)
    else:
        return {}

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
            elif not prompt:
                st.warning("Please enter a topic or keywords.")
            else:
                if action == "Marketing Campaign":
                    generate_marketing_campaign(prompt)
                elif action == "Game Plan":
                    generate_game_plan(prompt)
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
                file_name = image_prompt.replace(" ", "_") + ".png"
                image_url_or_data = generate_image(image_prompt)
                if image_url_or_data:
                    if isinstance(image_url_or_data, bytes):
                        image_data = image_url_or_data
                    else:
                        image_data = requests.get(image_url_or_data).content
                    if image_data:
                        st.session_state.generated_images.append(image_data)
                        display_image(image_data, "Generated Image")
                        add_file_to_global_storage(file_name, image_data)
                        analyze_and_store_image(file_name, image_data)
        elif media_type == "Video Generation":
            video_prompt = st.text_area("Enter a video prompt:")
            if st.button("Generate Video"):
                file_name, video_data = generate_video(video_prompt)
                if video_data:
                    st.session_state.generated_videos.append(video_data)
                    st.video(video_data)
                    add_file_to_global_storage(file_name, video_data)
        elif media_type == "Music Generation":
            music_prompt = st.text_area("Enter a music prompt:")
            if st.button("Generate Music"):
                file_name, music_data = generate_audio(music_prompt)
                if music_data:
                    st.audio(music_data)
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

    # Tab 4: File Management
    with tab4:
        file_management_tab()

# Generate Marketing Campaign Function
def generate_marketing_campaign(prompt):
    st.info("Generating campaign concept...")
    campaign_concept = generate_content(f"Create a detailed marketing campaign concept based on the following prompt:\n\n{prompt}\n\nInclude target audience analysis, unique selling points, and overall strategy.", "marketing")
    st.session_state.campaign_plan['campaign_concept'] = campaign_concept
    add_file_to_global_storage("campaign_concept.txt", campaign_concept)

    st.info("Generating marketing plan...")
    marketing_plan = generate_content(f"Develop a comprehensive marketing plan for the campaign:\n\n{campaign_concept}\n\nInclude social media strategies, content ideas, and advertising channels.", "marketing")
    st.session_state.campaign_plan['marketing_plan'] = marketing_plan
    add_file_to_global_storage("marketing_plan.txt", marketing_plan)

    st.info("Generating social media posts...")
    social_media_posts = generate_content(f"Create a series of engaging social media posts for the campaign:\n\n{campaign_concept}\n\nInclude captions and hashtags for platforms like Instagram, Twitter, and Facebook.", "marketing")
    st.session_state.campaign_plan['social_media_posts'] = social_media_posts
    add_file_to_global_storage("social_media_posts.txt", social_media_posts)

    st.info("Generating email marketing content...")
    email_content = generate_content(f"Write compelling email marketing content for the campaign:\n\n{campaign_concept}\n\nInclude subject lines, body text, and calls to action.", "marketing")
    st.session_state.campaign_plan['email_content'] = email_content
    add_file_to_global_storage("email_content.txt", email_content)

    st.info("Generating images...")
    images = {}
    descriptions = {
        "banner": "Design a captivating banner image that represents the campaign's theme. No text.",
        "instagram_post": "Create an eye-catching Instagram post image that aligns with the campaign. No text.",
        "facebook_ad": "Produce an attractive image suitable for a Facebook ad for the campaign. No text."
    }
    sizes = {
        "banner": "1792x828",
        "instagram_post": "1080x1080",
        "facebook_ad": "1200x628"
    }
    for key, desc in descriptions.items():
        image_data = generate_image(f"{desc}\n\nCampaign concept: {campaign_concept}", sizes[key])
        if image_data:
            if isinstance(image_data, bytes):
                images[f"{key}.png"] = image_data
            else:
                image_content = requests.get(image_data).content
                if image_content:
                    images[f"{key}.png"] = image_content
            add_file_to_global_storage(f"{key}.png", images[f"{key}.png"])
            analyze_and_store_image(f"{key}.png", images[f"{key}.png"])
    st.session_state.campaign_plan['images'] = images

    st.info("Compiling all materials into a master document...")
    master_doc = create_master_document(st.session_state.campaign_plan)
    st.session_state.campaign_plan['master_document'] = master_doc
    add_file_to_global_storage("master_document.txt", master_doc)

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
        master_doc += f"{key.replace('_', ' ').title()}:\n\n{content_dict[key]}\n\n{'='*50}\n\n"
    return master_doc

def create_zip(content_dict):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for key, value in content_dict.items():
            if key == 'images' and isinstance(value, dict):
                for img_name, img_data in value.items():
                    zip_file.writestr(img_name, img_data)
            elif isinstance(value, str):
                zip_file.writestr(f"{key}.txt", value)
    zip_buffer.seek(0)
    return zip_buffer

# Generate Game Plan Function
def generate_game_plan(prompt):
    st.info("Generating game concept...")
    game_concept = generate_content(f"Develop an intricate game concept based on the following prompt:\n\n{prompt}\n\nInclude genre, storyline, main characters, and unique gameplay mechanics.", "game development")
    st.session_state.game_plan['game_concept'] = game_concept
    add_file_to_global_storage("game_concept.txt", game_concept)

    st.info("Creating game design document...")
    game_design_doc = generate_content(f"Create a comprehensive game design document for the following game concept:\n\n{game_concept}\n\nInclude detailed descriptions of gameplay mechanics, levels, character abilities, and progression systems.", "game development")
    st.session_state.game_plan['game_design_document'] = game_design_doc
    add_file_to_global_storage("game_design_document.txt", game_design_doc)

    st.info("Designing character profiles...")
    character_profiles = generate_content(f"Develop detailed character profiles for the main characters in the game:\n\n{game_concept}\n\nInclude backstories, personalities, and motivations.", "game development")
    st.session_state.game_plan['character_profiles'] = character_profiles
    add_file_to_global_storage("character_profiles.txt", character_profiles)

    st.info("Generating concept art descriptions...")
    art_descriptions = generate_content(f"Provide detailed descriptions for concept art of key scenes and characters in the game:\n\n{game_concept}", "game development")
    st.session_state.game_plan['art_descriptions'] = art_descriptions
    add_file_to_global_storage("art_descriptions.txt", art_descriptions)

    st.info("Generating images...")
    images = {}
    descriptions = art_descriptions.split('\n\n')
    for idx, desc in enumerate(descriptions):
        image_data = generate_image(desc)
        if image_data:
            if isinstance(image_data, bytes):
                images[f"concept_art_{idx+1}.png"] = image_data
            else:
                image_content = requests.get(image_data).content
                if image_content:
                    images[f"concept_art_{idx+1}.png"] = image_content
            add_file_to_global_storage(f"concept_art_{idx+1}.png", images[f"concept_art_{idx+1}.png"])
            analyze_and_store_image(f"concept_art_{idx+1}.png", images[f"concept_art_{idx+1}.png"])
    st.session_state.game_plan['images'] = images

    st.info("Compiling all materials into a master document...")
    master_doc = create_master_document(st.session_state.game_plan)
    st.session_state.game_plan['master_document'] = master_doc
    add_file_to_global_storage("game_master_document.txt", master_doc)

    st.success("Game Plan Generated!")
    st.download_button(
        label="Download Game Plan ZIP",
        data=create_zip(st.session_state.game_plan).getvalue(),
        file_name="game_plan.zip",
        mime="application/zip"
    )

# Main function
def main():
    load_api_keys()
    initialize_global_files()
    sidebar()
    main_tabs()

if __name__ == "__main__":
    main()
