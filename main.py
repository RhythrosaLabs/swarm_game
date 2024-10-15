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
    page_title="The Super-Powered Automation App",
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

# --------------------------
# Sidebar with Four Tabs
# --------------------------
def sidebar_menu():
    """Configure the sidebar with four tabs: Keys, Models, About, Chat."""
    
    # Ensure session state keys are initialized before using them
    initialize_session_state()

    # Sidebar layout
    with st.sidebar:
        # Use unique key for option_menu
        selected = option_menu(
            menu_title="Main Menu",
            options=["🔑 Keys", "🛠️ Models", "ℹ️ About", "💬 Chat"],
            icons=["key", "tools", "info-circle", "chat-dots"],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            key="main_menu"
        )

        if selected == "🔑 Keys":
            st.header("🔑 API Keys")
            st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_keys['openai'],
                type="password",
                key="openai_api_key_input"
            )
            st.text_input(
                "Replicate API Key",
                value=st.session_state.api_keys['replicate'],
                type="password",
                key="replicate_api_key_input"
            )
            st.text_input(
                "Stability AI API Key",
                value=st.session_state.api_keys['stability'],
                type="password",
                key="stability_api_key_input"
            )
            st.text_input(
                "Luma AI API Key",
                value=st.session_state.api_keys['luma'],
                type="password",
                key="luma_api_key_input"
            )
            st.text_input(
                "RunwayML API Key",
                value=st.session_state.api_keys['runway'],
                type="password",
                key="runway_api_key_input"
            )
            st.text_input(
                "Clipdrop API Key",
                value=st.session_state.api_keys['clipdrop'],
                type="password",
                key="clipdrop_api_key_input"
            )
            if st.button("💾 Save API Keys", key="save_api_keys_button"):
                st.session_state.api_keys['openai'] = st.session_state.openai_api_key_input
                st.session_state.api_keys['replicate'] = st.session_state.replicate_api_key_input
                st.session_state.api_keys['stability'] = st.session_state.stability_api_key_input
                st.session_state.api_keys['luma'] = st.session_state.luma_api_key_input
                st.session_state.api_keys['runway'] = st.session_state.runway_api_key_input
                st.session_state.api_keys['clipdrop'] = st.session_state.clipdrop_api_key_input
                save_api_keys()
                st.success("API Keys saved successfully!")

        elif selected == "🛠️ Models":
            st.header("🛠️ Models Selection")
            
            # Use unique keys for each selectbox
            st.subheader("Code Models")
            st.session_state['selected_code_model'] = st.selectbox(
                "Select Code Model",
                ["gpt-4o", "gpt-4", "llama"],
                index=["gpt-4o", "gpt-4", "llama"].index(st.session_state['selected_code_model']),
                key="select_code_model"
            )

            st.subheader("Image Models")
            st.session_state['selected_image_model'] = st.selectbox(
                "Select Image Model",
                ["dalle3", "stable diffusion", "flux"],
                index=["dalle3", "stable diffusion", "flux"].index(st.session_state['selected_image_model']),
                key="select_image_model"
            )

            st.subheader("Video Models")
            st.session_state['selected_video_model'] = st.selectbox(
                "Select Video Model",
                ["stable diffusion", "luma"],
                index=["stable diffusion", "luma"].index(st.session_state['selected_video_model']),
                key="select_video_model"
            )

            st.subheader("Audio Models")
            st.session_state['selected_audio_model'] = st.selectbox(
                "Select Audio Model",
                ["music gen"],
                index=["music gen"].index(st.session_state['selected_audio_model']),
                key="select_audio_model"
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
            prompt = st.text_area("Enter your prompt here...", key="chat_prompt")
            if st.button("Send", key="send_button"):
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

# Initialize session state for default values
def initialize_session_state():
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

# Helper function for chat with GPT
def chat_with_gpt(prompt):
    # Placeholder function for GPT-4o API integration
    return f"Response to: {prompt}"

# Display chat history
def display_chat_history():
    for entry in st.session_state.get('chat_history', []):
        st.write(f"{entry['role']}: {entry['content']}")


# Call the sidebar menu function
sidebar_menu()


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

# Call the main tabs function
main_tabs()


# --------------------------
# Main Function
# --------------------------
def main():
    """Main function to run the Streamlit app."""
    load_api_keys()
    sidebar_menu()
    main_tabs()

if __name__ == "__main__":
    main()
