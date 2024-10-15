# session.py

import streamlit as st

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
