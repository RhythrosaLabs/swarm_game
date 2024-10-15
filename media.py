# ui/tabs/media.py

import streamlit as st
from helpers.media_generation import generate_image
from helpers.file_management import add_file_to_global_storage, analyze_and_store_file
from helpers.file_management import display_image, download_image

def media_generation_tab():
    """Media Generation Tab"""
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
                            file_name = f"generated_image_{len(st.session_state.generated_images)+1}.png"
                            add_file_to_global_storage(file_name, image_data)
                            st.session_state.generated_images.append(image_data)
                            display_image(image_data, "Generated Image")
                            analyze_and_store_file(file_name, image_data)
    elif media_type == "Video Generation":
        video_prompt = st.text_area("Enter a video prompt:", key="video_generation_prompt")
        if st.button("Generate Video", key="generate_video_button"):
            if video_prompt.strip() == "":
                st.warning("Please enter a video prompt.")
            else:
                with st.spinner("Generating video..."):
                    from helpers.media_generation import generate_video_logo
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
                    from helpers.media_generation import generate_audio_logo
                    file_name, audio_data = generate_audio_logo(audio_prompt, st.session_state.api_keys.get("replicate"))
                    if audio_data:
                        add_file_to_global_storage(file_name, audio_data)
                        st.audio(audio_data, format="audio/mp3")
                        st.success(f"Generated audio: {file_name}")
