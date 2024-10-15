# ui/sidebar.py

import streamlit as st
from streamlit_option_menu import option_menu
from helpers.api_keys import load_api_keys, save_api_keys

def sidebar_menu():
    """Configure the sidebar with four tabs: Keys, Models, About, Chat."""
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
                index=["gpt-4o", "gpt-4", "llama"].index(st.session_state.selected_code_model),
                key="selected_code_model"
            )

            st.subheader("Image Models")
            st.session_state['selected_image_model'] = st.selectbox(
                "Select Image Model",
                ["dalle3", "stable diffusion", "flux"],
                index=["dalle3", "stable diffusion", "flux"].index(st.session_state.selected_image_model),
                key="selected_image_model"
            )

            st.subheader("Video Models")
            st.session_state['selected_video_model'] = st.selectbox(
                "Select Video Model",
                ["stable diffusion", "luma"],
                index=["stable diffusion", "luma"].index(st.session_state.selected_video_model),
                key="selected_video_model"
            )

            st.subheader("Audio Models")
            st.session_state['selected_audio_model'] = st.selectbox(
                "Select Audio Model",
                ["music gen"],
                index=["music gen"].index(st.session_state.selected_audio_model),
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
                        from helpers.chat import chat_with_gpt
                        response = chat_with_gpt(prompt)
                        if response:
                            st.session_state.chat_history.append({"role": "user", "content": prompt})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            from helpers.file_management import display_chat_history
                            display_chat_history()

            st.markdown("### Chat History")
            from helpers.file_management import display_chat_history
            display_chat_history()
