
import streamlit as st
import json
import time
from helpers import chat_with_gpt, get_all_global_files, display_chat_history

def load_preset_bots():
    with open('presetBots.json') as f:
        return json.load(f)

def chat_tab(tab_key):
    st.header("Chat with GPT")

    use_personal_assistants = st.checkbox("Use Personal Assistants", key=f"use_personal_assistants_{tab_key}")

    preset_bots = load_preset_bots() if use_personal_assistants else None

    selected_bot = None
    if use_personal_assistants:
        categories = list(preset_bots.keys())
        selected_category = st.selectbox("Choose a category:", categories, key=f"category_select_{tab_key}")

        bots = preset_bots[selected_category]
        bot_names = [bot['name'] for bot in bots]
        selected_bot_name = st.selectbox("Choose a bot:", bot_names, key=f"bot_select_{tab_key}")

        selected_bot = next(bot for bot in bots if bot['name'] == selected_bot_name)
        bot_description = selected_bot.get('description', '')
        bot_instructions = selected_bot.get('instructions', '')

        st.write(f"**{selected_bot_name}**: {bot_description}")
        st.write(f"*Instructions*: {bot_instructions}")

    prompt = st.text_area("Enter your prompt here...", key=f"chat_prompt_{tab_key}")

    if st.button("Send", key=f"send_button_{tab_key}"):
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
            st.session_state[f"chat_response_{tab_key}"] = response

    # Display chat history
    st.markdown("### Chat History")
    display_chat_history()

if __name__ == "__main__":
    chat_tab("main")
