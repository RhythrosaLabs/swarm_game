import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image

# Constants
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
API_KEY_FILE = "api_key.json"

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'action' not in st.session_state:
    st.session_state.action = None

# Load API key from a file
def load_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            data = json.load(file)
            return data.get('api_key')
    return None

# Save API key to a file
def save_api_key(api_key):
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"api_key": api_key}, file)

# Get headers for OpenAI API
def get_headers():
    return {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json"
    }

# Generate content using GPT-4
def generate_content(prompt, role):
    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant specializing in {role}."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(CHAT_API_URL, headers=get_headers(), json=data)
        response.raise_for_status()
        response_data = response.json()
        if "choices" not in response_data:
            error_message = response_data.get("error", {}).get("message", "Unknown error")
            return f"Error: {error_message}"

        content_text = response_data["choices"][0]["message"]["content"]
        return content_text

    except requests.RequestException as e:
        return f"Error: Unable to communicate with the OpenAI API."

# Generate image using DALL-E
def generate_image(prompt, size="1024x1024"):
    data = {
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": "url"
    }
    try:
        response = requests.post(DALLE_API_URL, headers=get_headers(), json=data)
        response.raise_for_status()
        response_data = response.json()
        image_url = response_data['data'][0]['url']
        return image_url
    except requests.RequestException as e:
        return None

# Download image from URL
def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        return None

# Create a zip file from the content
def create_zip(content_dict):
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

# Generate comic book
def generate_comic_book(prompt):
    comic_book = {}
    try:
        st.info("Generating comic book concept...")
        comic_concept = generate_content(f"Create a detailed comic book concept based on the following prompt: {prompt}.", "comic book creation")
        comic_book['comic_concept'] = comic_concept

        st.info("Generating detailed plot...")
        comic_book['plot'] = generate_content(f"Create a detailed plot for the comic book: {comic_concept}", "comic book plot creation")

        st.info("Generating character designs...")
        character_design_prompt = f"Create character designs for the comic book: {comic_concept}"
        image_url = generate_image(character_design_prompt)
        if image_url:
            image_data = download_image(image_url)
            if image_data:
                comic_book['character_design.png'] = image_data

        st.info("Generating comic panels...")
        panel_prompt = f"Create comic panels for the story based on the plot: {comic_concept}"
        image_url = generate_image(panel_prompt)
        if image_url:
            image_data = download_image(image_url)
            if image_data:
                comic_book['comic_panel.png'] = image_data

        st.info("Generating cover page...")
        cover_prompt = f"Create a cover page for the comic book: {comic_concept}"
        image_url = generate_image(cover_prompt)
        if image_url:
            image_data = download_image(image_url)
            if image_data:
                comic_book['cover_page.png'] = image_data

        st.info("Generating recap...")
        comic_book['recap'] = generate_content(f"Recap the comic book content: {comic_concept}", "comic book recap")

        st.info("Generating master document...")
        comic_book['master_document'] = create_master_document(comic_book)

        return comic_book

    except Exception as e:
        return f"Error during comic book generation: {str(e)}"

# Generate game plan
def generate_game_plan(prompt):
    game_plan = {}
    try:
        st.info("Generating game concept...")
        game_concept = generate_content(f"Invent a new 2D game concept with a detailed theme, setting, and unique features based on the following prompt: {prompt}. Ensure the game has WASD controls.", "game development")
        game_plan['game_concept'] = game_concept

        st.info("Generating world concept...")
        game_plan['world_concept'] = generate_content(f"Create a detailed world concept for the 2D game: {game_concept}", "game development")

        st.info("Generating character concepts...")
        game_plan['character_concepts'] = generate_content(f"Create detailed character concepts for the player and enemies in the 2D game: {game_concept}", "game development")

        st.info("Generating plot...")
        game_plan['plot'] = generate_content(f"Create a plot for the 2D game based on the world and characters of the game: {game_concept}", "game development")

        st.info("Generating dialogue...")
        game_plan['dialogue'] = generate_content(f"Write some dialogue for the 2D game based on the plot of the game: {game_concept}", "game development")

        st.info("Generating images...")
        images = {}
        descriptions = [
            f"Full-body, hyper-realistic character for a 2D game, with no background, in Unreal Engine style, based on the character descriptions: {game_plan['character_concepts']}",
            f"Full-body, hyper-realistic enemy character for a 2D game, with no background, in Unreal Engine style, based on the character descriptions: {game_plan['character_concepts']}",
            f"High-quality game object for the 2D game, with no background, in Unreal Engine style, based on the world concept: {game_plan['world_concept']}",
            f"High-quality level background for the 2D game, in Unreal Engine style, based on the world concept: {game_plan['world_concept']}"
        ]
        for i, desc in enumerate(descriptions, start=1):
            st.info(f"Generating image {i}...")
            image_url = generate_image(desc)
            if image_url:
                image_data = download_image(image_url)
                if image_data:
                    images[f"image_{i}.png"] = image_data
        game_plan['images'] = images

        st.info("Generating Unity scripts...")
        scripts = {}
        script_descriptions = [
            f"Unity script for the player character in a 2D game with WASD controls and space bar to jump or shoot, based on the character descriptions: {game_plan['character_concepts']}",
            f"Unity script for an enemy character in a 2D game with basic AI behavior, based on the character descriptions: {game_plan['character_concepts']}",
            f"Unity script for a game object in a 2D game, based on the world concept: {game_plan['world_concept']}",
            f"Unity script for the level background in a 2D game, based on the world concept: {game_plan['world_concept']}"
        ]
        for i, desc in enumerate(script_descriptions, start=1):
            st.info(f"Generating script {i}...")
            script_content = generate_content(desc, "Unity scripting")
            scripts[f"script_{i}.cs"] = script_content
        game_plan['unity_scripts'] = scripts

        st.info("Generating recap...")
        game_plan['recap'] = generate_content(f"Recap the game plan for the 2D game: {game_concept}", "game development")

        st.info("Generating master document...")
        game_plan['master_document'] = create_master_document(game_plan)

        return game_plan

    except Exception as e:
        return f"Error during game plan generation: {str(e)}"

# Generate marketing campaign
def generate_marketing_campaign(prompt):
    campaign_plan = {}
    try:
        st.info("Generating campaign concept...")
        campaign_concept = generate_content(f"Create a detailed marketing campaign concept based on the following prompt: {prompt}.", "marketing")

        campaign_plan['campaign_concept'] = campaign_concept

        st.info("Generating marketing plan...")
        campaign_plan['marketing_plan'] = generate_content(f"Create a detailed marketing plan for the campaign: {campaign_concept}", "marketing")

        st.info("Generating images...")
        images = {}
        descriptions = {
            "banner": "Wide banner image in a modern and appealing style, with absolutely no text, matching the theme of: " + campaign_concept,
            "instagram_background": "Tall background image suitable for Instagram video, with absolutely no text, matching the theme of: " + campaign_concept,
            "square_post_1": "Square background image for social media post, with absolutely no text, matching the theme of: " + campaign_concept,
        }
        sizes = {
            "banner": "1024x512",
            "instagram_background": "512x1024",
            "square_post_1": "512x512",
        }
        for key, desc in descriptions.items():
            st.info(f"Generating {key.replace('_', ' ')}...")
            image_url = generate_image(desc, sizes[key])
            if image_url:
                image_data = download_image(image_url)
                if image_data:
                    images[f"{key}.png"] = image_data
        campaign_plan['images'] = images

        st.info("Generating resources and tips...")
        campaign_plan['resources_tips'] = generate_content(f"List resources and tips for executing the marketing campaign: {campaign_concept}", "marketing")

        st.info("Generating recap...")
        campaign_plan['recap'] = generate_content(f"Recap the marketing campaign: {campaign_concept}", "marketing")

        st.info("Generating master document...")
        campaign_plan['master_document'] = create_master_document(campaign_plan)

        return campaign_plan

    except Exception as e:
        return f"Error during marketing campaign generation: {str(e)}"

# Create master document
def create_master_document(content_dict):
    master_doc = ""
    for key in content_dict.keys():
        if key == "images" or key == "unity_scripts":
            continue
        master_doc += f"{key.replace('_', ' ').capitalize()}:\n{content_dict[key]}\n\n"
    return master_doc

# Main Streamlit app
def main():
    st.title("Quick Actions Generator")
    st.write("Generate a Comic Book, Game Plan, or Marketing Campaign based on your input.")

    # API Key Input
    if not st.session_state.api_key:
        st.session_state.api_key = load_api_key()
    if not st.session_state.api_key:
        st.session_state.api_key = st.text_input("Enter your OpenAI API Key", type="password")
        if st.button("Save API Key"):
            save_api_key(st.session_state.api_key)
            st.success("API Key saved!")
    else:
        st.sidebar.write("API Key Loaded.")

    # User Input
    prompt = st.text_input("Enter your topic/keywords:")
    action = st.selectbox("Choose an action:", ["Select an action", "Comic Book", "Game Plan", "Marketing Campaign"])

    if st.button("Generate"):
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API Key.")
            return

        if action == "Select an action":
            st.error("Please select an action.")
            return

        if not prompt:
            st.error("Please enter a topic or keywords.")
            return

        if action == "Comic Book":
            result = generate_comic_book(prompt)
        elif action == "Game Plan":
            result = generate_game_plan(prompt)
        elif action == "Marketing Campaign":
            result = generate_marketing_campaign(prompt)
        else:
            result = None

        if isinstance(result, dict):
            st.success(f"{action} generated successfully!")
            # Display master document
            if 'master_document' in result:
                st.subheader("Master Document")
                st.write(result['master_document'])

            # Display images
            if 'images' in result:
                st.subheader("Generated Images")
                for img_name, img_data in result['images'].items():
                    st.image(Image.open(BytesIO(img_data)), caption=img_name)

            # Display scripts if any
            if 'unity_scripts' in result:
                st.subheader("Unity Scripts")
                for script_name, script_content in result['unity_scripts'].items():
                    st.code(script_content, language='csharp')

            # Create a zip file for download
            zip_buffer = create_zip(result)
            st.download_button(
                label="Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{action.replace(' ', '_').lower()}.zip",
                mime="application/zip"
            )
        else:
            st.error(result)

if __name__ == "__main__":
    main()
