
import streamlit as st
import requests
import json
import os
import zipfile
from io import BytesIO
from PIL import Image
import replicate
import base64 
import re

# Constants
CHAT_API_URL = "https://api.openai.com/v1/chat/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"
API_KEY_FILE = "api_keys.json"

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {'openai': None, 'replicate': None}

if 'customization' not in st.session_state:
    st.session_state.customization = {
        'image_types': ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI'],
        'script_types': ['Player', 'Enemy', 'Game Object', 'Level Background'],
        'image_count': {t: 0 for t in ['Character', 'Enemy', 'Background', 'Object', 'Texture', 'Sprite', 'UI']},
        'script_count': {t: 0 for t in ['Player', 'Enemy', 'Game Object', 'Level Background']},
        'use_replicate': {'generate_music': False},
        'code_types': {'unity': False, 'unreal': False, 'blender': False},
        'generate_elements': {
            'game_concept': True,
            'world_concept': True,
            'character_concepts': True,
            'plot': True,
            'storyline': False,
            'dialogue': False,
            'game_mechanics': False,
            'level_design': False
        },
        'image_model': 'dall-e-3',
        'chat_model': 'gpt-4o',
        'code_model': 'gpt-4o',
    }

# Load API keys from a file
def load_api_keys():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, 'r') as file:
            data = json.load(file)
            return data.get('openai'), data.get('replicate')
    return None, None

# Save API keys to a file
def save_api_keys(openai_key, replicate_key):
    with open(API_KEY_FILE, 'w') as file:
        json.dump({"openai": openai_key, "replicate": replicate_key}, file)

# Get headers for OpenAI API
def get_openai_headers():
    return {
        "Authorization": f"Bearer {st.session_state.api_keys['openai']}",
        "Content-Type": "application/json"
    }

# Generate content using selected chat model
def generate_content(prompt, role):
    if st.session_state.customization['chat_model'] in ['gpt-4', 'gpt-4o-mini']:
        data = {
            "model": st.session_state.customization['chat_model'],
            "messages": [
                {"role": "system", "content": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(CHAT_API_URL, headers=get_openai_headers(), json=data)
            response.raise_for_status()
            response_data = response.json()
            if "choices" not in response_data:
                error_message = response_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}"

            content_text = response_data["choices"][0]["message"]["content"]
            return content_text

        except requests.RequestException as e:
            return f"Error: Unable to communicate with the OpenAI API: {str(e)}"
    elif st.session_state.customization['chat_model'] == 'llama':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={
                    "prompt": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development.\n\nHuman: {prompt}\n\nAssistant:",
                    "temperature": 0.75,
                    "top_p": 0.9,
                    "max_length": 500,
                    "repetition_penalty": 1
                }
            )
            return ''.join(output)
        except Exception as e:
            return f"Error: Unable to generate content using Llama: {str(e)}"
    else:
        return "Error: Invalid chat model selected."

# Generate images using selected image model
# Generate images using selected image model
def generate_image(prompt, size, steps=25, guidance=3.0, interval=2.0):
    if st.session_state.customization['image_model'] == 'dall-e-3':
        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": f"{size[0]}x{size[1]}",
            "n": 1,
            "response_format": "url"
        }
        try:
            response = requests.post(DALLE_API_URL, headers=get_openai_headers(), json=data)
            response.raise_for_status()
            response_data = response.json()
            if "data" not in response_data:
                error_message = response_data.get("error", {}).get("message", "Unknown error")
                return f"Error: {error_message}"
            if not response_data["data"]:
                return "Error: No data returned from API."
            return response_data["data"][0]["url"]
        except requests.RequestException as e:
            return f"Error: Unable to generate image: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SD Flux-1':
        try:
            # Convert size to aspect ratio
            width, height = size
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9" if width / height > 1.7 else "3:2"
            else:
                aspect_ratio = "9:16" if height / width > 1.7 else "2:3"

            # Debug print statement to check API key
            print(f"Debug: Replicate API key: {st.session_state.api_keys['replicate'][:5]}...")

            # Initialize Replicate client with API key
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])

            output = client.run(
                "black-forest-labs/flux-pro",
                input={
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "steps": steps,
                    "guidance": guidance,
                    "interval": interval,
                    "safety_tolerance": 2,
                    "output_format": "png",
                    "output_quality": 100
                }
            )
            return output
        except Exception as e:
            return f"Error: Unable to generate image using SD Flux-1: {str(e)}"
    elif st.session_state.customization['image_model'] == 'SDXL Lightning':
        try:
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            output = client.run(
                "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
                input={"prompt": prompt}
            )
            return output[0] if output else None
        except Exception as e:
            return f"Error: Unable to generate image using SDXL Lightning: {str(e)}"
    else:
        return "Error: Invalid image model selected."

# Generate music using Replicate's MusicGen
def generate_music(prompt):
    try:
        client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
        output = client.run(
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
            input={
                "prompt": prompt,
                "model_version": "stereo-large",
                "output_format": "mp3",
                "normalization_strategy": "peak"
            }
        )
        if isinstance(output, str) and output.startswith("http"):
            return output
        else:
            return None
    except Exception as e:
        st.error(f"Error: Unable to generate music: {str(e)}")
        return None

# Generate multiple images based on customization settings
def generate_images(customization, game_concept):
    images = {}
    
    image_prompts = {
        'Character': "Create a highly detailed, front-facing character concept art for a 2D game...",
        'Enemy': "Design a menacing, front-facing enemy character concept art for a 2D game...",
        'Background': "Create a wide, highly detailed background image for a level of the game...",
        'Object': "Create a detailed object image for a 2D game...",
        'Texture': "Generate a seamless texture pattern...",
        'Sprite': "Create a game sprite sheet with multiple animation frames...",
        'UI': "Design a cohesive set of user interface elements for a 2D game..."
    }
    
    sizes = {
        'Character': (1024, 1024),
        'Enemy': (1024, 1024),
        'Background': (1024, 1024),
        'Object': (1024, 1024),
        'Texture': (1024, 1024),
        'Sprite': (1024, 1024),
        'UI': (1024, 1024)
    }

    for img_type in customization['image_types']:
        for i in range(customization['image_count'].get(img_type, 0)):
            prompt = f"{image_prompts[img_type]} The design should fit the following game concept: {game_concept}. Variation {i + 1}"
            size = sizes[img_type]
            
            image_url = generate_image(prompt, size)
            
            if image_url and not isinstance(image_url, str) and not image_url.startswith('Error'):
                images[f"{img_type.lower()}_image_{i + 1}"] = image_url
            else:
                images[f"{img_type.lower()}_image_{i + 1}"] = image_url

    return images

# Generate scripts based on customization settings and code types
def generate_scripts(customization, game_concept):
    script_descriptions = {
        'Player': "Create a comprehensive player character script for a 2D game. Include movement, input handling, and basic interactions.",
        'Enemy': "Develop a detailed enemy AI script for a 2D game. Include patrolling, player detection, and attack behaviors.",
        'Game Object': "Script a versatile game object that can be interacted with, collected, or activated by the player.",
        'Level Background': "Create a script to manage the level background in a 2D game, including parallax scrolling if applicable."
    }
    
    scripts = {}
    selected_code_types = customization['code_types']
    code_model = customization['code_model']

    for script_type in customization['script_types']:
        for i in range(customization['script_count'].get(script_type, 0)):
            for code_type, selected in selected_code_types.items():
                if selected:
                    if code_type == 'unity':
                        lang = 'csharp'
                        file_ext = '.cs'
                    elif code_type == 'unreal':
                        lang = 'cpp'
                        file_ext = '.cpp'
                    elif code_type == 'blender':
                        lang = 'python'
                        file_ext = '.py'
                    else:
                        continue  # Skip if it's an unknown code type
                    
                    desc = f"{script_descriptions[script_type]} The script should be for {code_type.capitalize()}. Generate ONLY the code, without any explanations or comments outside the code. Ensure the code is complete and can be directly used in a project."

                    if code_model in ['gpt-4o', 'gpt-4o-mini']:
                        script_code = generate_content(desc, "game development")
                    elif code_model == 'llama':
                        try:
                            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
                            output = client.run(
                                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                                input={
                                    "prompt": desc,
                                    "temperature": 0.7,
                                    "top_p": 0.95,
                                    "max_length": 2048,
                                    "repetition_penalty": 1.1
                                }
                            )
                            script_code = ''.join(output)
                        except Exception as e:
                            script_code = f"Error: Unable to generate script using Llama: {str(e)}"
                    else:
                        script_code = "Error: Invalid code model selected."

                    # Clean up the generated code
                    script_code = script_code.strip()
                    script_code = re.sub(r'^```\w*\n|```$', '', script_code, flags=re.MULTILINE)  # Remove code block markers
                    script_code = re.sub(r'^.*?Here\'s.*?:\n', '', script_code, flags=re.DOTALL)  # Remove introductory text
                    script_code = re.sub(r'\n+//.+?$', '', script_code, flags=re.MULTILINE)  # Remove trailing comments

                    scripts[f"{script_type.lower()}_{code_type}_script_{i + 1}{file_ext}"] = script_code

    return scripts

# Generate a complete game plan
def generate_game_plan(user_prompt, customization):
    game_plan = {}
    
    # Status updates
    status = st.empty()
    progress_bar = st.progress(0)
    
    def update_status(message, progress):
        status.text(message)
        progress_bar.progress(progress)

    # Generate game elements
    elements_to_generate = customization['generate_elements']
    for element, should_generate in elements_to_generate.items():
        if should_generate:
            update_status(f"Generating {element.replace('_', ' ')}...", 0.1)
            game_plan[element] = generate_content(f"Create a detailed {element.replace('_', ' ')} for the following game concept: {user_prompt}", "game design")
    
    # Generate images
    if any(customization['image_count'].values()):
        update_status("Generating game images...", 0.5)
        game_plan['images'] = generate_images(customization, game_plan.get('game_concept', ''))
    
    # Generate scripts
    if any(customization['script_count'].values()):
        update_status("Writing game scripts...", 0.7)
        game_plan['scripts'] = generate_scripts(customization, game_plan.get('game_concept', ''))
    
    # Optional: Generate music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", 0.9)
        music_prompt = f"Create background music for the game: {game_plan.get('game_concept', '')}"
        game_plan['music'] = generate_music(music_prompt)

    update_status("Game plan generation complete!", 1.0)

    return game_plan

# Function to display images
def display_image(image_url, caption):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad responses
        image = Image.open(BytesIO(response.content))
        st.image(image, caption=caption, use_column_width=True)
    except requests.RequestException as e:
        st.warning(f"Unable to load image: {caption}")
        st.error(f"Error: {str(e)}")
    except Exception as e:
        st.warning(f"Unable to display image: {caption}")
        st.error(f"Error: {str(e)}")

# Streamlit app layout
st.markdown('<p class="main-header">Game Dev Automation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Settings")

    # API Key Inputs
    with st.expander("API Keys"):
        openai_key = st.text_input("OpenAI API Key", value=st.session_state.api_keys['openai'], type="password")
        replicate_key = st.text_input("Replicate API Key", value=st.session_state.api_keys['replicate'], type="password")
        if st.button("Save API Keys"):
            save_api_keys(openai_key, replicate_key)
            st.session_state.api_keys['openai'] = openai_key
            st.session_state.api_keys['replicate'] = replicate_key
            st.success("API Keys saved successfully!")

    
    # Model Selection
    st.markdown("### AI Model Selection")
    st.session_state.customization['chat_model'] = st.selectbox(
        "Select Chat Model",
        options=['gpt-4o-mini', 'llama'],
        index=1  # Set default to gpt-4o-mini
    )
    st.session_state.customization['image_model'] = st.selectbox(
        "Select Image Generation Model",
        options=['dall-e-3', 'SD Flux-1', 'SDXL Lightning'],
        index=0
    )
    st.session_state.customization['code_model'] = st.selectbox(
        "Select Code Generation Model",
        options=['gpt-4o-mini', 'llama'],
        index=1  # Set default to gpt-4o-mini
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Game Concept", "Image Generation", "Script Generation", "Additional Elements"])

with tab1:
    st.markdown('<p class="section-header">Define Your Game</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Describe your game concept in detail. This will be used as the foundation for generating all other elements.</p>', unsafe_allow_html=True)
    user_prompt = st.text_area("Game Concept", "Enter a detailed description of your game here...", height=200)

with tab2:
    st.markdown('<p class="section-header">Image Generation</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Customize the types and number of images you want to generate for your game.</p>', unsafe_allow_html=True)
    
    for img_type in st.session_state.customization['image_types']:
        st.session_state.customization['image_count'][img_type] = st.number_input(
            f"Number of {img_type} Images", 
            min_value=0, 
            value=st.session_state.customization['image_count'][img_type]
        )

with tab3:
    st.markdown('<p class="section-header">Script Generation</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Specify the types and number of scripts you need for your game.</p>', unsafe_allow_html=True)
    
    for script_type in st.session_state.customization['script_types']:
        st.session_state.customization['script_count'][script_type] = st.number_input(
            f"Number of {script_type} Scripts", 
            min_value=0, 
            value=st.session_state.customization['script_count'][script_type]
        )

    st.markdown("### Code Type Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.customization['code_types']['unity'] = st.checkbox("Unity C# Scripts", value=st.session_state.customization['code_types']['unity'], key="unity")
    with col2:
        st.session_state.customization['code_types']['unreal'] = st.checkbox("Unreal C++ Scripts", value=st.session_state.customization['code_types']['unreal'], key="unreal")
    with col3:
        st.session_state.customization['code_types']['blender'] = st.checkbox("Blender Python Scripts", value=st.session_state.customization['code_types']['blender'], key="blender")

with tab4:
    st.markdown('<p class="section-header">Additional Game Elements</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Select additional elements to enhance your game design.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.customization['generate_elements']['storyline'] = st.checkbox("Detailed Storyline", value=st.session_state.customization['generate_elements']['storyline'])
        st.session_state.customization['generate_elements']['dialogue'] = st.checkbox("Sample Dialogue", value=st.session_state.customization['generate_elements']['dialogue'])
    with col2:
        st.session_state.customization['generate_elements']['game_mechanics'] = st.checkbox("Game Mechanics Description", value=st.session_state.customization['generate_elements']['game_mechanics'])
        st.session_state.customization['generate_elements']['level_design'] = st.checkbox("Level Design Document", value=st.session_state.customization['generate_elements']['level_design'])
    
    st.session_state.customization['use_replicate']['generate_music'] = st.checkbox("Generate Background Music", value=st.session_state.customization['use_replicate']['generate_music'])

# Generate Game Plan
if st.button("Generate Game Plan", key="generate_button"):
    if not st.session_state.api_keys['openai'] or not st.session_state.api_keys['replicate']:
        st.error("Please enter and save both OpenAI and Replicate API keys.")
    else:
        with st.spinner('Generating game plan...'):
            game_plan = generate_game_plan(user_prompt, st.session_state.customization)
        st.success('Game plan generated successfully!')

        # Display game plan results
        st.markdown('<p class="section-header">Generated Game Plan</p>', unsafe_allow_html=True)

        if 'game_concept' in game_plan:
            st.subheader("Game Concept")
            st.write(game_plan['game_concept'])

        if 'world_concept' in game_plan:
            st.subheader("World Concept")
            st.write(game_plan['world_concept'])

        if 'character_concepts' in game_plan:
            st.subheader("Character Concepts")
            st.write(game_plan['character_concepts'])

        if 'plot' in game_plan:
            st.subheader("Plot")
            st.write(game_plan['plot'])

        if 'images' in game_plan:
            st.subheader("Generated Assets")
            st.write("### Images")
            for img_name, img_url in game_plan['images'].items():
                if isinstance(img_url, str) and not img_url.startswith('Error'):
                    display_image(img_url, img_name)
                else:
                    st.write(f"{img_name}: {img_url}")

        if 'scripts' in game_plan:
            st.write("### Scripts")
            for script_name, script_code in game_plan['scripts'].items():
                with st.expander(f"View {script_name}"):
                    st.code(script_code, language=script_name.split('.')[-1])

        if 'additional_elements' in game_plan:
            st.subheader("Additional Game Elements")
            for element_name, element_content in game_plan['additional_elements'].items():
                with st.expander(f"View {element_name.capitalize()}"):
                    st.write(element_content)

        # Save results
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add text documents
            for key in ['game_concept', 'world_concept', 'character_concepts', 'plot']:
                if key in game_plan:
                    zip_file.writestr(f"{key}.txt", game_plan[key])
            
            # Add images
            if 'images' in game_plan:
                for asset_name, asset_url in game_plan['images'].items():
                    if isinstance(asset_url, str) and asset_url.startswith('http'):
                        img_response = requests.get(asset_url)
                        img = Image.open(BytesIO(img_response.content))
                        img_file_name = f"{asset_name}.png"
                        with BytesIO() as img_buffer:
                            img.save(img_buffer, format='PNG')
                            zip_file.writestr(img_file_name, img_buffer.getvalue())
            
            # Add scripts
            if 'scripts' in game_plan:
                for script_name, script_code in game_plan['scripts'].items():
                    zip_file.writestr(script_name, script_code)
            
            # Add additional elements
            if 'additional_elements' in game_plan:
                for element_name, element_content in game_plan['additional_elements'].items():
                    zip_file.writestr(f"{element_name}.txt", element_content)
            
            # Add music if generated
            if 'music' in game_plan and game_plan['music']:
                try:
                    music_response = requests.get(game_plan['music'])
                    music_response.raise_for_status()
                    zip_file.writestr("background_music.mp3", music_response.content)
                except requests.RequestException as e:
                    st.error(f"Error downloading music: {str(e)}")

        st.download_button(
            "Download Game Plan ZIP",
            zip_buffer.getvalue(),
            file_name="game_plan.zip",
            mime="application/zip",
            help="Download a ZIP file containing all generated assets and documents."
        )

        # Display generated music if applicable
        if 'music' in game_plan and game_plan['music']:
            st.subheader("Generated Music")
            st.audio(game_plan['music'], format='audio/mp3')
        else:
            st.warning("No music was generated or an error occurred during music generation.")

# Footer
st.markdown("---")
st.markdown("""
    Created by [Daniel Sheils](http://linkedin.com/in/danielsheils/) | 
    [GitHub](https://github.com/RhythrosaLabs/game-maker) | 
    [Twitter](https://twitter.com/rhythrosalabs) | 
    [Instagram](https://instagram.com/rhythrosalabs)
    """, unsafe_allow_html=True)

# Initialize Replicate client
if st.session_state.api_keys['replicate']:
    replicate.Client(api_token=st.session_state.api_keys['replicate'])

# Main execution
if __name__ == "__main__":
    # Load API keys
    openai_key, replicate_key = load_api_keys()
    if openai_key and replicate_key:
        st.session_state.api_keys['openai'] = openai_key
        st.session_state.api_keys['replicate'] = replicate_key
