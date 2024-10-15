import streamlit as st
import os
import json
import zipfile
from io import BytesIO
from PIL import Image
import requests
import replicate
import re
import openai

# Constants
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
        'chat_model': 'gpt-4',
        'code_model': 'gpt-4',
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

# Set OpenAI API key
def set_openai_api_key():
    openai.api_key = st.session_state.api_keys['openai']

# Generate content using OpenAI GPT-4
def generate_content(prompt, role):
    try:
        response = openai.ChatCompletion.create(
            model=st.session_state.customization['chat_model'],
            messages=[
                {"role": "system", "content": f"You are a highly skilled assistant specializing in {role}. Provide detailed, creative, and well-structured responses optimized for game development."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            n=1,
            stop=None,
        )
        content_text = response['choices'][0]['message']['content'].strip()
        return content_text
    except Exception as e:
        return f"Error: Unable to generate content: {str(e)}"

# Generate images using selected image model
def generate_image(prompt, size):
    try:
        if st.session_state.customization['image_model'] == 'dall-e-3':
            # Using DALL-E 3 via OpenAI API
            headers = {
                "Authorization": f"Bearer {st.session_state.api_keys['openai']}",
                "Content-Type": "application/json"
            }
            data = {
                "prompt": prompt,
                "n": 1,
                "size": f"{size[0]}x{size[1]}",
                "response_format": "url"
            }
            response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["data"][0]["url"]
        else:
            # Using Replicate models
            client = replicate.Client(api_token=st.session_state.api_keys['replicate'])
            if st.session_state.customization['image_model'] == 'SD Flux-1':
                output = client.run(
                    "black-forest-labs/flux-pro",
                    input={
                        "prompt": prompt,
                        "steps": 25,
                        "guidance": 3.0,
                        "interval": 2.0
                    }
                )
                return output
            elif st.session_state.customization['image_model'] == 'SDXL Lightning':
                output = client.run(
                    "bytedance/sdxl-lightning-4step:5f24084160c9089501c1b3545d9be3c27883ae2239b6f412990e82d4a6210f8f",
                    input={"prompt": prompt}
                )
                return output[0] if output else None
    except Exception as e:
        return f"Error: Unable to generate image: {str(e)}"

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
        return f"Error: Unable to generate music: {str(e)}"

# Generate multiple images based on customization settings
def generate_images(customization, game_concept):
    images = {}
    
    image_prompts = {
        'Character': f"Create a highly detailed, front-facing character concept art for a 2D game based on the concept: {game_concept}.",
        'Enemy': f"Design a menacing, front-facing enemy character concept art for a 2D game based on the concept: {game_concept}.",
        'Background': f"Create a wide, highly detailed background image for a level of the game based on the concept: {game_concept}.",
        'Object': f"Create a detailed object image for a 2D game based on the concept: {game_concept}.",
        'Texture': f"Generate a seamless texture pattern for a 2D game based on the concept: {game_concept}.",
        'Sprite': f"Create a game sprite sheet with multiple animation frames for a 2D game based on the concept: {game_concept}.",
        'UI': f"Design a cohesive set of user interface elements for a 2D game based on the concept: {game_concept}."
    }
    
    sizes = {
        'Character': (1024, 1024),
        'Enemy': (1024, 1024),
        'Background': (1920, 1080),
        'Object': (512, 512),
        'Texture': (512, 512),
        'Sprite': (1024, 1024),
        'UI': (800, 600)
    }

    for img_type in customization['image_types']:
        count = customization['image_count'].get(img_type, 0)
        for i in range(count):
            prompt = f"{image_prompts[img_type]} Variation {i + 1}."
            size = sizes[img_type]
            image_url = generate_image(prompt, size)
            images[f"{img_type.lower()}_image_{i + 1}"] = image_url
    
    return images

# Generate scripts based on customization settings and code types
def generate_scripts(customization, game_concept):
    scripts = {}
    
    script_descriptions = {
        'Player': f"Create a comprehensive player character script for a 2D game based on the concept: {game_concept}. Include movement, input handling, and basic interactions.",
        'Enemy': f"Develop a detailed enemy AI script for a 2D game based on the concept: {game_concept}. Include patrolling, player detection, and attack behaviors.",
        'Game Object': f"Script a versatile game object that can be interacted with, collected, or activated by the player in a 2D game based on the concept: {game_concept}.",
        'Level Background': f"Create a script to manage the level background in a 2D game based on the concept: {game_concept}, including parallax scrolling if applicable."
    }
    
    for script_type in customization['script_types']:
        count = customization['script_count'].get(script_type, 0)
        for i in range(count):
            for code_type, selected in customization['code_types'].items():
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
                    
                    prompt = f"{script_descriptions[script_type]} The script should be written in {code_type.capitalize()}. Generate ONLY the code, without any explanations or comments outside the code. Ensure the code is complete and can be directly used in a project."
                    script_code = generate_content(prompt, "game development")
                    
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
    
    # Step 1: Generate Game Elements
    elements_to_generate = customization['generate_elements']
    total_elements = sum(1 for v in elements_to_generate.values() if v)
    current_progress = 0
    for element, should_generate in elements_to_generate.items():
        if should_generate:
            update_status(f"Generating {element.replace('_', ' ')}...", current_progress / (total_elements + 2))
            prompt = f"Create a detailed {element.replace('_', ' ')} for the following game concept: {user_prompt}"
            response = generate_content(prompt, "game design")
            game_plan[element] = response
            current_progress += 1
    
    # Step 2: Generate Images
    if any(customization['image_count'].values()):
        update_status("Generating game images...", (current_progress + 1) / (total_elements + 2))
        game_concept = game_plan.get('game_concept', user_prompt)
        images = generate_images(customization, game_concept)
        game_plan['images'] = images
        current_progress += 1
    
    # Step 3: Generate Scripts
    if any(customization['script_count'].values()):
        update_status("Writing game scripts...", (current_progress + 1) / (total_elements + 2))
        scripts = generate_scripts(customization, game_concept)
        game_plan['scripts'] = scripts
        current_progress += 1
    
    # Step 4: Optional - Generate Music
    if customization['use_replicate']['generate_music']:
        update_status("Composing background music...", (current_progress + 1) / (total_elements + 2))
        music_prompt = f"Create background music for the game: {game_concept}"
        music_url = generate_music(music_prompt)
        game_plan['music'] = music_url
        current_progress += 1
    
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
st.set_page_config(page_title="Game Dev Automation", layout="wide")
st.markdown('<h1 style="text-align: center; color: #4B0082;">Game Dev Automation</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API Key Inputs
    with st.expander("API Keys"):
        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
        replicate_key = st.text_input("Replicate API Key", type="password", placeholder="Enter your Replicate API key")
        if st.button("Save API Keys"):
            save_api_keys(openai_key, replicate_key)
            st.session_state.api_keys['openai'] = openai_key
            st.session_state.api_keys['replicate'] = replicate_key
            st.success("API Keys saved successfully!")
    
    # Load existing API keys on startup
    if st.session_state.api_keys['openai'] is None or st.session_state.api_keys['replicate'] is None:
        loaded_openai, loaded_replicate = load_api_keys()
        if loaded_openai:
            st.session_state.api_keys['openai'] = loaded_openai
        if loaded_replicate:
            st.session_state.api_keys['replicate'] = loaded_replicate
    
    # Set OpenAI API key
    if st.session_state.api_keys['openai']:
        set_openai_api_key()
    
    # Model Selection
    st.subheader("AI Model Selection")
    st.session_state.customization['chat_model'] = st.selectbox(
        "Select Chat Model",
        options=['gpt-4', 'gpt-3.5-turbo'],
        index=0
    )
    st.session_state.customization['image_model'] = st.selectbox(
        "Select Image Generation Model",
        options=['dall-e-3', 'SD Flux-1', 'SDXL Lightning'],
        index=0
    )
    st.session_state.customization['code_model'] = st.selectbox(
        "Select Code Generation Model",
        options=['gpt-4', 'gpt-3.5-turbo'],
        index=0
    )

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Game Concept", "Image Generation", "Script Generation", "Marketing Elements"])

with tab1:
    st.markdown('<h2 style="color: #4B0082;">Define Your Game</h2>', unsafe_allow_html=True)
    st.markdown('<p>Describe your game concept in detail. This will be used as the foundation for generating all other elements.</p>', unsafe_allow_html=True)
    user_prompt = st.text_area("Game Concept", "Enter a detailed description of your game here...", height=200)

with tab2:
    st.markdown('<h2 style="color: #4B0082;">Image Generation</h2>', unsafe_allow_html=True)
    st.markdown('<p>Customize the types and number of images you want to generate for your game.</p>', unsafe_allow_html=True)
    
    for img_type in st.session_state.customization['image_types']:
        st.session_state.customization['image_count'][img_type] = st.number_input(
            f"Number of {img_type} Images", 
            min_value=0, 
            value=st.session_state.customization['image_count'][img_type],
            key=img_type
        )

with tab3:
    st.markdown('<h2 style="color: #4B0082;">Script Generation</h2>', unsafe_allow_html=True)
    st.markdown('<p>Specify the types and number of scripts you need for your game.</p>', unsafe_allow_html=True)
    
    for script_type in st.session_state.customization['script_types']:
        st.session_state.customization['script_count'][script_type] = st.number_input(
            f"Number of {script_type} Scripts", 
            min_value=0, 
            value=st.session_state.customization['script_count'][script_type],
            key=script_type
        )
    
    st.markdown("### Code Type Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.customization['code_types']['unity'] = st.checkbox("Unity C# Scripts", value=st.session_state.customization['code_types']['unity'], key="unity_checkbox")
    with col2:
        st.session_state.customization['code_types']['unreal'] = st.checkbox("Unreal C++ Scripts", value=st.session_state.customization['code_types']['unreal'], key="unreal_checkbox")
    with col3:
        st.session_state.customization['code_types']['blender'] = st.checkbox("Blender Python Scripts", value=st.session_state.customization['code_types']['blender'], key="blender_checkbox")

with tab4:
    st.markdown('<h2 style="color: #4B0082;">Marketing Elements</h2>', unsafe_allow_html=True)
    st.markdown('<p>Select additional marketing elements to enhance your game promotion.</p>', unsafe_allow_html=True)
    
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
        st.error("Please enter and save both OpenAI and Replicate API keys in the sidebar.")
    elif not user_prompt.strip():
        st.error("Please enter a detailed game concept.")
    else:
        with st.spinner('Generating game plan...'):
            game_plan = generate_game_plan(user_prompt, st.session_state.customization)
        st.success('Game plan generated successfully!')
    
        # Display game plan results
        st.markdown('<h2 style="color: #4B0082;">Generated Game Plan</h2>', unsafe_allow_html=True)

        if 'game_concept' in game_plan and game_plan['game_concept']:
            st.subheader("Game Concept")
            st.write(game_plan['game_concept'])

        if 'world_concept' in game_plan and game_plan['world_concept']:
            st.subheader("World Concept")
            st.write(game_plan['world_concept'])

        if 'character_concepts' in game_plan and game_plan['character_concepts']:
            st.subheader("Character Concepts")
            st.write(game_plan['character_concepts'])

        if 'plot' in game_plan and game_plan['plot']:
            st.subheader("Plot")
            st.write(game_plan['plot'])

        if 'images' in game_plan and game_plan['images']:
            st.subheader("Generated Images")
            for img_name, img_url in game_plan['images'].items():
                if isinstance(img_url, str) and not img_url.startswith('Error'):
                    display_image(img_url, img_name)
                else:
                    st.write(f"{img_name}: {img_url}")

        if 'scripts' in game_plan and game_plan['scripts']:
            st.subheader("Generated Scripts")
            for script_name, script_code in game_plan['scripts'].items():
                with st.expander(f"View {script_name}"):
                    language = script_name.split('.')[-1]
                    st.code(script_code, language=language)

        if 'music' in game_plan and game_plan['music']:
            st.subheader("Generated Music")
            st.audio(game_plan['music'], format='audio/mp3')
        else:
            st.warning("No music was generated or an error occurred during music generation.")

        # Save results as ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add textual content
            for key in ['game_concept', 'world_concept', 'character_concepts', 'plot']:
                if key in game_plan and game_plan[key]:
                    zip_file.writestr(f"{key}.txt", game_plan[key])
            
            # Add images
            if 'images' in game_plan and game_plan['images']:
                for img_name, img_url in game_plan['images'].items():
                    if isinstance(img_url, str) and img_url.startswith('http'):
                        try:
                            img_response = requests.get(img_url)
                            img_response.raise_for_status()
                            img = Image.open(BytesIO(img_response.content))
                            img_buffer = BytesIO()
                            img.save(img_buffer, format='PNG')
                            zip_file.writestr(f"{img_name}.png", img_buffer.getvalue())
                        except:
                            pass  # Skip if there's an error
            
            # Add scripts
            if 'scripts' in game_plan and game_plan['scripts']:
                for script_name, script_code in game_plan['scripts'].items():
                    zip_file.writestr(script_name, script_code)
            
            # Add music
            if 'music' in game_plan and game_plan['music']:
                try:
                    music_response = requests.get(game_plan['music'])
                    music_response.raise_for_status()
                    zip_file.writestr("background_music.mp3", music_response.content)
                except:
                    pass  # Skip if there's an error
        
        st.download_button(
            label="Download Game Plan ZIP",
            data=zip_buffer.getvalue(),
            file_name="game_plan.zip",
            mime="application/zip",
            help="Download a ZIP file containing all generated assets and documents."
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Created by [Your Name](https://yourwebsite.com) | 
        [GitHub](https://github.com/yourusername) | 
        [Twitter](https://twitter.com/yourusername)</p>
    </div>
    """, unsafe_allow_html=True)
