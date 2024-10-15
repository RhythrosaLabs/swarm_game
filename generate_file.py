def generate_file_with_gpt(prompt):
    api_keys = get_api_keys()
    openai_api_key = api_keys.get("openai")
    replicate_api_key = api_keys.get("replicate")

    if not openai_api_key:
        st.error("OpenAI API key is not set. Please add it in the sidebar.")
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
        if not replicate_api_key:
            st.error("Replicate API key is not set. Please add it in the sidebar.")
            return None, None
        specific_prompt = prompt.replace("/video ", "").strip()
        return generate_video_with_replicate(specific_prompt, replicate_api_key)
    
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
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
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
    elif prompt.startswith("/txt "):
        file_extension = ".txt"
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
        response = requests.post(DALLE_API_URL, headers=headers, json=data)
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

def generate_video_with_replicate(prompt, api_key):
    input_data = {
        "prompt": prompt,
        "num_frames": 30,  # 1 second at 30 FPS
        "frame_rate": 30
    }

    try:
        replicate_client = replicate.Client(api_token=api_key)
        output_url = replicate_client.run(
            "deforum/deforum_stable_diffusion",
            input=input_data
        )
        video_data = requests.get(output_url).content
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error generating video: {str(e)}")
        return None, None
    except requests.RequestException as e:
        st.error(f"Error downloading video: {e}")
        return None, None

def generate_speech_with_gtts(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None, None

    file_name = text.replace(" ", "_") + ".mp3"
    file_data = audio_data

    return file_name, file_data

    file_name = prompt.replace(" ", "_") + "_1second.mp4"
    file_data = video_data

    return file_name, file_data
