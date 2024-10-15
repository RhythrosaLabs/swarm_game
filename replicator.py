import os
import datetime
import tkinter as tk
from tkinter import messagebox, filedialog
import replicate
from PIL import Image, ImageTk
import io
import requests
import threading

class LaunchScreen:
    def __init__(self, root):
        self.root = root
        self.root.title("API Key Input")
        self.root.geometry("300x150")

        self.api_key_label = tk.Label(root, text="Enter your Replicate API Key:")
        self.api_key_label.pack()

        self.api_key_entry = tk.Entry(root, width=30)
        self.api_key_entry.pack()

        self.submit_button = tk.Button(root, text="Submit", command=self.save_api_key)
        self.submit_button.pack()

    def save_api_key(self):
        api_key = self.api_key_entry.get()
        if not api_key:
            messagebox.showinfo("Error", "Please enter your API key")
        else:
            # Save API key to environment variable
            os.environ['REPLICATE_API_TOKEN'] = api_key
            self.root.destroy()  # Close the launch screen window

            # Launch the main application
            root = tk.Tk()
            app = App(root)
            root.mainloop()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Replicate Image Generator")
        self.root.geometry("500x550")

        self.model_var = tk.StringVar(root)
        self.model_var.set("lora_openjourney_v4")  # default value

        self.model_select = tk.OptionMenu(root, self.model_var, "lora_openjourney_v4", "material_stable_diffusion", "shap-e", "stable_diffusion_videos_openjourney", "openjourney", "animate-diff", "stable_diffusion_infinite_zoom", "future-diffusion", "riffusion", "deforum_stable_diffusion")
        self.model_select.pack()

        self.prompt_label = tk.Label(root, text="Enter Prompt(s) - separate multiple prompts by '|'")
        self.prompt_label.pack()

        self.prompt_entry = tk.Entry(root, width=50)
        self.prompt_entry.insert(0, "hyperrealistic, cinematic lighting, digital art, concept art, mdjrny-v4 style")
        self.prompt_entry.pack()

        self.generate_button = tk.Button(root, text="Generate", command=self.generate_image)
        self.generate_button.pack()

        self.download_button = tk.Button(root, text="Download", command=self.download_image, state=tk.DISABLED)
        self.download_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

    def generate_image(self):
        model = self.model_var.get()
        prompt = self.prompt_entry.get()
        if not prompt:
            messagebox.showinfo("Error", "Please enter a prompt")
            return

        # Run model in a separate thread to prevent freezing the GUI
        thread = threading.Thread(target=self.run_model, args=(model, prompt,))
        thread.start()

    def run_model(self, model, prompt):
        model_map = {
            "lora_openjourney_v4": "zhouzhengjun/lora_openjourney_v4:f8e5074f993f6852679bdac9f604590827f11698fdbfc3f68a1f0c3395b46db6",
            "material_stable_diffusion": "tommoore515/material_stable_diffusion:3b5c0242f8925a4ab6c79b4c51e9b4ce6374e9b07b5e8461d89e692fd0faa449",
            "shap-e": "cjwbw/shap-e:5957069d5c509126a73c7cb68abcddbb985aeefa4d318e7c63ec1352ce6da68c",
            "stable_diffusion_videos_openjourney": "wcarle/stable_diffusion_videos_openjourney:bd5fd4290fc2ab4b6931c90aee17581a62047470422737e035f34badb8af4132",
            "openjourney": "prompthero/openjourney:ad59ca21177f9e217b9075e7300cf6e14f7e5b4505b87b9689dbd866e9768969",
            "animate-diff": "lucataco/animate-diff:1531004ee4c98894ab11f8a4ce6206099e732c1da15121987a8eef54828f0663",
            "stable_diffusion_infinite_zoom": "arielreplicate/stable_diffusion_infinite_zoom:a2527c5074fc0cf9fa6015a40d75d080d1ddf7082fabe142f1ccd882c18fce61",
            "future-diffusion": "cjwbw/future-diffusion:b5c46a3b3f0db2a154d4be534ba7758caded970b748a2e26e6d02e9b3bd7da2a",
            "riffusion": "riffusion/riffusion:8cf61ea6c56afd61d8f5b9ffd14d7c216c0a93844ce2d82ac1c9ecc9c7f24e05",
            "deforum_stable_diffusion": "deforum/deforum_stable_diffusion:e22e77495f2fb83c34d5fae2ad8ab63c0a87b6b573b6208e1535b23b89ea66d6"
        }

        input_format = {
            "lora_openjourney_v4": {"prompt": f"{prompt}"},
            "material_stable_diffusion": {"prompt": f"{prompt}"},
            "shap-e": {"input": {"prompt": f"{prompt}"}},
            "stable_diffusion_videos_openjourney": {"prompts": f"{prompt}"},
            "openjourney": {"prompts": f"{prompt}"},
            "animate-diff": {"motion_module": "mm_sd_v14"},
            "stable_diffusion_infinite_zoom": {"prompt": f"{prompt}"},
            "future-diffusion": {"prompt": f"{prompt}"},
            "riffusion": {"prompt_a": f"{prompt}"},
            "deforum_stable_diffusion": {"max_frames": 100}
        }

        output = replicate.run(
            model_map[model],
            input=input_format[model],
            api_key=os.environ.get('REPLICATE_API_TOKEN', '')  # Retrieve API key from environment variable
        )

        if model in ["openjourney", "stable_diffusion_videos_openjourney"]:
            for item in output:
                self.display_image(item)
        else:
            self.display_image(output[0])
        
        # Enable the download button after generating the image
        self.download_button.config(state=tk.NORMAL)

    def display_image(self, image_url):
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def download_image(self):
        # Retrieve the image URL from the label's image object
        image_url = self.image_label.image.cget("file")
        if not image_url:
            messagebox.showinfo("Error", "No image to download")
            return

        # Prompt user to select a location to save the image
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            # Download the image from the URL and save it to the selected location
            response = requests.get(image_url)
            with open(file_path, "wb") as f:
                f.write(response.content)
            messagebox.showinfo("Success", "Image downloaded successfully!")


# Create the launch screen window
root = tk.Tk()
launch_screen = LaunchScreen(root)
root.mainloop()
