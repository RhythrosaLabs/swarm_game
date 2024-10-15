import streamlit as st
from helpers import (
    generate_content,
    generate_budget_spreadsheet,
    generate_social_media_schedule,
    generate_images,
    create_master_document,
    create_zip,
    enhance_content,
    add_to_chat_knowledge_base,
    add_file_to_global_storage,
    create_gif,
    generate_audio_logo,
    generate_video_logo,
    animate_image_to_video,
    fetch_generated_video
)

def generate_tab():
    st.title("Generate Marketing Campaign")

    api_key = st.session_state["api_keys"]["openai"]
    replicate_api_key = st.session_state["api_keys"].get("replicate", None)

    if not api_key:
        st.warning("Please provide a valid OpenAI API Key.")
        return

    prompt = st.text_area("Prompt", "Describe your product or campaign...")

    budget = st.text_input("Budget", "1000")

    with st.expander("Advanced Options"):
        st.subheader("Social Media Platforms")
        platforms = {
            "facebook": st.checkbox("Facebook"),
            "twitter": st.checkbox("Twitter", value=True),  # Auto-check Twitter
            "instagram": st.checkbox("Instagram"),
            "linkedin": st.checkbox("LinkedIn")
        }

        st.subheader("Image Tools")
        bypass_images = st.checkbox("Bypass image generation", value=True)
        
        image_size_options = {
            "Wide": "1792x1024",
            "Tall": "1024x1792",
            "Square": "1024x1024"
        }

        if not bypass_images:
            if "image_prompts" not in st.session_state:
                st.session_state["image_prompts"] = [""]
                st.session_state["image_sizes"] = ["Square"]

            for i in range(len(st.session_state["image_prompts"])):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.session_state["image_prompts"][i] = st.text_input(f"Image {i+1} Prompt:", st.session_state["image_prompts"][i])
                with cols[1]:
                    st.session_state["image_sizes"][i] = st.selectbox(f"Size {i+1}:", options=list(image_size_options.keys()), index=["Wide", "Tall", "Square"].index(st.session_state["image_sizes"][i]))
                with cols[2]:
                    if st.button("➖", key=f"remove_image_{i}"):
                        st.session_state["image_prompts"].pop(i)
                        st.session_state["image_sizes"].pop(i)
                        st.experimental_rerun()

            if len(st.session_state["image_prompts"]) < 5:
                if st.button("➕ Add Image"):
                    st.session_state["image_prompts"].append("")
                    st.session_state["image_sizes"].append("Square")

            hd_images = st.checkbox("Generate HD images")

            create_gif_checkbox = st.checkbox("Create GIF from images", value=False)
            filter_type = st.selectbox("Select GIF Filter:", ["None", "Sepia", "Greyscale", "Negative", "Solarize", "Posterize"])
            filter_type = filter_type.lower() if filter_type != "None" else None

        st.subheader("Other Settings")
        add_audio_logo = st.checkbox("Add audio logo")
        add_video_logo = st.checkbox("Add video logo")
        # auto_pdf = st.checkbox("Auto PDF")  # Commenting out auto PDF option

    if st.button("Generate Marketing Campaign"):
        with st.spinner("Generating..."):
            campaign_plan = {}

            # Generate and analyze campaign concept
            st.info("Generating campaign concept...")
            campaign_concept = generate_content("Generate campaign concept", prompt, budget, platforms, api_key)
            campaign_plan['campaign_concept'] = campaign_concept
            add_file_to_global_storage("campaign_concept.txt", campaign_concept)

            st.info("Analyzing campaign concept...")
            analyzed_concept = enhance_content(campaign_concept, "Campaign Concept")
            add_to_chat_knowledge_base("Campaign Concept", analyzed_concept)
            add_file_to_global_storage("analyzed_campaign_concept.txt", analyzed_concept)

            # Generate and analyze marketing plan
            st.info("Generating marketing plan...")
            marketing_plan = generate_content("Generate marketing plan", prompt, budget, platforms, api_key)
            campaign_plan['marketing_plan'] = marketing_plan
            add_file_to_global_storage("marketing_plan.txt", marketing_plan)

            st.info("Analyzing marketing plan...")
            analyzed_plan = enhance_content(marketing_plan, "Marketing Plan")
            add_to_chat_knowledge_base("Marketing Plan", analyzed_plan)
            add_file_to_global_storage("analyzed_marketing_plan.txt", analyzed_plan)

            # Generate and analyze budget spreadsheet
            st.info("Generating budget spreadsheet...")
            budget_spreadsheet = generate_budget_spreadsheet(budget)
            campaign_plan['budget_spreadsheet'] = budget_spreadsheet
            add_file_to_global_storage("budget_spreadsheet.xlsx", budget_spreadsheet)

            st.info("Analyzing budget spreadsheet...")
            analyzed_budget = enhance_content(budget_spreadsheet, "Budget Spreadsheet")
            add_to_chat_knowledge_base("Budget Spreadsheet", analyzed_budget)
            add_file_to_global_storage("analyzed_budget_spreadsheet.txt", analyzed_budget)

            # Generate and analyze social media schedule
            st.info("Generating social media schedule...")
            social_media_schedule = generate_social_media_schedule(campaign_concept, platforms)
            campaign_plan['social_media_schedule'] = social_media_schedule
            add_file_to_global_storage("social_media_schedule.xlsx", social_media_schedule)

            st.info("Analyzing social media schedule...")
            analyzed_schedule = enhance_content(social_media_schedule, "Social Media Schedule")
            add_to_chat_knowledge_base("Social Media Schedule", analyzed_schedule)
            add_file_to_global_storage("analyzed_social_media_schedule.txt", analyzed_schedule)

            # Generate images if not bypassed
            if not bypass_images:
                st.info("Generating images...")
                custom_prompts = st.session_state["image_prompts"]
                image_sizes = [image_size_options[size] for size in st.session_state["image_sizes"]]
                images = generate_images(api_key, custom_prompts, image_sizes, hd_images)
                campaign_plan['images'] = images

                for image_key, image_data in images.items():
                    st.info(f"Analyzing {image_key}...")
                    analyzed_image = enhance_content(image_data, image_key)
                    add_to_chat_knowledge_base(image_key, analyzed_image)
                    add_file_to_global_storage(image_key, image_data)

                if create_gif_checkbox and images:
                    st.info("Creating GIF...")
                    gif_data = create_gif(list(images.values()), filter_type)
                    campaign_plan['images']['instagram_video.gif'] = gif_data.getvalue()
                    add_file_to_global_storage("instagram_video.gif", gif_data.getvalue())

            # Generate audio logo if selected and replicate API key is provided
            if add_audio_logo:
                if replicate_api_key:
                    st.info("Generating audio logo...")
                    audio_prompt = f"Generate an audio logo for the following campaign concept: {campaign_concept}"
                    file_name, audio_data = generate_audio_logo(audio_prompt, replicate_api_key)
                    if audio_data:
                        campaign_plan['audio_logo'] = audio_data
                        add_file_to_global_storage(file_name, audio_data)
                else:
                    st.warning("Replicate API Key is required to generate an audio logo.")

            # Generate video logo if selected
            if add_video_logo:
                st.info("Generating video logo...")
                video_prompt = f"Generate a video logo for the following campaign concept: {campaign_concept}"
                file_name, video_logo_data = generate_video_logo(video_prompt, api_key)
                if video_logo_data:
                    st.info("Animating video logo...")
                    generation_id = animate_image_to_video(video_logo_data, video_prompt)
                    if generation_id:
                        video_data = fetch_generated_video(generation_id)
                        if video_data:
                            campaign_plan['video_logo'] = video_data
                            add_file_to_global_storage("video_logo.mp4", video_data)

            # Generate and analyze resources and tips
            st.info("Generating resources and tips...")
            resources_tips = generate_content("Generate resources and tips", prompt, budget, platforms, api_key)
            campaign_plan['resources_tips'] = resources_tips
            add_file_to_global_storage("resources_tips.txt", resources_tips)

            st.info("Analyzing resources and tips...")
            analyzed_resources = enhance_content(resources_tips, "Resources and Tips")
            add_to_chat_knowledge_base("Resources and Tips", analyzed_resources)
            add_file_to_global_storage("analyzed_resources_tips.txt", analyzed_resources)

            # Generate and analyze recap
            st.info("Generating recap...")
            recap = generate_content("Generate recap", prompt, budget, platforms, api_key)
            campaign_plan['recap'] = recap
            add_file_to_global_storage("recap.txt", recap)

            st.info("Analyzing recap...")
            analyzed_recap = enhance_content(recap, "Recap")
            add_to_chat_knowledge_base("Recap", analyzed_recap)
            add_file_to_global_storage("analyzed_recap.txt", analyzed_recap)

            st.info("Generating master document...")
            master_document = create_master_document(campaign_plan)
            campaign_plan['master_document'] = master_document
            add_file_to_global_storage("master_document.txt", master_document)

            # if auto_pdf:
            #     st.info("Compiling to PDF...")
            #     pdf_data = compile_to_pdf(campaign_plan)
            #     add_file_to_global_storage("campaign_summary.pdf", pdf_data)
            #     st.session_state.pdf_data = pdf_data

            st.info("Packaging into ZIP...")
            zip_data = create_zip(campaign_plan)

            st.session_state.campaign_plan = campaign_plan
            st.success("Marketing Campaign Generated")
            st.download_button(label="Download ZIP", data=zip_data.getvalue(), file_name="marketing_campaign.zip", key="download_campaign_zip")

if __name__ == "__main__":
    generate_tab()
