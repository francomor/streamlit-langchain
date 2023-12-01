import streamlit as st
import streamlit_authenticator as stauth
import yaml
from PIL import Image
from yaml.loader import SafeLoader

with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

st.set_page_config(
    page_title="AI summarizer",
    page_icon="🧊",
)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    image = Image.open("home_image.jpg")

    st.write(f"# Welcome *{name}* to AI summarizer!")

    st.image(image, use_column_width="always")

    authenticator.logout("Logout", "main")
    st.markdown(
        """
        ### What is this?
        This is a tool to help you summarize large amounts of text. It uses the
        [OpenAI API](https://openai.com/blog/openai-api/) to generate summaries
        based on your input.
        """
    )

elif authentication_status is None:
    st.warning("Please enter your username and password")
elif not authentication_status:
    st.error("Username/password is incorrect")
