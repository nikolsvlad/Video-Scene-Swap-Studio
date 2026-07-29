import os
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = Client("innova-ai/video-background-removal", token=HF_TOKEN)
client.view_api()