import os
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from config import LLAVA_MODEL_NAME, OLLAMA_HOST

def process_image(image: Image.Image, prompt: str) -> str:
    try:
        if image is None:
            return "Please upload an image first for vision analysis."

        # Initialize local LLaVA model via Ollama (No API key needed)
        llm = ChatOllama(model=LLAVA_MODEL_NAME, base_url=OLLAMA_HOST, temperature=0.2)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt if prompt else "Analyze this image and extract all relevant numerical data, text, or chart details."},
                {"type": "image_url", "image_url": image}
            ]
        )

        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return f"Error in Vision Agent: {str(e)}"