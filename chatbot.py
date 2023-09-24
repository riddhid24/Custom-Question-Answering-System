# Import necessary libraries
import openai
import gradio as gr
import os
import tiktoken
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
from collections import defaultdict
import glob

# Set OpenAI API key
openai.api_key = "<API KEY>"

# Define a system prompt for the chatbot
system_prompt = "You render plain text into markdown format."

# Define a function for the chatbot
def chatbot(input_text):
    # Initialize messages with a system message
    messages = [{"role": "system", "content": system_prompt}]

    if input_text:
        # Add the user's message to the messages
        messages.append({"role": "user", "content": input_text})

        # Use OpenAI's Chat API to get a response
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Get the assistant's reply
        reply = chat.choices[0].message.content

        # Append the assistant's reply to messages
        messages.append({"role": "assistant", "content": reply})

        return reply

# Define the input and output components for Gradio UI
inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

# Create a Gradio interface for the chatbot
gr.Interface(
    fn=chatbot,
    inputs=inputs,
    outputs=outputs,
    title="AI Chatbot",
    description="Ask anything you want",
    theme="compact"
).launch(share=True)

