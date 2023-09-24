import openai
import gradio as gr
import os
import tiktoken
import numpy as np
from collections import defaultdict
import json
from datetime import datetime
import glob
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

md_styles = {
    'h1': '# ',
    'h2': '## ',
    'pre': '',
    'blockquote': '> ',
    'p': '',
}

# Set OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

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

# Define the paths to your input JSON data and output folders
input_json_folder = '/content/drive/MyDrive/Colab Notebooks/30'
output_folder = '/content/drive/MyDrive/Colab Notebooks/30/output_folder'

# Check if the output folder exists, and if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load and process JSON data
for json_file in os.listdir(input_json_folder):
    if json_file.endswith('.json'):
        # Load JSON data
        with open(os.path.join(input_json_folder, json_file), 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Extract relevant data from JSON
        article_id = json_data[0].get('id', '')
        article_text = json_data[0].get('text', '')
        article_title = json_data[0].get('title', '')

        # Prepare data for output files
        date_obj = datetime.now()
        formatted_date = date_obj.strftime('%Y-%m-%d')

        # Output file paths
        output_file_txt = os.path.join(output_folder, f"{article_id}_{formatted_date}.txt")
        output_file_md = os.path.join(output_folder, f"{article_id}_{formatted_date}.md")

        print(f"Processing article: {article_title}")

        # Create text and Markdown files
        with open(output_file_txt, 'w', encoding='utf-8') as txt_file, open(output_file_md, 'w', encoding='utf-8') as md_file:
            # Prepare metadata
            date_text = formatted_date
            title_text = article_title

            # Split the article into sections (e.g., paragraphs)
            article_sections = article_text.split('\n\n')

            for i, section in enumerate(article_sections):
                section = section.strip()
                if section:
                    section_lines = section.split('\n')
                    style = 'p' if len(section_lines) == 1 else 'pre'  # Detect code blocks

                    for line in section_lines:
                        filtered_line = line.strip()
                        if filtered_line:
                            md_file.write(md_styles.get(style, '') + filtered_line + '\n')
                            txt_file.write(filtered_line + '\n')

                    if style != 'blockquote':
                        md_file.write('\n')  # Add blank line between sections in Markdown

        print(f"Processed article saved as: {output_file_txt} and {output_file_md}\n")

# Specify the paths to the output folders for text and Markdown files
txt_folder = '/content/drive/MyDrive/Colab Notebooks/30/output_folder'  # Update with the correct path
md_folder = '/content/drive/MyDrive/Colab Notebooks/30/output_folder'   # Update with the correct path

# Use glob to get a list of text and Markdown files
txt_files = glob.glob(f'{txt_folder}/*.txt')
print(len(txt_files))
md_files = glob.glob(f'{md_folder}/*.md')
print(len(md_files))

# Prepare data for fine-tuning
train_file = open("training.jsonl", "w")
test_file = open("testing.jsonl", "w")

system_prompt = "You render plain text into markdown format."

for i, (txt_file, md_file) in enumerate(zip(txt_files, md_files)):
    # Read the content from the txt file
    with open(txt_file, 'r') as f:
        content_string = f.read()

    # Read the reference markdown content
    with open(md_file, 'r') as f:
        reference_md = f.read()

    item = {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_string},
        {"role": "assistant", "content": reference_md}
      ]}

    item_str = json.dumps(item)
    if i%10==0:
      print(i, item_str)
    if i < len(txt_files)-12:
        train_file.write(item_str + "\n")
    else:
        test_file.write(item_str + "\n")

train_file.close()
test_file.close()

# Perform fine-tuning with OpenAI API
openai.File.create(file=open("training.jsonl", "rb"), purpose='fine-tune')
openai.File.create(file=open("testing.jsonl", "rb"), purpose='fine-tune')

results = openai.FineTuningJob.create(
  training_file="file-l6dFcHi7JKXVq6R8PtWLBTrV",
  validation_file="file-U7cqDk0G2KCnmV4BD6ITzRGo",
  suffix="fine_tuned",
  model="gpt-3.5-turbo"
)

print(results)

# Define a function to construct the AI index
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

# Define the chatbot function using the AI index
def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

# Create a Gradio interface for the chatbot
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
    outputs="text",
    title="Custom-trained AI Chatbot"
)

# Launch the Gradio interface
iface.launch(share=True)
