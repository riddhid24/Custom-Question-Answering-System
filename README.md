# AI Chatbot with Fine-tuning and Custom Indexing

This repository contains code for building an AI chatbot powered by OpenAI's GPT-3.5 Turbo model. The chatbot is fine-tuned on custom data and includes a custom indexing mechanism for faster responses.

## Prerequisites

Before using this code, you'll need the following:

1. OpenAI API Key: You must have an OpenAI API key to use the GPT-3.5 Turbo model. You can sign up for one on the [OpenAI website](https://beta.openai.com/signup/).

2. Python Environment: Make sure you have a Python environment set up. You can use a virtual environment to manage dependencies.

## Installation

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/your-username/ai-chatbot.git
   ```

2. Navigate to the project directory:

   ```
   cd ai-chatbot
   ```

3. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Fine-tuning the Chatbot

Before running the chatbot, you need to fine-tune it on your custom data. Follow these steps:

#### a. Prepare Your Data

- Place your custom JSON data files in the `input_json_folder`. Each JSON file should contain articles with titles and text content.

#### b. Fine-tuning Configuration

- Open the `chatbot_with_finetuning.py` file and configure the fine-tuning parameters, such as the number of epochs and batch size.

#### c. Run Fine-tuning

- Execute the fine-tuning script:

  ```
  python chatbot_with_finetuning.py
  ```

- This script will create training and validation sets, upload them to OpenAI, and initiate the fine-tuning process.

### 2. Indexing for Faster Responses

The chatbot uses a custom indexing mechanism for faster responses. Here's how to set it up:

#### a. Index Construction

- Open the `indexing.py` file and configure the indexing parameters, such as chunk size and maximum tokens.

- Run the indexing script:

  ```
  python indexing.py
  ```

- This will create an AI index that can be used for faster responses.

### 3. Chatbot with Gradio Interface

The chatbot is accessible through a Gradio interface. To run the chatbot with the interface, follow these steps:

- Open the `chatbot.py` file.

- Replace `"YOUR_OPENAI_API_KEY"` with your actual OpenAI API key.

- Run the Gradio interface:

  ```
  python chatbot.py
  ```

- Access the chatbot in your web browser and interact with it.

## Additional Information

- You can use normal chatbot with `chatbot.py`

- The fine-tuned model and indexing data are saved locally for future use.

- Ensure that your custom data and fine-tuned models comply with OpenAI's usage policies and guidelines.


## Acknowledgments

- This project is powered by OpenAI's GPT-3.5 Turbo model.

- Gradio is used to create the user-friendly interface for the chatbot.

- Special thanks to the open-source community for their valuable contributions.

## Questions and Support

If you have any questions or need support, feel free to contact [your email] or open an issue in the repository.

Enjoy using your custom AI chatbot!