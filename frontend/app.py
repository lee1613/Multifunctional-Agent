from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
device = torch.device("cpu")  # You can change to "cuda" if you have a GPU
model_name = "gpt2"  # GPT-2 small version (124M parameters)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']

    # Prepend context to the user input to guide the model
    prompt = f"Human: {user_input}\nBot:"

    # Tokenize the user input and generate a response
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    output = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)

    # Decode the generated output
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the "Human:" and "Bot:" labels from the response, so it only returns the bot's reply
    bot_response = bot_response.split('Bot:')[-1].strip()

    # Return the bot response as JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
