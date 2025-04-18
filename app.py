from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.retrieve_10k import pipeline
from src.cross_checker import multimodal_agent

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.secret_key = 'your_secret_key_here'  # Needed for session handling

# Load GPT-2
device = torch.device("cpu")
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/')
def landing_page():
    session.clear()
    return render_template('landing.html')

@app.route('/select_company', methods=['POST'])
def select_company():
    company = request.form.get('company')
    year = request.form.get('year')
    
    # Debugging output
    print(f"Selected Company: {company}, Selected Year: {year}")
    
    session['company'] = company
    session['year'] = year
    return redirect(url_for('chat'))

@app.route('/chat')
def chat():
    if 'company' not in session or 'year' not in session:
        return redirect(url_for('landing_page'))
    
    company = session.get('company', '')
    year = session.get('year', '')
    
    # Debugging output
    print(f"Company: {company}, Year: {year}")
    
    return render_template('index.html', company=company, year=year)

# @app.route('/get_response', methods=['POST'])
# def get_response():
#     user_input = request.json['user_input']
#     prompt = f"Human: {user_input}\nBot:"
#     inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
#     output = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)
#     bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
#     bot_response = bot_response.split('Bot:')[-1].strip()
#     return jsonify({'response': bot_response})

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    company = session.get('company', '')
    year = session.get('year', '')

    # Ensure company and year are selected
    if not company or not year:
        return jsonify({'response': 'Company and year not set. Please go back and select again.'})

    # Call multimodal_agent with the user input and selected company/year
    try:
        bot_response = multimodal_agent(user_input, company, year)
    except Exception as e:
        bot_response = f"An error occurred: {str(e)}"

    return jsonify({'response': bot_response})

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()  # Clear the entire session
    return redirect(url_for('landing_page'))

if __name__ == '__main__':
    app.run(debug=True)
