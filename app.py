import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_FILENAME = "pop2016.pdf"

if not API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found. Please create a .env file and add your API key.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Flask App Setup ---
app = Flask(__name__)

# ✅ Fix: Only allow the specific frontend origin (adjust this if your frontend URL changes)
CORS(app, resources={
    r"/chat": {
        "origins": "https://scaling-spork-pjrq7qvr6x9v2rgg5-5500.app.github.dev/",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# --- State Management ---
pdf_context = ""

# --- Core Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a single PDF file."""
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def load_initial_context():
    """Loads the PDF context when the server starts."""
    global pdf_context
    if not os.path.exists(PDF_FILENAME):
        print(f"FATAL ERROR: The file '{PDF_FILENAME}' was not found in the same directory.")
        pdf_context = f"Error: Could not find the knowledge document '{PDF_FILENAME}'."
    else:
        print(f"Loading knowledge base from '{PDF_FILENAME}'...")
        context = extract_text_from_pdf(PDF_FILENAME)
        if context:
            pdf_context = context
            print("Knowledge base loaded successfully.")
        else:
            pdf_context = "Error: The provided PDF could not be read or is empty."
            print(f"Error processing {PDF_FILENAME}. The bot may not have context.")

def get_gemini_response(user_query, context):
    """Gets a response from the Gemini API based on the query and context."""
    system_prompt = """You are 'Krishi Sakhi', a friendly and knowledgeable AI farming assistant for farmers in Kerala, India.
- Your purpose is to answer farming-related questions.
- Analyze the user's query to determine if it is in English or Malayalam. Your final response MUST be in the same language.
- After your main answer, add a language tag on a new line, like this: [lang:ml] for Malayalam or [lang:en] for English. This tag is for the application and should not be spoken.
- Prioritize using the information from the 'CONTEXT FROM DOCUMENTS' section to answer.
- If the documents don't have the answer, use your general knowledge.
- Keep your answers clear, concise, and easy for a farmer to understand."""

    full_prompt = f"CONTEXT FROM DOCUMENTS:\n---\n{context}\n---\n\nFARMER'S QUERY:\n{user_query}"

    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3)
        )
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "Sorry, I am having trouble connecting to my brain right now. Please try again later. [lang:en]"

# --- API Endpoint ---

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat queries."""
    global pdf_context
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    bot_response_text = get_gemini_response(user_query, pdf_context)

    # Parse the response to separate text and language
    response_text = bot_response_text
    lang = 'en'
    lang_match = bot_response_text.strip().split('\n')[-1]
    if '[lang:ml]' in lang_match:
        lang = 'ml'
        response_text = bot_response_text.replace('[lang:ml]', '').strip()
    elif '[lang:en]' in lang_match:
        lang = 'en'
        response_text = bot_response_text.replace('[lang:en]', '').strip()

    return jsonify({"response": response_text, "lang": lang})

# --- App Entrypoint ---

if __name__ == '__main__':
    # ✅ Fix: Only run PDF loading in the reloader process (avoid duplicate loads)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("FLASK_ENV") != "development":
        load_initial_context()

    app.run(host='0.0.0.0', port=5000, debug=True)
