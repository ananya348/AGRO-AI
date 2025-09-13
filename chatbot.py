# Krishi Sakhi - AI Farming Assistant (Python Terminal Version)
#
# Description:
# This script implements a conversational AI chatbot for farmers. It uses multiple PDF documents
# as a knowledge base, understands both English and Malayalam queries through text or voice,
# and responds in the same language, also via text and voice.
#
# Author: Gemini
# Date: September 11, 2025

# --- Installation ---
# Before running, you need to install the required Python libraries.
# Open your terminal or command prompt and run the following command:
# pip install google-generativeai python-dotenv speechrecognition pyaudio pdfplumber gtts playsound langdetect

import os
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import pdfplumber
from gtts import gTTS
from playsound import playsound
from langdetect import detect, LangDetectException

# --- Configuration ---
# 1. Create a file named ".env" in the same directory as this script.
# 2. Inside the .env file, add your Google API Key like this:
#    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
# 3. This script will automatically load the key.

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found. Please create a .env file and add your API key.")
    exit()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- Core Functions ---

def extract_text_from_pdfs(pdf_paths):
    """Extracts text content from a list of PDF files."""
    print("Extracting text from PDF documents...")
    full_text = ""
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
            print(f"  - Successfully processed {os.path.basename(path)}")
        except Exception as e:
            print(f"  - Could not read {os.path.basename(path)}: {e}")
    print("Extraction complete.\n")
    return full_text

def listen_for_input():
    """Captures audio from the microphone and converts it to text, prioritizing Malayalam."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening... (Speak now)")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        # Prioritize Malayalam recognition
        print("Recognizing (as Malayalam)...")
        query = r.recognize_google(audio, language='ml-IN')
        print(f"Heard (ml): {query}")
        return query, 'ml' # Return the query and the detected language 'ml'
    except sr.UnknownValueError:
        # If Malayalam fails, fall back to English
        try:
            print("Recognizing (as English)...")
            query = r.recognize_google(audio, language='en-IN')
            print(f"Heard (en): {query}")
            return query, 'en' # Return the query and the detected language 'en'
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None, None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None, None
    except Exception as e:
        print(f"An unknown error occurred during speech recognition: {e}")
        return None, None


def speak_response(text, lang='en'):
    """Converts text to speech and plays it using playsound."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_file = "response.mp3"
        tts.save(audio_file)
        print(f"Krishi Sakhi ({lang}): {text}")
        playsound(audio_file)

    except Exception as e:
        print(f"\nError in text-to-speech or playback: {e}")
        # Fallback to just printing if TTS fails
        print(f"Krishi Sakhi ({lang}): {text}")


def get_gemini_response(user_query, context):
    """Gets a response from the Gemini API based on the query and context."""
    system_prompt = """You are 'Agro AI', a friendly and knowledgeable AI farming assistant for farmers in Kerala, India. Your purpose is to answer farming-related questions.

    Follow these rules carefully:
    1.  Analyze the user's query to determine if it is in English or Malayalam. Your final response MUST be in the same language.
    2.  Prioritize using the information from the 'CONTEXT FROM DOCUMENTS' section to answer the question, as this is specific to the user.
    3.  If the documents do not contain the answer, you should use your general knowledge to provide the most helpful and accurate response possible.
    4.  When using general knowledge, you can optionally state that the specific information wasn't in the uploaded documents.
    5.  Keep your answers clear, concise, and easy for a farmer to understand.
    """

    full_prompt = f"CONTEXT FROM DOCUMENTS:\n---\n{context}\n---\n\nFARMER'S QUERY:\n{user_query}"

    # Combine the system prompt and the user-facing prompt into a single string
    # This is a more compatible way to pass system instructions.
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"

    try:
        print("Thinking...")
        response = model.generate_content(
            combined_prompt, # Pass the combined prompt
            generation_config=genai.types.GenerationConfig(
                temperature=0.2 # Lower temperature for more factual, less creative answers
            )
            # The unsupported 'system_instruction' argument has been removed.
        )
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "Sorry, I am having trouble connecting to my brain right now. Please try again later."


# --- Main Application Logic ---

def main():
    """The main function to run the chatbot."""
    print("="*50)
    print("      🌱 Welcome to Agro AI! 🌱")
    print("="*50)

    # 1. Get PDF files from user
    pdf_files = []
    print("Please provide the paths to your agricultural PDF files.")
    print("Enter the full path for each file and press Enter. Type 'done' when you are finished.")
    while True:
        path = input(f"PDF File {len(pdf_files) + 1}: ").strip()
        if path.lower() == 'done':
            if not pdf_files:
                print("No PDF files were provided. The chatbot needs at least one to work.")
                continue
            break
        if os.path.exists(path) and path.lower().endswith('.pdf'):
            pdf_files.append(path)
        else:
            print("Invalid file path or not a PDF. Please try again.")

    # 2. Extract context from PDFs
    knowledge_context = extract_text_from_pdfs(pdf_files)
    if not knowledge_context.strip():
        print("Could not extract any text from the provided PDFs. Exiting.")
        return

    # 3. Start conversation loop
    print("You can now start chatting with Krishi Sakhi.")
    print("Type your message or say 'speak' to use your voice.")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input == 'exit':
            break
            
        query = ""
        lang = "en" # Default language

        if user_input == 'speak':
            query, lang = listen_for_input()
            if not query:
                continue
        else:
            query = user_input
            # Detect language only for typed input
            try:
                # We are primarily concerned with malayalam vs others (english)
                detected_lang = detect(query)
                if detected_lang == 'ml':
                    lang = 'ml'
            except LangDetectException:
                lang = 'en' # Keep default if detection fails

        # Get and process the response
        bot_response = get_gemini_response(query, knowledge_context)
        speak_response(bot_response, lang)

    print("\nThank you for using Agro AI. Have a great day! 🌱")


if __name__ == "__main__":
    main()

