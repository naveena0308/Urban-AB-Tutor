import os 
import google.generativeai as genai 
 
api_key = os.environ.get('GEMINI_API_KEY') 
if not api_key: 
    raise ValueError('GEMINI_API_KEY environment variable not set!') 
genai.configure(api_key=api_key)

import dotenv
from dotenv import load_dotenv 
load_dotenv()  # reads .env file into environment variables 
genai.configure(api_key=os.environ.get('GEMINI_API_KEY')) 