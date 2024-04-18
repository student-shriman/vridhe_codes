import google.generativeai as genai
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama
import google.generativeai as genai
import google.generativeai as geminiai
from dotenv import find_dotenv, load_dotenv






# Set up the model
generation_config = {"temperature": 0.5, "top_p": 1, "top_k": 30, "max_output_tokens": 4096}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

# Models ..
model_1= genai.GenerativeModel(model_name="gemini-1.0-pro-001", generation_config=generation_config, safety_settings=safety_settings)
model_2 = geminiai.GenerativeModel(model_name="gemini-1.0-pro-001", generation_config=generation_config, safety_settings=safety_settings)


##  Gemini stable models .. ..
def gen_with_gemini(prompt, model):
  convo = model.start_chat()
  convo.send_message(prompt)
  return(convo.last.text)




# Model setup ..
def gen_content_with_openai(prompt):
    model_name = "gpt-4-1106-preview"
    temperature = 0.2
    model = OpenAI(model_name=model_name, temperature=temperature)
    output = model(prompt)
    return output


##  ollama model ..
def get_ans(prompt):
    llm = Ollama(model='zephyr')
    out = llm.invoke(prompt)
    return out