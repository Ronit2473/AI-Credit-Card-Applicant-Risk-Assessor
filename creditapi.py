from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from fastapi.responses import JSONResponse, FileResponse

load_dotenv()
api_key = os.getenv('NVIDIA_API_KEY')

# --- NEW: Import LangChain Modules ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the LangChain LLM (Pointing to NVIDIA)
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key, # Put your API key here
    model="meta/llama-3.1-8b-instruct",
    temperature=0.5,
    max_tokens=100
)

# 2. Create a LangChain Prompt Template
prompt_template = PromptTemplate.from_template(
    """You are an expert AI credit risk assistant. An ensemble machine learning model just evaluated a credit card applicant and flagged them as: {risk_label}.
    
    Here is the applicant's exact profile:
    - Age: {age}
    - Total Income: ${total_income}
    - Years Employed: {years_employed}
    - Income Type: {income_type}
    - Account Length: {account_length} years
    - Family Size: {num_family}

    Briefly explain to a human loan officer why this specific profile might be considered {risk_label}. 
    Keep it professional, insightful, and strictly limit it to 2-3 short sentences. Do not use markdown."""
)

# 3. Build the Chain using LCEL (LangChain Expression Language)
# This pipes the Prompt -> into the LLM -> and Parses the output to a string
explain_chain = prompt_template | llm | StrOutputParser()
# -------------------------------------

# Load the ML model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
app = FastAPI()

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

income_mapping = {
    'Commercial associate': 0, 'Pensioner': 1, 'State servant': 2, 'Student': 3, 'Working': 4
}

class Userinput(BaseModel):
    Gender: Annotated[int, Field(..., description="0 for Female, 1 for Male")]
    Total_income: Annotated[float, Field(..., ge=0, description="Total Income")]
    Income_type: Annotated[Literal['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'], Field(..., description="Income_Type")]
    Num_family: Annotated[int, Field(..., ge=1, description="Number of family members")]
    Account_length: Annotated[int, Field(..., ge=0, le=50, description="Account age in years")]
    Age: Annotated[int, Field(..., ge=0, le=100, description="Age of the user")]
    Years_employed: Annotated[float, Field(..., ge=0, le=100, description="Years Employed")]
    Is_Working: Annotated[int, Field(..., description="1 if working, 0 if not")]

# --- NEW: Cleaner Explainer Function using LangChain ---
def get_llm_explanation(data: Userinput, risk_label: str):
    try:
        # We simply pass a dictionary of our variables into the chain!
        response = explain_chain.invoke({
            "risk_label": risk_label,
            "age": data.Age,
            "total_income": data.Total_income,
            "years_employed": data.Years_employed,
            "income_type": data.Income_type,
            "account_length": data.Account_length,
            "num_family": data.Num_family
        })
        return response
    except Exception as e:
        print(f"LangChain Error: {e}")
        return "The ML model generated a risk score, but the LLM assistant is currently unavailable to provide a summary."

@app.post('/predict')
def predict_risk(data: Userinput):
    mapped_income = income_mapping.get(data.Income_type, 4)

    input_df = pd.DataFrame([{
        'Gender': data.Gender,
        'Total_income': data.Total_income,
        'Income_type': mapped_income,
        'Num_family': data.Num_family,
        'Account_length': data.Account_length,
        'Age': data.Age,
        'Years_employed': data.Years_employed,
        'Is_Working': data.Is_Working
    }])
    
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    risk_label = 'High Risk User' if prediction == 1 else 'Low Risk User'
    
    # Generate the explanation using the LangChain pipeline!
    bot_explanation = get_llm_explanation(data, risk_label)
    
    return JSONResponse(status_code=200, content={
        'risk': risk_label, 
        'explanation': bot_explanation
    })