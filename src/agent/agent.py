
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from modeling.predictor import PotentialPredictor
import pandas as pd

load_dotenv()

# Initialize predictor
predictor = PotentialPredictor()

# Load city aggregates for context
try:
    df_city_data = pd.read_csv('data/city_potential_data.csv')
except:
    df_city_data = None

from typing import Optional

@tool
def get_city_solar_potential(city_name: str):
    """
    Retrieves the predicted annual average daily solar output (kWh/kWp/day) for a specific Australian city
    based on its long-term average weather profile.
    """
    if df_city_data is None:
        return "City potential data not available."
    
    city_name = city_name.capitalize().replace(" ", "")
    city_row = df_city_data[df_city_data['Location'] == city_name]
    
    if city_row.empty:
        return f"No data found for city: {city_name}. Please check the spelling or try a major Australian city."
    
    weather_profile = city_row.to_dict('records')[0]
    prediction = predictor.predict(weather_profile)
    
    return f"City: {city_name}, Potential: {prediction:.2f} kWh/kWp/day"

@tool
def estimate_custom_site_potential(avg_sunshine: Optional[float] = None, avg_max_temp: Optional[float] = None, avg_cloud_cover: Optional[float] = None):
    """
    Estimates the solar potential for a custom site given its average annual weather conditions.
    Inputs (All Optional):
    - avg_sunshine: Average daily sunshine hours.
    - avg_max_temp: Average daily maximum temperature (Celsius).
    - avg_cloud_cover: Average cloud cover (oktas 0-8).
    """
    profile = {}
    if avg_sunshine is not None: profile['Sunshine'] = avg_sunshine
    if avg_max_temp is not None: profile['MaxTemp'] = avg_max_temp
    if avg_cloud_cover is not None: profile['Cloud3pm'] = avg_cloud_cover
    
    prediction = predictor.predict(profile)
    return f"Estimated Potential: {prediction:.2f} kWh/kWp/day (Note: Missing parameters were filled with Australian averages)."

@tool
def calculate_solar_farm_yield(solar_potential: float, farm_size_m2: float):
    """
    Calculates the final estimated daily energy yield (kWh) for a solar farm.
    This tool ALREADY accounts for installation density (1 kWp per 5 m2).
    Inputs:
    - solar_potential: The numeric potential (kWh/kWp/day) returned by other tools.
    - farm_size_m2: Total area of the farm in square meters.
    """
    kwp_installed = farm_size_m2 / 5.0
    daily_yield = solar_potential * kwp_installed
    return f"FINAL CALCULATION RESULT: A {farm_size_m2}m2 farm with {solar_potential:.2f} kWh/kWp/day potential will yield EXACTLY {daily_yield:.2f} kWh per day."

# Setup Agent
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
tools = [get_city_solar_potential, estimate_custom_site_potential, calculate_solar_farm_yield]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Solar Potential Intelligence Agent. 
    You help users understand the long-term solar energy potential of various locations in Australia.
    
    STRICT MISSING DATA POLICY:
    - Before calling 'estimate_custom_site_potential', you MUST check if the user has provided all three: sunshine hours, temperature, and cloud cover.
    - If ANY of these three are missing, you MUST NOT call the tool immediately.
    - INSTEAD, your FIRST response must be to inform the user which values are missing and ASK them if they want to provide them for accuracy OR proceed using national averages.
    - ONLY call the 'estimate_custom_site_potential' tool if the user explicitly says "proceed", "use averages", or provides the missing data.
    
    CALCULATION RULES:
    1. For multi-step tasks (e.g., getting potential THEN calculating yield), ALWAYS perform steps sequentially.
    2. USE THE EXACT numeric values returned by tools. 
    3. CRITICAL: When a tool provides a "FINAL CALCULATION RESULT", you MUST report that number exactly. 
    4. DO NOT perform your own mathematical calculations or override tool results.
    5. Your final response must be a clean, natural language summary. DO NOT include internal tool syntax.
    
    Always maintain a professional, data-driven, and business-friendly tone.
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def query_agent(question):
    response = agent_executor.invoke({"input": question})
    return response["output"]

if __name__ == "__main__":
    print(query_agent("What is the typical solar potential in Sydney and how much would a 1000m2 farm produce there?"))
