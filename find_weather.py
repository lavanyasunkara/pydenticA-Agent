"""
Weather Assistant using Pydantic-AI and OpenWeatherMap API
This script creates a conversational agent that can respond to weather-related queries
using the OpenWeatherMap API and a Groq-hosted LLaMA model.
"""

# Allow nested async loops (useful for Jupyter or async environments)
import nest_asyncio
nest_asyncio.apply()

# Standard libraries
import os
import requests

# Pydantic model for structured data
from pydantic import BaseModel

# Core AI libraries from pydantic_ai
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

os.environ["GROQ_API_KEY"] = ""

# 1. Define the output schema of the tool using Pydantic
class WeatherForecast(BaseModel):
    location: str
    description: str
    temperature_celsius: float

# 2. Create the AI agent using Groq’s LLaMA 3 model
weather_agent = Agent(
    model="groq:llama-3.1-8b-instant",
    model_settings=ModelSettings(temperature=0.2),
    output_type=str,
    system_prompt=(
        "You are a helpful weather assistant. Use the 'get_weather_forecast' tool to "
        "find current weather conditions for any city. Provide clean and friendly answers."
    )
)

# 3. Register a tool with the agent to fetch real-time weather using OpenWeatherMap API
@weather_agent.tool
def get_weather_forecast(ctx: RunContext, city: str) -> WeatherForecast:
    """
    Tool: get_weather_forecast
    Description: Fetches current weather for a city using the OpenWeatherMap API.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    
    # Replace this with your own API key for production use
    api_key = ""
    
    # Query parameters
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    # Send request to weather API
    res = requests.get(url, params=params).json()

    # Return the formatted weather information
    return WeatherForecast(
        location=res["name"],
        description=res["weather"][0]["description"].capitalize(),
        temperature_celsius=res["main"]["temp"]
    )

# 4. Run continuous user interaction loop
if __name__ == "__main__":
    print("🌦️  Weather Assistant is ready! Type 'exit' to quit.")
    print("‒" * 50)
    while True:
        question = input("🌤️ Ask about the weather: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            print("\n👋 Exiting weather assistant. Have a nice day!")
            break

        try:
            print(question)
            result = weather_agent.run_sync(question)
            print("\n📍 Forecast:", result.output)
        except Exception as e:
            print("⚠️ Error:", str(e))

        print("‒" * 50)
