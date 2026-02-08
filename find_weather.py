"""
🌦️ Weather Assistant
- Streamlit UI
- Pydantic-AI + Groq
- OpenWeatherMap API
- Secrets loaded from .env
"""

# -----------------------------
# Load environment variables
# -----------------------------
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not found in .env")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("❌ OPENWEATHER_API_KEY not found in .env")

# Ensure Groq libraries can auto-detect the key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# -----------------------------
# Imports
# -----------------------------
import requests
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Weather Assistant",
    page_icon="🌦️",
    layout="centered",
)

st.title("🌦️ Weather Assistant")
st.caption("Ask about the weather in any city — powered by Groq + OpenWeatherMap")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Model temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown("---")
    st.subheader("🔐 Keys loaded")
    st.success("GROQ_API_KEY loaded")
    st.success("OPENWEATHER_API_KEY loaded")

    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Pydantic output schema
# -----------------------------
class WeatherForecast(BaseModel):
    location: str
    description: str
    temperature_celsius: float


# -----------------------------
# AI Agent
# -----------------------------
weather_agent = Agent(
    model="groq:llama-3.1-8b-instant",
    model_settings=ModelSettings(temperature=temperature),
    output_type=str,
    system_prompt=(
        "You are a helpful weather assistant. "
        "When a user asks about weather, extract the city name and call "
        "the `get_weather_forecast` tool. "
        "Respond in a friendly, clear sentence."
    ),
)


# -----------------------------
# Tool: OpenWeatherMap API
# -----------------------------
@weather_agent.tool
def get_weather_forecast(ctx: RunContext, city: str) -> WeatherForecast:
    """Fetch current weather for a city"""

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    res = requests.get(url, params=params, timeout=15)
    data = res.json()

    if res.status_code != 200:
        msg = data.get("message", "Unknown error")
        raise ValueError(f"Could not fetch weather for '{city}': {msg}")

    return WeatherForecast(
        location=data["name"],
        description=data["weather"][0]["description"].capitalize(),
        temperature_celsius=float(data["main"]["temp"]),
    )


# -----------------------------
# Session state (chat history)
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 Ask me about the weather in any city (e.g., *What's the weather in London?*)",
        }
    ]


# -----------------------------
# Render chat history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask about the weather...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Checking the weather... 🌍"):
            try:
                result = weather_agent.run_sync(user_input)
                answer = result.output

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"⚠️ {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
