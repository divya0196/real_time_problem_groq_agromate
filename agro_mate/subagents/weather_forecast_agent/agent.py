from google.adk.agents import Agent
from google.adk.tools import google_search


weather_forecast_agent = Agent(
    name="weather_forecast_agent",
    model="gemini-2.0-flash-exp",
    description="Weather forecasting agent",
    instruction="""
        You are a weather forecasting assistant for farmers.
        Provide accurate and timely weather updates for their location.
        You can use the following tool:
        google_search: to look up weather forecasts and climate data.
        Keep your responses short, helpful, and conversational.

        IMPORTANT NOTE:

        - Summarise the response and try not to exceed more than 20 words.
    """,
    tools=[google_search],
)