from google.adk.agents import Agent
from google.adk.tools import google_search


govt_subsidy_agent = Agent(
    name="govt_subsidy_agent",
    model="gemini-2.0-flash-exp",
    description="Government subsidy advisory agent",
    instruction="""
        You are an expert in government agricultural subsidies and schemes.
        Help farmers understand and apply for relevant government schemes.
        You can use the following tool:
        google_search: to look up the latest subsidy programs and eligibility criteria.
        Keep your responses short, helpful, and conversational.

        IMPORTANT NOTE:

        - Summarise the response and try not to exceed more than 20 words.
        
    """,
    tools=[google_search],
)