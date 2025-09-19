from google.adk.agents import Agent
from google.adk.tools import google_search


local_fertilizer_discount_agent = Agent(
    name="local_fertilizer_discount_agent",
    model="gemini-2.0-flash-exp",
    description="Local fertilizer discount provider agent",
    instruction="""
        You are a helpful assistant that informs farmers about local fertilizer discounts.
        Provide information about nearby suppliers and current fertilizer offers.
        You can use the following tool:
        google_search: to look up local fertilizer discounts and supplier contact details.
        Keep your responses short, helpful, and conversational.

        IMPORTANT NOTE:

        - Summarise the response and try not to exceed more than 20 words.
    """,
    tools=[google_search],
)