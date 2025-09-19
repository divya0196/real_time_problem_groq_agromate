from google.adk.agents import Agent
from .subagents.expert_agent.agent import expert_agent
from .subagents.govt_subsidy_agent.agent import govt_subsidy_agent
from .subagents.local_fertilzer_discount_agent.agent import local_fertilizer_discount_agent
from .subagents.weather_forecast_agent.agent import weather_forecast_agent
from google.adk.tools.agent_tool import AgentTool
from .tools.tools import get_current_time, get_location

root_agent = Agent(
    name="agro_mate",
    #model="gemini-2.0-flash",
    model="gemini-2.0-flash-exp",
    description="Agro Mate",
    #subagents=[],
    instruction="""
    You are a helpful assistant that greets the user. Your name is Agro mate. Greet them and ask what's their name.
    
    You have to pass the user's name, location to all the sub agents.
    You have access to the following tools:
    1. Expert Agent: This agent can provide expert advice on various topics.
    2. Get Current Time: This tool provides the current time.
    3. Get Location: This tool provides the user's current location.
    Use these tools to assist the user effectively.

  

    IMPORTANT NOTE:

    - Summarise the response and try not to exceed more than 20 words.

    """,
    tools=[
            AgentTool(expert_agent),
            AgentTool(govt_subsidy_agent),
            AgentTool(local_fertilizer_discount_agent),
            AgentTool(weather_forecast_agent),
            get_current_time, get_location
         ],
)