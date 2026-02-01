from google.adk.agents.llm_agent import Agent
from google.adk.models import Gemini
from google.adk.tools import AgentTool, url_context
from .youtube_script_tool import get_subtitles


script_writer = Agent(
    name="script_writer",
    model = Gemini(
        model = "gemini-3-pro-preview"
    ),
    instruction = "You are a helpful script writer",
    

)

root_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash"
    ),
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction="""
    Your task is to fetch the context of url and provide it to the user.
    """,
    tools = [get_subtitles, url_context]
)

