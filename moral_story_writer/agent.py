from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from google.adk.models import Gemini
from google.genai import types


retry_config = types.HttpRetryOptions(
    attempts=5,
    initial_delay=2,
    exp_base=3,
    http_status_codes=[503,429]
)


