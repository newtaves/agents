from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


outline_agent = Agent(
    name="OutlineAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""Create a blog outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A concluding thought""",
    output_key="outline"
)
print("Outline Agnet created!!")


writerAgent=Agent(
    name="WriterAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config
    ),
    instruction="""Following this outline: {outline}
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
    output_key="blog_draft"
)
print("Writer Agent Created")


editorAgent=Agent(
    name="WriterAgent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
    ),
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, 
    improving the flow and sentence structure, and enhancing overall clarity.""",
    output_key="final_blog"
)
print("Editor Agent Created!!")


root_agent=SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_agent, writerAgent, editorAgent]
)