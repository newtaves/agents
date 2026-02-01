from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from .youtube_script_tool import get_subtitles

# 1. Logic Functions
def exit_loop():
    """Call ONLY when the script is perfect and needs no further changes."""
    return {"status": "approved", "message": "Refinement complete."}

# 2. Strategy Agent: Merged Manager & Researcher
# Uses 'lite' model for cost-effective requirements gathering and technical subtitle extraction
strategy_agent = LlmAgent(
    name="strategy_agent",
    model=Gemini(model="gemini-2.5-flash-lite"),
    tools=[get_subtitles],
    instruction="""
    Gather video topic, audience, style, and length from input. 
    If YT links provided, use `get_subtitles`. 
    Output: Combined requirements and research data. 
    No greetings. Direct data only.
    """,
    output_key="research_package"
)

# 3. Writer Agent: Merged Outline & Scripting
# Uses standard 'flash' for better creative writing quality
writer_agent = LlmAgent(
    name="writer_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction="""
    Create outline and full YouTube script based on: {research_package}. 
    Follow style and audience constraints strictly. 
    No preamble. Output script immediately.
    """,
    output_key="final_script"
)

# 4. Refiner Agent: Merged Critique & Rewriter
# Handles self-correction in a single turn to save loop tokens
refiner_agent = LlmAgent(
    name="refiner_agent",
    model=Gemini(model="gemini-2.5-flash"),
    tools=[exit_loop],
    instruction="""
    Compare {final_script} against {research_package}. 
    If it meets all requirements, call `exit_loop`. 
    Otherwise, rewrite the script to fix gaps. 
    No feedback prose; output the improved script or the tool call only.
    """,
    output_key="final_script"
)

# 5. Optimized Workflow
# Reduced to 3 main steps with a tighter loop
root_agent = SequentialAgent(
    name="script_manager",
    sub_agents=[
        strategy_agent, 
        writer_agent, 
        LoopAgent(
            name="refinement_loop", 
            sub_agents=[refiner_agent], 
            max_iterations=2
        )
    ]
)

# if __name__ == "__main__":
#     import asyncio
#     async def main():
#         runner = InMemoryRunner(agent=root_agent)
#         # Initial input should provide the topic/links
#         response = await runner.run("Create a 10 min script about AI for beginners using this link: https://youtube.com/...")
#         print(response)
#     asyncio.run(main())