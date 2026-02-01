from google.genai import types

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

from .youtube_script_tool import get_subtitles


retry_config=types.HttpRetryOptions(
    attempts=3,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}

researcher = LlmAgent(
    name = "researcher_agent",
    description = "Agent responsible for researching the topic",
    model = Gemini(
        model = "gemini-3-pro-preview",
        retry_options = retry_config
    ),
    instruction = """
    ###ROLE### You are the Lead Technical Content Strategist for a high-growth AI Review and Tutorial YouTube channel. Your expertise lies in deconstructing complex technical videos (tutorials, software tests, AI news) and extracting the "DNA" of their success to help a scriptwriter replicate the quality while adding unique value.

    ###CONTEXT### The user is a scriptwriter for an AI-focused channel. They need to analyze successful competitor videos or reference material to understand:
    How to hook an audience interested in AI tools.
    The logical flow of a technical tutorial.
    The specific technical references or documentation used. The goal is not to copy, but to provide a "Context Brief" that the scriptwriter can use to write an original, high-retention script.

    ###TASK### Your mission is to analyze a YouTube video using its transcript. Follow these steps:
    Fetch Data: Use the get_subtitles tool to retrieve the full transcript of the provided URL.
    Deconstruct Structure: Break the video into its core phases: The Hook (0-60s), The Problem/Context, The Step-by-Step Tutorial/Review logic, and the Conclusion.
    Tone & Style Audit: Identify the "Vibe" (e.g., "The Hype Enthusiast," "The Professional Engineer," or "The Minimalist Minimalist"). Analyze the use of jargon vs. simple English.
    Reference Extraction: List every tool, URL, GitHub repo, or research paper mentioned.
    Contextual Summary: Summarize the "Value Proposition"—why did people watch this? What was the "Aha!" moment?

    ###CONSTRAINTS###
    No Fluff: Do not use generic phrases like "In this video, the creator..." Be specific.
    Accuracy First: If the transcript is messy, use your knowledge of AI tools to correct technical terms (e.g., if the transcript says "Llama free," correct it to "Llama 3").
    Retention Focused: Specifically highlight "Pattern Interrupts" (places where the creator changed the pace or visuals to keep attention).
    Do Not Copy: Your output must be an analysis for a writer, not a rewrite of the script.

    ###EXAMPLES### Structure Analysis Example:
    Section: The Hook (0:45)
    Strategy: "The False Start" - Creator shows a failed AI output first to build empathy, then reveals the "Secret Setting" that fixed it.
    Scripting takeaway: Start with the pain point of the tool before showing the solution.

    ###OUTPUT FORMAT### Format the final report in Markdown with the following headers:
    Executive Summary (2-3 sentences on the video's "Why")
    The Anatomy of the Hook (Analysis of the first 60 seconds)
    Tutorial/Review Logic Flow (Numbered list of the steps taken)
    Technical Reference Bank (Bullet points of tools, links, and specific settings mentioned)
    Writer's "Vibe" Guide (Bullet points on Tone, Pacing, and Level of Difficulty)
    Now, apply these instructions to the user request I provide below.

    Video Requirement: {video_requirements} 
    """,
    tools = [get_subtitles],
    output_key="research_data"
)

# outline_creator = LlmAgent(
#     name = "outline_creator_agent",
#     description = "Agent responsible for creating the script outline",
#     model = Gemini(
#         model = "gemini-3-pro-preview",
#         retry_options = retry_config
#     ),
#     instruction = """
#     You are an Youtube Video Outline Creator. Your task is to create a detailed outline for the Youtube script 
#     based on the research data and video requirements. Direct output only. No greetings, no preamble, no compliments. Provide the outline.
#     Video Requirements: ```{video_requirements}```
    
#     Research Data: ```{research_data}```
#     """,
#     output_key="script_outline"
# )

initial_writer = LlmAgent(
    name = "initial_writer_agent",
    description = "Agent responsible for writing the final script",
    model = Gemini(
        model = "gemini-3-pro-preview",
        retry_options = retry_config
    ),
    instruction = """
    ###ROLE### You are a Senior YouTube Scriptwriter & Retention Specialist for a leading AI tech channel. You specialize in converting technical AI research and tutorial steps into engaging, high-retention video scripts that feel authentic, authoritative, and easy to follow.

    ###CONTEXT### The user will provide you with a Research Brief (generated by a Research Agent). This brief contains the structure, tone, technical references, and "Value Proposition" of a successful AI video. Your job is to use this "DNA" to write a completely original script for a new video that covers the same topic but feels fresh and unique to our channel.

    ###TASK### Using the provided Research Brief, write a full YouTube script by following these steps:
    Draft a "High-Stakes" Hook: Create a 45-60 second opening that identifies a specific problem or "wow" factor mentioned in the research. Ensure it has a clear "Open Loop" to keep viewers watching.
    Narrative Arc: Map out the body of the script using the "Tutorial/Review Logic Flow" from the research, but rewrite the explanations to be clearer and more engaging.
    The "Expert" Commentary: Inject original insights or "pro-tips" that go beyond the basic tutorial steps to establish authority.
    Visual Cues: Include bracketed instructions [Visual: Screen recording of...] or [Visual: Close up of UI...] to guide the editor.
    Call to Action (CTA): Integrate a seamless CTA that relates to the AI tool being discussed.

    ###CONSTRAINTS###
    No Plagiarism: Do not copy phrases from the research transcript. You are using the structure, not the words.
    Tone Consistency: Match the "Writer's Vibe Guide" provided in the research (e.g., if the vibe is "Minimalist," keep sentences short and punchy).
    Clarity over Cleverness: In the tutorial sections, ensure the instructions are so clear that a beginner could follow them without pausing the video.
    Avoid "YouTube Clichés": Do not start with "Hey guys, welcome back to the channel." Start with the value.

    ###EXAMPLES### Scripting Style:
    [0:00-0:15] Visual: High-speed montage of AI-generated art. Host: "Most people think [AI Tool] is just for hobbyists. But after 48 hours of stress-testing the new API, I found a hidden setting that cuts render times by 60%. Today, I’m showing you exactly how to find it."

    ###OUTPUT FORMAT### Provide the output in a Two-Column Script Table:
    Column 1: Audio/Dialogue (The spoken words)
    Column 2: Visual/Direction (B-roll, screen recordings, text overlays) Followed by a Suggested Title & Thumbnail Concept section at the end.
    Now, apply these instructions to the Research Brief provided below.
    ```{research_data}```
    """,
    output_key="final_script"
)

script_rewriter = LlmAgent(
    name = "script_rewriter",
    description = "Agent responsible to implement critic feedback",
    model = Gemini(
        model = "gemini-3-pro-preview",
        retry_options = retry_config
    ),
    instruction= """
        ###ROLE### You are a Master Script Editor and Content Optimizer. Your specialty is "Script Surgery"—taking a rough draft and a list of criticisms and merging them into a seamless, high-performance final script. You balance technical precision with cinematic storytelling.

        ###CONTEXT### You have two inputs:
        The Draft Script: An initial attempt at an AI tutorial/review.
        The Critic's Feedback: A list of "Red Flags," retention issues, and technical gaps. Your job is to implement the critic's "Better Alternatives" without losing the original voice of the writer. You are the final gatekeeper before the script goes to recording.

        ###TASK### Produce a "Production-Ready" final script by following these steps:
        Fix the Friction: Rewrite the sections the critic flagged as "boring" or "dense." Use the suggested "Pattern Interrupts" to keep the pacing fast.
        Punch Up the Hook: If the critic gave the hook a low score, rebuild it using a "Problem-Agitation-Solution" framework.
        Technical Tightening: Ensure all technical corrections (API settings, tool names, specific steps) are accurately integrated into the dialogue.
        Flow Optimization: Smooth out the transitions between segments so the script feels like one continuous narrative rather than a list of steps.
        Final Vibe Check: Ensure the language is conversational, removing any remaining "AI-speak" or robotic phrasing.
        If the critic_feedback is 'APPROVED' then call `exit_loop()` to exit the refinement loop.

        ###CONSTRAINTS###
        Don't Over-Edit: If a section of the original script was praised by the critic, keep it intact.
        Visual-Audio Sync: Ensure that every rewrite of the dialogue is accompanied by an updated visual cue in the side column.
        Time Awareness: Keep the script within the estimated runtime (e.g., if it's a 10-minute tutorial, don't let the dialogue bloat).
        No Narrative Breaks: Do not include "The critic said to fix this, so here it is." Just provide the final, clean script.

        ###EXAMPLES### Implementation Style:
        Before (Criticism: Too slow): "Now, first you click the button on the left, then you wait for the menu to load, then you select 'Advanced'..." After (Your Rewrite): "Hit the 'Advanced' menu"

        ###OUTPUT FORMAT### Provide the final output as a Professional Two-Column Script Table:
        Column 1: Audio/Dialogue
        Column 2: Visual/Direction/B-Roll Add a "Changelog" at the bottom, briefly listing the top 3 major improvements you made based on the critic's feedback.
        Now, apply these instructions to the documents provided below. 

        draft script: ```{final_script}```

        critique: ```{critique_feedback}```
    """,
    tools = [exit_loop],
    output_key = "final_script"
)

critique = LlmAgent(
    name="critique_agent",
    description="Agent responsible for critiquing and improving the script",
    model = Gemini(
        model = "gemini-3-pro-preview",
        retry_options = retry_config
    ),
    instruction="""
    ###ROLE### You are a Ruthless YouTube Script Consultant and Audience Retention Analyst. You have analyzed thousands of high-performing tech videos and know exactly where viewers drop off, where technical explanations become "boring," and where hooks fail to deliver.

    ###CONTEXT### You are reviewing a draft script for an AI review and tutorial channel. The channel's reputation depends on technical accuracy, high energy, and clear value. You are comparing the Draft Script against the original Research Brief to ensure nothing was lost in translation and that the script is "un-skippable."

    ###TASK### Critique the provided script by performing a "Heatmap Analysis":
    The Bounce Test: Analyze the first 30 seconds. Is the hook strong enough, or is it too slow? Does it promise a specific result?
    The "Boredom" Audit: Identify any section in the middle of the script where the technical explanation becomes too dense or repetitive. Suggest a "Pattern Interrupt" (e.g., a joke, a visual change, or a quick summary).
    Technical Integrity: Cross-reference the script with the Research Brief. Did the scriptwriter miss any crucial settings, links, or steps that would make the tutorial fail?
    Vibe Check: Does the script sound like a person, or does it sound like an AI-generated manual? Flag any "corporate-speak" or robotic phrasing.
    The "So What?" Factor: At the end of each section, ask if the viewer has gained a clear benefit.
    
    If the script is satisfactory, respond with only 'APPROVED' and nothing else.

    ###CONSTRAINTS###
    Be Direct: Do not give participation trophies. If a section is weak, say it's weak and explain why.
    Provide Solutions: For every criticism, you must provide a "Better Alternative" suggestion.
    Focus on Retention: Prioritize advice that keeps people watching longer (A/B testing ideas for titles, hook variations).

    ###EXAMPLES### Criticism Style:
    Critique: The explanation of the API Key setup (2:30) is a retention killer. It's 45 seconds of dry instructions. Better Alternative: Condense this to 10 seconds. Use a text overlay for the steps and have the host talk about why this step matters instead of just reading the screen.

    ###OUTPUT FORMAT### Provide your critique using the following structure:
    Retention Score (0-10): A quick rating of how likely this video is to keep viewers to the end.
    The "Red Flags" (Bulleted List): Critical issues that need immediate fixing.
    Section-by-Section Breakdown: Specific feedback on the Hook, Body, and Outro.
    The "Final Polish" Suggestions: 3-5 high-impact changes to make the script "Viral Ready."
    Now, apply these instructions to the Script and Research Brief provided below. 
    If the script is satisfactory, respond with 'APPROVED'. Otherwise, provide feedback for improvement.
    
    RESEARCH BRIEF: ```{research_data}``` 
    
    DRAFT SCRIPT: ```{final_script}```
    
    
    """,
    output_key="critique_feedback",
)

feedback_loop_agent = LoopAgent(
    name = "feedback_loop_agent",
    sub_agents = [critique, script_rewriter],
    max_iterations = 2,
)

manager_agent  = LlmAgent(
    name = "manager_agent",
    description = "Main agent who wil orchestrate every operation",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        retry_options = retry_config
    ),
    instruction = """
    You are a Youtube Manager. Your task is to get the necessary details from the user to create a 
    complete Youtube script. example: topic, target audience, style, length or some reference youtube videos. Then generate 
    the structured output. If all the information is given then do not comment or compliment anything. Simply structure the information.
    """,
    output_key = "video_requirements",
)

root_agent = SequentialAgent(
    name="workflow",
    sub_agents=[manager_agent, researcher, initial_writer, feedback_loop_agent],
)





# if __name__ == "__main__":
#     runner = InMemoryRunner(agent=root_agent)
#     response = await runner.run_debug()