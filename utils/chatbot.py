from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the LLM
# This connects to our local Ollama server and specifies the model we want to use.
llm = ChatOllama(model="phi3:mini")

# 2. Define the Prompt Template
# This is the instruction manual for our AI. It tells the LLM its role, the context
# it will receive, and what kind of output it should generate.
# We use {resume_skills} and {job_descriptions} as placeholders for our data.
prompt_template = """
You are an expert AI Career Advisor. Your goal is to provide a detailed, encouraging,
and actionable career analysis for the user based on their skills and relevant job openings.

**Context:**
- User's Resume Skills: {resume_skills}
- Top 3 Job Descriptions They Matched With: {job_descriptions}

**Your Task:**
Based on the provided context, generate the following analysis in a clear, well-structured format using Markdown:

1.  **### Overall Summary:**
    Start with a brief, encouraging summary of the user's current position and their suitability for the matched roles.

2.  **### 🎯 Skill Gap Analysis:**
    Identify the top 5 most important skills that are mentioned in the job descriptions but are MISSING from the user's resume skills. List them as a bulleted list. For each missing skill, briefly explain why it's important for these roles.

3.  **### 💡 Personalized Learning Plan:**
    For the top 3 missing skills you identified, suggest a concrete, project-based way to learn them. Be specific. For example, instead of "Learn React," suggest "Build a personal portfolio website using React and host it on GitHub Pages."

4.  **### ✅ Key Strengths:**
    Identify 3-5 of the user's existing skills that are highly relevant to the job descriptions and will make them a strong candidate. List them and explain why they are a good match.

Use Markdown for formatting (e.g., ### for headers, * for bullet points). Do not mention that you are an AI. Speak directly to the user.
"""

# 3. Create a LangChain "Chain"
# A chain pipes components together. Here, it takes the prompt, formats it with
# user data, sends it to the LLM, and then parses the output into a simple string.
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

# The | symbol is the LangChain Expression Language (LCEL) pipe operator.
chain = prompt | llm | output_parser

# 4. Create a helper function to run the chain
def generate_career_advice(user_skills, matched_jobs):
    """
    Generates career advice by running the LangChain chain.
    """
    # Format the inputs as strings
    skills_str = ", ".join(user_skills)
    jobs_str = "\n\n".join([f"**{job['title']}**\n{job['description']}" for job in matched_jobs])
    
    # Invoke the chain with the formatted context
    response = chain.invoke({
        "resume_skills": skills_str,
        "job_descriptions": jobs_str
    })
    
    return response