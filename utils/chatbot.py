from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="phi3:mini")

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

prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

def generate_career_advice(user_skills, matched_jobs):
    """
    Generates career advice by running the LangChain chain.
    """

    skills_str = ", ".join(user_skills)
    jobs_str = "\n\n".join([f"**{job['title']}**\n{job['description']}" for job in matched_jobs])
    
    response = chain.invoke({
        "resume_skills": skills_str,
        "job_descriptions": jobs_str
    })
    
    return response
