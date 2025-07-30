import streamlit as st
from utils.resume_parser import process_resume
from utils.job_matcher import find_matching_jobs
from utils.chatbot import generate_career_advice

st.set_page_config(layout="wide")

st.title("AI Career Guidance Platform")
st.write("Upload your resume, find matching jobs, and get personalized AI-powered career advice.")

if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'job_matches' not in st.session_state:
    st.session_state.job_matches = None
if 'career_advice' not in st.session_state:
    st.session_state.career_advice = None


col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: Your Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)",
        type=['pdf', 'docx']
    )
    
    if uploaded_file:
        if st.session_state.resume_data is None:
            with st.spinner("Analyzing resume..."):
                st.session_state.resume_data = process_resume(uploaded_file)

    if st.session_state.resume_data:
        if "error" in st.session_state.resume_data:
            st.error(st.session_state.resume_data["error"])
        else:
            st.success("Resume Analyzed!")
            st.subheader(f"👤 {st.session_state.resume_data.get('name', 'Candidate')}")
            st.write(f"📧 {st.session_state.resume_data.get('email', 'No email found')}")
            
            st.subheader("🛠️ Your Skills")
            skills = st.session_state.resume_data.get("skills", [])
            if skills:
                st.expander("View Skills", expanded=False).write(", ".join(skills))
            else:
                st.warning("No skills found.")

with col2:
    if st.session_state.resume_data and "error" not in st.session_state.resume_data:
        st.header("Step 2: Job Matching")
        if st.button("Find Matching Jobs"):
            with st.spinner("Searching for the best job matches..."):
                user_skills = st.session_state.resume_data["skills"]
                st.session_state.job_matches = find_matching_jobs(user_skills)

        if st.session_state.job_matches:
            st.subheader("Top 3 Job Recommendations")
            for job in st.session_state.job_matches:
                st.info(f"**{job['title']}**")
            
            st.header("Step 3: Get AI Advice")
            if st.button("Generate Career Advice"):
                with st.spinner("🤖 Our AI Advisor is thinking... This may take a moment."):
                    st.session_state.career_advice = generate_career_advice(
                        st.session_state.resume_data["skills"],
                        st.session_state.job_matches
                    )
    
    if st.session_state.career_advice:
        st.subheader("✨ Your Personalized Career Analysis ✨")
        st.markdown(st.session_state.career_advice)

