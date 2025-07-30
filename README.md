# AI Career Guidance Platform

This project is an end-to-end AI-powered platform that analyzes resumes, matches them to relevant job descriptions using semantic search, and generates personalized career advice with a Retrieval-Augmented Generation (RAG) pipeline running on a local LLM.


# Key Features

- Resume Parsing:** Upload a PDF or DOCX resume and automatically extract skills, name, and email using spaCy.
- Semantic Job Matching:** Instead of keywords, it uses sentence embeddings to find jobs based on contextual meaning.
- AI-Powered Analysis:** Leverages a local LLM (Phi-3) to provide a detailed skill-gap analysis, identify strengths, and suggest a personalized learning plan.
- 100% Open-Source & Local:** Runs entirely on your machine for free, with no API keys required.

# Tech Stack

- **Frontend:** Streamlit
- **AI Orchestration:** LangChain
- **LLM:** Ollama with `phi3:mini`
- **NLP & Parsing:** spaCy, PyPDF2, python-docx
- **Semantic Search (RAG):** Sentence-Transformers, ChromaDB, scikit-learn
- **Data Handling:** Python (Pandas, NumPy)


## Getting Started ##

# Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running.

## Installation & Setup

1.  Clone the repository:**
    ```bash
    git clone https://github.com/[YourUsername]/ai-career-guide.git
    cd ai-career-guide
    ```

2.  Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  Download the NLP model for spaCy:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  Pull the LLM model with Ollama:**
    ```bash
    ollama pull phi3:mini
    ```

## Running the Application

Once setup is complete, run the Streamlit app:
```bash
streamlit run app.py
```

### 🎥 Live Demo

[Watch a full demonstration of the project on Loom](https://www.loom.com/share/e5db890e46ff4dafaf24dbe694f42a1f?sid=4877805b-973f-4998-9612-3c6a2658814e)
