from datetime import datetime
import os
import requests
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import tracemalloc
tracemalloc.start()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === AGENT SETUP ===
PROMPT = """
You are a helpful tutor assistant for school students.
Your role is to help students with their studies, answer questions, explain concepts, 
and provide educational guidance across various subjects.

You should:
- Provide clear, age-appropriate explanations
- Help with homework and study questions
- Explain concepts step by step
- Be encouraging and supportive
- Adapt your teaching style to the student's level
"""

# Master model (fallback) - port 8081
master_model = OpenAIModel(
    model_name='qwen/qwen3-8b',
    provider=OpenAIProvider(base_url='http://localhost:8081/v1'),
)

# Slave model (primary) - port 8082
slave_model = OpenAIModel(
    model_name='qwen/qwen3-8b',
    provider=OpenAIProvider(base_url='http://localhost:8082/v1'),
)

class StudentQuestion(BaseModel):
    question: str = Field(description="The student's question or problem")
    subject: Optional[str] = Field(description="The subject area if specified", default=None)
    grade_level: Optional[str] = Field(description="Student's grade level if mentioned", default=None)

class TutorResponse(BaseModel):
    answer: str = Field(description="The tutor's response to the student's question")
    explanation: Optional[str] = Field(description="Additional explanation or context", default=None)
    helpful_tips: Optional[List[str]] = Field(description="Additional study tips or resources", default=None)

class TutorDeps(BaseModel):
    student_question: str
    context: Optional[str] = None

def check_model_health(base_url: str) -> bool:
    """Check if a model endpoint is healthy and responsive."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_model():
    """Return the slave model if available, otherwise fallback to master model."""
    slave_url = 'http://localhost:8082/v1'
    master_url = 'http://localhost:8081/v1'
    
    # Try slave model first
    if check_model_health(slave_url):
        print("Using slave model (port 8082)")
        return slave_model
    
    # Fallback to master model
    if check_model_health(master_url):
        print("Slave model unavailable, using master model (port 8081)")
        return master_model
    
    # If both are down, still return slave as default (will error appropriately)
    print("Warning: Both models appear to be down, defaulting to slave model")
    return slave_model

# Create agent with dynamic model selection
def create_tutor_agent():
    current_model = get_available_model()
    return Agent(
        model=current_model,
        deps_type=TutorDeps,
        result_type=TutorResponse,
        system_prompt=PROMPT
    )

def help_student(question: str, subject: Optional[str] = None, grade_level: Optional[str] = None):
    """Main function to help a student with their question."""
    agent = create_tutor_agent()
    deps = TutorDeps(student_question=question)
    
    # Add context if subject or grade level is provided
    context_parts = []
    if subject:
        context_parts.append(f"Subject: {subject}")
    if grade_level:
        context_parts.append(f"Grade level: {grade_level}")
    
    if context_parts:
        deps.context = " | ".join(context_parts)
    
    result = agent.run_sync(question, deps=deps)
    return result.data

# Create agent instances
tutor_agent = create_tutor_agent()

@app.post("/tutor", response_model=TutorResponse)
async def tutor_endpoint(question_data: StudentQuestion):
    """API endpoint to get tutor help for a student question."""
    try:
        start_time = datetime.now()
        print(f"Received question at {start_time}: {question_data.question}")
        
        response = help_student(
            question_data.question, 
            question_data.subject, 
            question_data.grade_level
        )
        
        end_time = datetime.now()
        print(f"Response generated at {end_time} in {end_time - start_time}")
        print(f"Response: {response.answer[:100]}...")  # Log first 100 chars
        
        return response
        
    except Exception as e:
        print(f"Error processing tutor request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    slave_url = 'http://localhost:8082/v1'
    master_url = 'http://localhost:8081/v1'
    
    slave_status = check_model_health(slave_url)
    master_status = check_model_health(master_url)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "slave_model": {"url": slave_url, "healthy": slave_status},
            "master_model": {"url": master_url, "healthy": master_status}
        }
    }

if __name__ == "__main__":
    print("=== Tutor Agent - FastAPI Service ===")
    print("Master Model: http://localhost:8081")
    print("Slave Model: http://localhost:8082")
    print("API Server: http://localhost:8000")
    print("=" * 50)
    
    uvicorn.run("tutor_agent:app", host="0.0.0.0", port=8000, reload=True)
