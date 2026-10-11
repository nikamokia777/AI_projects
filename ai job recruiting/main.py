from google import genai
from google.genai import types
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from typing import Literal


class JobPosting(BaseModel):
    title: str
    required_experience_years: int | None
    required_skills: list[str]
    nice_to_have_skills: list[str]
    work_type: Literal['remote', 'hybrid', 'onsite', 'not_specified']
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    english_level: str | None


class JobAnalysis(JobPosting):
    skills_match: list[str] = Field(description='Candidate skills that match required_skills')
    skills_missing: list[str] = Field(description='Required skills the candidate does not have')

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = 'gemini-2.5-flash'

SYSTEM_PROMPT = (
    'You extract structured data from job postings and compare them with candidate skills. '
    'Use only information explicitly present in the posting. '
    'If something is not mentioned, use null for optional values, an empty list for lists, '
    'and not_specified for work_type. '
    'required_skills are must-have skills, nice_to_have_skills are optional extras. '
    'skills_match are candidate skills that appear in required_skills, case-insensitive. '
    'skills_missing are required skills the candidate does not have.'
)


def analyze_job(job_text: str, candidate_skills: list[str]) -> JobAnalysis:
    skills_text = ', '.join(candidate_skills)
    contents = f'Job posting:\n{job_text}\n\nCandidate skills:\n{skills_text}'

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            response_schema=JobAnalysis,
            temperature=0,
        ),
    )
    return response.parsed

CANDIDATE_SKILLS = ['Python', 'FastAPI', 'PostgreSQL', 'Docker', 'Git']

JOB_POSTINGS = [
    '''
We are looking for a Python Backend Developer.
The candidate should have at least 2 years of experience with
Python and Django. Experience with PostgreSQL and Docker is required.
The position is remote and the salary is $1500-$2500 per month.
English B2 or higher is required.
''',
    '''
NM Studio is hiring a Senior Frontend Engineer for our Tbilisi office.
You will build web apps with React and TypeScript, so 4+ years of experience
with both is a must. Knowledge of Git is required.
Nice to have: Next.js, GraphQL and experience with Docker.
We work in a hybrid mode (3 days in the office). Salary: 4000 - 5500 EUR.
Fluent English (C1) is expected.
''',
    '''
Startup is looking for a Backend Developer to join our team.
You should know Python and be comfortable working with REST APIs.
Great team, interesting product, friendly atmosphere!
''',
]
for job_text in JOB_POSTINGS:
    result = analyze_job(job_text, CANDIDATE_SKILLS)
    print(result.model_dump_json(indent=4))