from dotenv import load_dotenv
load_dotenv()

from google import genai
from sqlalchemy.orm import Session
from sqlalchemy import select
from langchain_google_genai import ChatGoogleGenerativeAI

from database import engine
from models import DocumentChunk

client = genai.Client()

question = input("Enter your question: ")
query_results = client.models.embed_content(
    model = 'gemini-embedding-001',
    contents = question
)
query_embedding = query_results.embeddings[0].values

query = select(DocumentChunk).order_by(DocumentChunk.embedding.cosine_distance(query_embedding)).limit(3)

with Session(engine) as session:
    results = session.execute(query).scalars().all()

llm = ChatGoogleGenerativeAI(model_name="gemini-3.5-flash", temperature=0.2)
context = "\n".join(result.content for result in results)
response = llm.invoke(f'answer the question only ny folllowing context below:   Context = {context} Quetion = {question}')
print(response.content)