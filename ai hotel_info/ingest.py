from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from models import DocumentChunk
from sqlalchemy.orm import Session
from database import engine

EMBEDDING_MODEL = "gemini-embedding-001"
client = genai.Client()

loader = UnstructuredMarkdownLoader("hotel_info.md")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)

chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]

result = client.models.embed_content(
    model=EMBEDDING_MODEL,
    content=texts,
)

embeddings = [item.values for item in result.embeddings]

Session = Session(engine)

with Session as session:
    for chunk, embedding in zip(chunks, embeddings):

        document_chunk = DocumentChunk(
            content=chunk.page_content,
            chunk_metadata=chunk.metadata,
            embedding=embedding
        )

        session.add(document_chunk)

    session.commit()

