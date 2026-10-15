from dotenv import load_dotenv
load_dotenv() 
EMBEDDING_MODEL = "gemini-embedding-001"

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

client = genai.Client()

loader = UnstructuredMarkdownLoader("hotel_info.md")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]
results = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
embeddings = [e.values for e in results.embeddings]
question = input("შეიყვანეთ კითხვა: ")
query_result = client.models.embed_content(model=EMBEDDING_MODEL, contents=question)
query_embedding = query_result.embeddings[0].values
similarities = cosine_similarity([query_embedding], embeddings)[0]
results = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)


for i, (chunk, similarities) in enumerate(results[:3]):
    print(f"Chunk {i + 1}:")
    print(f"Score: {similarities}")
    print(f"Data: {chunk.page_content}")
    print("=======================================")