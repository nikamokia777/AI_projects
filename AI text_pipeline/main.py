from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = 'gemini-3.5-flash-lite'
llm = ChatGoogleGenerativeAI(model = MODEL)

class State(TypedDict):
    product_text: str
    product_info: str
    marketing_text: str
    final_description: str

def extract(state: State) :
    prompt =('extract the product information from the text using the fields below' \
    'Product:...\nDisplay:...\nStorage:...\nCamera:...\nProcessor:...\nConnector:...\nColors:...'
    f'Text{state["product_text"]}' 
    )
    response = llm.invoke(prompt)
    return{"product_info": response.content}
def marketing(state: State) :
    prompt =("Write a short, marketing text (one paragraph) for an online store "
        "based on the product information below. "
        "Use only the facts given and do not invent features.\n\n"
        f"Product information:\n{state['product_info']}")
    response = llm.invoke(prompt)
    return{'marketing_text': response.content}
def format_text(state:State):
    final =(
        '=== PRODUCT DESCRIPTION ===\n\n'
        f'{state["marketing_text"]}'
        '=== END ==='
    )
    return {'final_description': final}
builder = StateGraph(State)
builder.add_node('extract', extract)
builder.add_node('marketing', marketing)
builder.add_node('format text', format_text)

builder.add_edge(START, 'extract')
builder.add_edge('extract', 'marketing')
builder.add_edge('marketing', 'format text')
builder.add_edge('format text', END)

graph = builder.compile()

result = graph.invoke(
    {
    "product_text": "iphone 15, 6.1 inch oled, 128gb, a16 bionic, 48mp camera, usb-c, battery good, available in black and blue",
    "product_info": "",
    "marketing_text": "",
    "final_description": "",
    }
)
print(result['final_description'])
    
    

    