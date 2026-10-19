from dotenv import load_dotenv
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage

MODEL = "gemini-3.5-flash-lite"
@tool
def hotel_search(city: str):
    'Get hotel information for a given city'
    hotels = [
        {"name": "Hotel A", "city": 'kutaisi', "price": 100},
        {"name": "Hotel B", "city": 'batumi', "price": 150},
        {"name": "Hotel C", "city": 'tbilisi', "price": 200},
    ]
    result = [hotel for hotel in hotels if hotel["city"].lower() == city.lower()]
    if not result:
        return f"No hotels found in {city}."
    return result

@tool
def get_exchange_rate(currency: str):
    'Get exchange rate for a given currency'
    exchange_rates = {
        "USD": 2.70,
        "EUR": 2.85,
        "GBP": 3.6,
    }
    rate = exchange_rates.get(currency.upper())
    if rate is None:
        return f"Exchange rate for {currency} not found."
    return f"1 GEL = {rate} {currency.upper()}"

@tool
def get_attractions(city: str):
    'Get attraction information for a given city'
    attractions = {
        "kutaisi": ["Bagrati Cathedral", "Gelati Monastery", "Prometheus Cave"],
        "batumi": ["Batumi Boulevard", "Batumi Botanical Garden", "Ali and Nino Statue"],
        "tbilisi": ["Narikala Fortress", "Tbilisi Old Town", "Rustaveli Avenue"],
    }
    result = attractions.get(city.lower())
    if not result:
        return f"No attractions found in {city}."
    return result

llm = ChatGoogleGenerativeAI(model = MODEL)
tools = [hotel_search, get_exchange_rate, get_attractions]
llm_with_tools = llm.bind_tools(tools)
question = input("Ask a question: ")
response = llm_with_tools.invoke(question)
tool_results = []
for tool_call in response.tool_calls:
    tool_name = tool_call['name']
    tool_args = tool_call['args']
    selected_tool = next(tool for tool in tools if tool_name == tool.name)

    result = selected_tool.invoke(tool_args)

    tool_results.append(ToolMessage(
         tool_call_id=tool_call["id"],
         content=str(result)
     ))

messages = [
     {
         "role": "user",
         "content": question
     },
     response,
     *tool_results
 ]

final_response = llm_with_tools.invoke(messages)

print("Final Answer:")
print(final_response.text)
