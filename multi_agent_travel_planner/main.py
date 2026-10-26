from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = 'gemini-3.6-flash'

llm = ChatGoogleGenerativeAI(model = MODEL)

class State(TypedDict):
    user_request: str    
    destination_info: str    
    food_info: str    
    budget_info: str    
    final_plan: str

def destination_agent(state: State):
    user_request = state['user_request']
    prompt = [
        SystemMessage(
            content="""
            You are destination_agent, a travel destination expert.
            Analyze the destination mentioned in the user's request.
            Determine which types of activities and attractions would interest the user based on their stated interests.
            Return specific recommendations (places, neighborhoods, activities), taking the number of days into account.
            Do not write food recommendations, budget calculations, or a full travel itinerary.
            """
        ),
        HumanMessage(
            content = f'User request : {user_request}'
        )
    ]
    response = llm.invoke(prompt)
    return {
        'destination_info': response.text
    }
def food_agent(state : State):
    user_request = state['user_request']
    prompt = [
        SystemMessage(
         content = """
            You are food_agent, a food and gastronomy expert.
            Determine dining recommendations based on the destination.
            Take the user's budget and food-related preferences into account.
            Suggest local dishes, drinks, and types of restaurants or places to eat.
            Do not plan the whole trip, and do not write attraction recommendations or a cost analysis.
            """   
        ),
        HumanMessage(content = f'User request: {user_request}')
    ]

    response = llm.invoke(prompt)

    return{
        'food_info': response.text
    }

def budget_agent(state: State):
    user_request = state['user_request']
    prompt = [
        SystemMessage(
            content = """
            You are budget_agent, a travel cost analyst.
            Analyze the estimated trip costs based on the user's budget level and trip duration.
            Break the costs into the main categories: hotel, food, transport, activities.
            Give an estimated amount (range) for each category and a total estimate.
            Do not write activity recommendations, restaurant lists, or a day-by-day itinerary.
            """
        ),
        HumanMessage(content = f'User request : {user_request}')
    ]

    response = llm.invoke(prompt)

    return{
        'budget_info': response.text
    }
def planner_agent(state : State):
    user_request = state['user_request']
    destination_info = state['destination_info']
    food_info = state['food_info']
    budget_info = state['budget_info']

    prompt =[
        SystemMessage(
            content = """
            You are planner_agent, a travel plan compiler.
            Receive the information prepared by the destination, food, and budget agents.
            Merge it into a single, logical travel plan organized by day.
            Do not generate new information from scratch. Use only the data provided by the agents.
            If some information is missing, do not invent it.
            """
        ),
        HumanMessage (
            content = f"""
            User request : {user_request}
            Destination agent output : {destination_info}
            Food agent output : {food_info}
            Budget agent output : {budget_info}
            """
        )
    ]

    response = llm.invoke(prompt)

    return{
        'final_plan': response.text
    }

builder = StateGraph(State)
builder.add_node('destination_agent', destination_agent)
builder.add_node('food_agent', food_agent)
builder.add_node('budget_agent', budget_agent)
builder.add_node('planner agent', planner_agent)

builder.add_edge(START, 'destination_agent')
builder.add_edge(START, 'food_agent')
builder.add_edge(START, 'budget_agent')
builder.add_edge(['destination_agent', 'food_agent', 'budget_agent'], 'planner agent')
builder.add_edge('planner agent', END)

graph = builder.compile()

initial_state ={
    'user_request': "I want to go to Paris for 4 days. I'm interested in museums, good local food, and I have a medium budget.",
    "destination_info": "",
    "food_info": "",
    "budget_info": "",
    "final_plan": ""

}

result = graph.invoke(initial_state)

print(f"Destination info: {result['destination_info']}")
print("=" * 50)
print(f"Food info: {result['food_info']}")
print("=" * 50)
print(f"Budget info: {result['budget_info']}")
print("=" * 50)
print(f"Final plan: {result['final_plan']}")