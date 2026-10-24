from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL= 'gemini-3.5-flash-lite'
llm = ChatGoogleGenerativeAI(model = MODEL)
MAX_ATTEMPTS = 3

class State(TypedDict):
    ingredients: str
    recipe: str
    feedback: str
    approved: bool
    attempts : int

class Review(TypedDict):
    feedback: str
    approved: bool

checker_llm = llm.with_structured_output(Review)


def generate_recipe(state: State):
    prompt = (
        'create a recipe using this ingredients:'
        f'{state["ingredients"]}\n'
        'include ingredient quantities, preparation time and step by step instruction'
    )
    response = llm.invoke(prompt)
    return {'recipe': response.content, 'attempts': 1}
def check_recipe(state: State):
    prompt = (
        'you are a strict food critic.\n'
        f'Required ingredients: {state["ingredients"]}.\n'
        f'Provided recipe: {state["recipe"]}.\n'
        'check that:\n'
        'all the required ingredients are used\n'
        'the recipe is easy and clear to understand\n'
        'it has preparation steps\n'
        'the ingredient quantities are logical'
        "Return 'approved' as true only if every point is satisfied, otherwise false. "
        "Return short 'feedback' explaining your decision."

    )

    review = checker_llm.invoke(prompt)
    return{'feedback': review['feedback'], 'approved': review['approved']}

def route_after_check(state: State):
    if state['approved']:
        return END
    if state['attempts'] >= MAX_ATTEMPTS:
        return END
    return 'improve'

def improve(state:State):
    prompt =(
        'improve the recipe below based on the feedback.\n'
        f'ingredients: {state["ingredients"]}\n'
        f'current recipe: {state["recipe"]}\n'
        f'feedback: {state["feedback"]}\n'
        'return only the full improved recipe.'
    )
    response = llm.invoke(prompt)
    return {'recipe': response.content, 'attempts': state['attempts'] +1}

graph_builder = StateGraph(State)

graph_builder.add_node('generate recipe', generate_recipe)
graph_builder.add_node('check recipe', check_recipe)
graph_builder.add_node('route after check', route_after_check)
graph_builder.add_node('improve', improve)

graph_builder.add_edge(START, 'generate recipe')
graph_builder.add_edge('generate recipe', 'check recipe')
graph_builder.add_conditional_edges (
    'check recipe',
    route_after_check,
    {'improve': 'improve', END: END},
    )
graph_builder.add_edge('improve', 'check recipe')

graph = graph_builder.compile()

initial_state = {
    'ingredients': 'chicken, flour, egg, tomato, butter',
    'recipe': '',
    'feedback': '',
    'approved': False,
    'attempts': 0,
}

result = graph.invoke(initial_state)

print('RECIPE\n')
print(result['recipe'])
print('FEEDBACK\n')
print(result['feedback'])

