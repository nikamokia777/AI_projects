from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
MODEL = 'gemini-2.5-flash'
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class ProductReview(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field( 
        description="The overall sentiment of the product review."
        )
    pros: list[str] = Field(
        description="A list of positive aspects or advantages of the product."
        )
    cons: list[str] = Field(
        description="A list of negative aspects or disadvantages of the product."
        )
    summary : str = Field(
        description="A brief summary of the product review."
        )
    recommend: bool = Field(
        description="Whether the reviewer recommends the product."
        )

parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant that analyzes product reviews. "
        "Analyze the review provided by the user and return the result "
        "strictly in the requested format.\n\n{format_instructions}",
    ),
    ("human", "Analyze this product review:\n\n{review}"),
]).partial(format_instructions=parser.get_format_instructions())

model = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
chain = prompt | model | parser
tests = [
    "This laptop is fast, lightweight, and has an excellent display. However, its battery life could be better.",
    "The phone is slow, overheats frequently, and the camera quality is disappointing.",
    "The headphones have average sound quality. They are comfortable, but the price is relatively high.",
]
for i, review in enumerate(tests, start=1):
    result = chain.invoke({"review": review})
    print(f"--- TEST {i} ---")
    print(result)
    print(type(result))



