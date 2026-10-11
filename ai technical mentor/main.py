from google import genai
import os
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
system_prompt = ('You are an experienced senior software engineer and programming mentor. Explain technical concepts clearly, adapt your explanations to the users level, and use practical examples when appropriate.')
history = []
print("AI Technical Mentor enter history to see conversation, enter 'exit' to quit.")

while True:
    user_input = input('YOU :')
    if not user_input:
        continue
    if user_input.lower() == 'exit':
        print('\n Goodbye')
        break
    if user_input.lower() == 'history':
        print("\nConversation History:")
        for message in history:
            if message['role'] == 'user':
                print("USER:")
            else:
                print("ASSISTANT:")
            print(message["parts"][0]["text"])
            print()
        continue

    history.append({"role": "user", "parts": [{"text": user_input}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )

    history.append({"role": "model", "parts": [{"text": response.text}]})

    print("\nAI:", response.text, "\n")