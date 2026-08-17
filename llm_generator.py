import os

from groq import Groq
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )


client = Groq(
    api_key=api_key
)


def generate_answer(
    prompt: str
) -> str:

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict 3GPP technical "
                    "assistant. Answer only from "
                    "the supplied evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        max_tokens=1000
    )

    return response.choices[0].message.content.strip()