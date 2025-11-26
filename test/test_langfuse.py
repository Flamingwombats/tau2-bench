import os

from dotenv import load_dotenv
from langfuse.openai import openai

load_dotenv()

# Load environment variables
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# Configure langfuse.openai client to use Nebius API
# langfuse.openai wraps the OpenAI SDK and automatically traces calls
# Create a client instance with Nebius configuration
client = openai.OpenAI(
    api_key=NEBIUS_API_KEY,
    base_url="https://api.tokenfactory.nebius.com/v1/",
)

print("Testing langfuse with Nebius API...")
print("Model: openai/gpt-oss-120b")
print("Base URL: https://api.tokenfactory.nebius.com/v1/")
print()

# Make a completion call - langfuse will automatically trace this
completion = client.chat.completions.create(
    name="test-chat-nebius",
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": "You are a very accurate calculator. You output only the result of the calculation.",
        },
        {"role": "user", "content": "1 + 1 = "},
    ],
    metadata={"someMetadataKey": "someValue", "provider": "nebius"},
)

print("Response:")
print(completion.choices[0].message.content)
print()
print("✓ Langfuse tracing should be visible in your langfuse dashboard")
