import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 100  # Reduced for more concise responses
TEMPERATURE = 0.7

# System prompts
SYSTEM_PROMPT = """You are a protein expert. Provide extremely concise, focused answers.
Keep responses to 2-3 key points maximum. Be direct and precise."""

# Chat templates
CHAT_TEMPLATE = """Protein Context:
ID: {protein_id} ({symbols})
Location: {location}
Expression: {log2fc} (log2FC, p={pvalue})

Previous conversation:
{chat_history}

Give a highly focused answer to: {question}""" 