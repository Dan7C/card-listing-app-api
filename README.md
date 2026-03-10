# Card Listing App - API
AI-powered trading card listing generator. Processes card images through a tiered vision LLM pipeline to extract structured metadata, stores results in a Supabase PostgreSQL database, and generates e-commerce listings - all exposed via a FastAPI REST API designed for multiple client consumers. Desktop access via PySide6. Built with SQLAlchemy and Alembic.

## Related
- [Card Listing App - Desktop](https://github.com/dan7c/card-listing-app-desktop) — PySide6 desktop client

## Prerequisites
- Python 3.14+
- PostgreSQL client (for Alembic migrations)
- A GroqCloud account and API key
- A Supabase account and project

## Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and populate with your API keys
5. Copy `config.example/sets_config.example.json` to `config/sets_config.json`
6. Copy `prompts.example/` contents to `prompts/` and replace placeholders with your prompts

## Project Status
In development.