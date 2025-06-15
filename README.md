# Whisper RAG

A Retrieval-Augmented Generation (RAG) implementation using Whisper.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- Python 3.8 or higher
- Poetry

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/whisper_rag.git
   cd whisper_rag
   ```

2. Install dependencies with Poetry:
   ```
   poetry install
   ```

3. Activate the virtual environment:
   ```
   poetry shell
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file with your API keys and configuration.

## Usage

Run the main script:
```
poetry run python main.py
```

## Development

Run tests:
```
poetry run pytest
```
