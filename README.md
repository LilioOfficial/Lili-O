# Overview
# Lili-O Project

This project implements a real-time agent for multi-speaker environments based on an architecture we developed.

## Architecture

*(Add a brief description of the architecture here)*

## Use Cases

*(List the use cases for the project here)*

## Quickstart / Installation

To get started, create a virtual environment and install the required packages by running the following commands:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

## Run 

### Server Mode

To run the server, use the following command:

```bash
python main.py --mode online --port XXXX --host XXX.X.X.X
```

Note: We used Mimi and did not focus on design, so we utilized the already implemented Moshi web interface. The server stops after 10 seconds of inactivity.

### Meeting Mode (Join a Meeting)

1. First, change the meeting URL and WebSocket URL in `curl.sh`.
2. Then run the following commands:

```bash
bash curl.sh

python main.py --mode meeting --port XXXX --host XXX.X.X.X
```

## Roadmap

- [ ] Implement the turn-taking model using Mili-O 

## License and Credits

- **Authors**: 
  - [Ludovic Maitre]
  - [Léo Viguié]

MIT License - 2025
