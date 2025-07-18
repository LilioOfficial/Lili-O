# Overview

This project implements a real-time agent for multi-speaker environments based on a ASR - DIAR - Turn-taking pipeline. The agent is designed to handle a conversation with multiple speakers, allowing for real-time interaction and processing.

It's the genese of the project. There is still a lot to do, feel free to contribute!

## Architecture

You can find our architecture diagram in the [doc](https://github.com/LilioOfficial/realtime-multi-speaker-interaction/tree/master/doc) folder. The architecture is designed to handle multiple speakers in a meeting environment, allowing for real-time interaction and processing.


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
- [x] Implement ASR pipeline on RX subject
- [x] Implement the DIAR pipeline on RX subject => well it works but we need to improve the performance
- [ ] Create dataset for lili model training => see [Lili / Mili-O Hugging Face](https://huggingface.co/datasets/Lili-O/mili-o)
- [ ] Implement the Lili model using the Transformer architecture
- [ ] Implement the turn-taking model using Mili-O
- [ ] add it to the pipeline
- [ ] add the orchestrator to manage the different components of the pipeline + send the message to Big LLM
  
## License and Credits

- **Authors**: 
  - [Ludovic Maitre] : ludovik.maitre@gmail.com
  - [Léo Viguié] : leoviguie91@gmail.com

MIT License - 2025
