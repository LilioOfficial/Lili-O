# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import random
import os
import sys
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import torch
import asyncio
from moshi.client_utils import log
from moshi.models import loaders
from fastapi.middleware.cors import CORSMiddleware


# RxPY imports
import numpy as np
from serverState import OnlineServerState, OfflineServerState, MeetingServerState
from utils import DictWebSocketQueue

def seed_all(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
import json



def create_app(server_state):
    """Create and configure the FastAPI application"""
    app = FastAPI(title="Audio Transcription Server")
    app.state.frontend_clients = DictWebSocketQueue()

    ''' Configure CORS for the frontend extension, check the README for more info '''
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["chrome-extension://oidgmmmmdnkggpcolpbcckipoodendac"],  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket("/ws/frontend")
    async def frontend_ws(websocket: WebSocket, name: str = "frontend"):
        await websocket.accept()
        frontend_clients : DictWebSocketQueue = websocket.app.state.frontend_clients
        print("🌐 Frontend connected")
        handshake_response = {"title": "handshake", "content": "ready", "fullDescription": name}
        await websocket.send_text(json.dumps(handshake_response))
        frontend_clients.add_key(name)
        frontend_clients.add_websocket(name, websocket)
        try:
            while True:
                await websocket.receive_text()  # could be ping or noop
        except WebSocketDisconnect:
            print("❎ Frontend disconnected")
            frontend_clients.dequeue(name, websocket)
    

    @app.websocket("/api/chat")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for audio transcription"""
        await websocket.accept()
        frontend_clients : DictWebSocketQueue = websocket.app.state.frontend_clients
        frontend_clients.add_key(
            "moshi",
        )
        frontend_clients.add_websocket("moshi", websocket)
        # Use the server state's handle_chat method adapted for FastAPI
        await server_state.handle_chat_fastapi(websocket, key="moshi", clients=frontend_clients )
    

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, url: str = None):
        """WebSocket endpoint for audio transcription"""
        await websocket.accept()
        frontend_clients = websocket.app.state.frontend_clients
        frontend_clients.add_key(url)
        # Use the server state's handle_chat method adapted for FastAPI
        await server_state.handle_chat_fastapi(websocket, key=url, clients=frontend_clients)

    
    @app.get("/api/prompt")
    def websocket_endpoint(url: str = None):
        """WebSocket endpoint for audio transcription"""
        print(f"Getting prompt for URL: {url}")
        abc =  server_state.get_prompt(url)
        if abc is None:
            return {"error": "No prompt found for the given URL"}
        return {"prompt": abc}


    @app.get("/")
    async def read_root():
        """Serve the main HTML page"""
        static_path = getattr(app.state, 'static_path', 'static')
        return FileResponse(os.path.join(static_path, "index.html"))
    
    return app


def create_server_state(mode: str, mimi, text_tokenizer, lm, audio_delay_seconds, 
                       padding_token_id, audio_silence_prefix_seconds, device):
    """Factory function to create appropriate server state based on mode"""
    if mode == "online":
        return OnlineServerState(
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            audio_delay_seconds=audio_delay_seconds,
            padding_token_id=padding_token_id,
            audio_silence_prefix_seconds=audio_silence_prefix_seconds,
            device=device
        )
    elif mode == "offline":
        return OfflineServerState(
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            audio_delay_seconds=audio_delay_seconds,
            padding_token_id=padding_token_id,
            audio_silence_prefix_seconds=audio_silence_prefix_seconds,
            device=device
        )
    elif mode == "meeting":
        return MeetingServerState(
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            audio_delay_seconds=audio_delay_seconds,
            padding_token_id=padding_token_id,
            audio_silence_prefix_seconds=audio_silence_prefix_seconds,
            device=device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


async def run_offline_mode(args, server_state):
    """Run in offline mode - process audio file"""
    log("info", f"Starting offline transcription of: {args.input_file}")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        log("error", f"Input file does not exist: {args.input_file}")
        return False
    
    # Process the audio file
    transcription = await server_state.process_audio_file_async(args.input_file, args.output)
    
    if transcription:
        log("info", "Transcription completed successfully")
        if not args.output:
            print("\n" + "="*50)
            print("TRANSCRIPTION:")
            print("="*50)
            print(transcription)
            print("="*50)
        return True
    else:
        log("error", "Transcription failed")
        return False


async def run_online_mode(args, server_state):
    """Run in online mode - start WebSocket server"""
    
    # Create FastAPI app
    app = create_app(server_state)
    
    # Configure static files
    static_path = args.static
    app.state.static_path = static_path
    
    # Mount static files
    print(f"Static path: {static_path}")
    if os.path.exists(static_path):
        app.mount("/assets", StaticFiles(directory=static_path + "/assets", check_dir=True), name="static")
        log("info", f"serving static content from {static_path}")
    else:
        log("warning", f"static directory {static_path} does not exist")
    
    # Configure SSL
    ssl_keyfile = args.ssl_keyfile
    ssl_certfile = args.ssl_certfile
    
    if ssl_keyfile and ssl_certfile:
        protocol = "https"
        log("info", f"SSL enabled with keyfile: {ssl_keyfile}, certfile: {ssl_certfile}")
    else:
        protocol = "http"
        ssl_keyfile = None
        ssl_certfile = None

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    
    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": "info",
        "access_log": True,
        "reload": True,
    }
    
    # Add SSL configuration if provided
    if ssl_keyfile and ssl_certfile:
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile,
        })
    
    # Run the server
    config = uvicorn.Config(**uvicorn_config)
    server = uvicorn.Server(config)
    log("info", "Starting online server...")
    await server.serve()


async def main():
    parser = argparse.ArgumentParser(description="Online/Offline Audio Transcription Server")
    
    # Mode selection
    parser.add_argument("--mode", choices=["online", "offline", "meeting"], default="online",
                        help="Run in online (WebSocket server) or offline (file processing) mode")
    
    # Offline mode arguments
    parser.add_argument("--input-file", type=str, help="Input audio file for offline mode")
    parser.add_argument("--output", "-o", type=str, help="Output file for transcription (offline mode)")
    
    # Online mode arguments
    parser.add_argument("--host", default="localhost", type=str, help="Host for online mode")
    parser.add_argument("--port", default=8998, type=int, help="Port for online mode")
    parser.add_argument("--static", type=str, default="static", help="Static files directory")
    parser.add_argument("--ssl-keyfile", type=str, help="Path to SSL key file for HTTPS")
    parser.add_argument("--ssl-certfile", type=str, help="Path to SSL certificate file for HTTPS")
    
    # Model arguments
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi")
    parser.add_argument("--hf-repo", type=str, default="kyutai/stt-1b-en_fr",
                        help="HF repo to look into, defaults to Moshiko")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "offline":
        if not args.input_file:
            log("error", "Input file is required for offline mode")
            sys.exit(1)
        
        # Check if librosa is available for offline mode
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            log("error", "librosa and soundfile are required for offline mode. Install with: pip install librosa soundfile")
            sys.exit(1)
    
    elif args.mode == "online":
        # Check if sphn is available for online mode
        try:
            import sphn
        except ImportError:
            log("error", "sphn is required for online mode. Make sure it's installed.")
            sys.exit(1)
    
    # Set random seed
    seed_all(42424242)
    
    # Load model
    log("info", "Retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
    )
    
    log("info", "Loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "Mimi loaded")
    
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    
    log("info", "Loading STT model")
    lm = checkpoint_info.get_moshi(device=args.device)
    log("info", "STT model loaded")
    
    # Create appropriate server state based on mode
    server_state = create_server_state(
        mode=args.mode,
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        audio_delay_seconds=checkpoint_info.stt_config.get("audio_delay_seconds", 5.0),
        padding_token_id=checkpoint_info.raw_config.get("text_padding_token_id", 3),
        audio_silence_prefix_seconds=checkpoint_info.stt_config.get("audio_silence_prefix_seconds", 1),
        device=args.device
    )
    
    log("info", f"Created {args.mode} server state: {type(server_state).__name__}")
    
    log("info", "Warming up the model")
    server_state.warmup()
    
    # Run appropriate mode
    if args.mode == "offline":
        success = await run_offline_mode(args, server_state)
        if not success:
            sys.exit(1)
    else:
        await run_online_mode(args, server_state)


if __name__ == "__main__":
    with torch.no_grad():
        asyncio.run(main())