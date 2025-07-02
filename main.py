# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass
import inspect
import random
import os
from pathlib import Path
import tarfile
import secrets
import sys
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from moshi.client_utils import log
from moshi.models import loaders

# RxPY imports
from pathlib import Path
import numpy as np
from serverState import ServerState
 

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str, default="static")

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default="kyutai/stt-1b-en_fr",
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
  
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")

    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
  

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
       )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading stt")
    lm = checkpoint_info.get_moshi(device=args.device)
    log("info", "moshi stt")

    state = ServerState(
        # checkpoint_info.model_type, 
        mimi, 
        text_tokenizer, 
        lm, 
        checkpoint_info.stt_config.get("audio_delay_seconds", 5.0), 
        checkpoint_info.raw_config.get("text_padding_token_id", 3), 
        checkpoint_info.stt_config.get("audio_silence_prefix_seconds", 1), 
        args.device
    )
    
    log("info", "warming up the model")
    state.warmup()
    
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    
    static_path = args.static
    async def handle_root(_):
        return web.FileResponse(os.path.join(static_path, "index.html"))

    log("info", f"serving static content from {static_path}")
    app.router.add_get("/", handle_root)
    app.router.add_static(
        "/", path=static_path, follow_symlinks=True, name="static"
    )
    
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
   
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()