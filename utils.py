from huggingface_hub import WebhookInfo
from regex import D
import torch
from dataclasses import dataclass
import numpy as np
import torch
from abc import ABC
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import websocket

@dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]

    def __str__(self):
        return f"{self.text}"


@dataclass
class AudioChunk:
    data: np.ndarray
    timestamp: float


@dataclass
class ProcessedTokens:
    tokens: torch.Tensor
    codes: torch.Tensor
    timestamp: float

def tokens_to_timestamped_text(
    text_tokens,
    tokenizer,
    frame_rate,
    end_of_padding_id,
    padding_token_id,
    offset_seconds,
) -> list[TimestampedText]:
    text_tokens = text_tokens.cpu().view(-1)

    sequence_timestamps = []

    def _tstmp(start_position, end_position):
        return (
            max(0, start_position / frame_rate - offset_seconds),
            max(0, end_position / frame_rate - offset_seconds),
        )

    def _decode(t):
        t = t[t > padding_token_id]
        return tokenizer.decode(t.numpy().tolist())

    def _decode_segment(start, end):
        nonlocal text_tokens
        nonlocal sequence_timestamps
        if (type(end) is torch.Tensor):
            end = end.type(torch.int64).item()
        text = _decode(text_tokens[start:end])
        words_inside_segment = text.split()
        
        if len(words_inside_segment) == 0:
            return
        if len(words_inside_segment) == 1:
            sequence_timestamps.append(
                TimestampedText(text=text, timestamp=_tstmp(start, end))
            )
        else:
            for adjacent_word in words_inside_segment[:-1]:
                n_tokens = len(tokenizer.encode(adjacent_word))
                sequence_timestamps.append(
                    TimestampedText(
                        text=adjacent_word, timestamp=_tstmp(start, start + n_tokens)
                    )
                )
                start += n_tokens

            adjacent_word = words_inside_segment[-1]
            sequence_timestamps.append(
                TimestampedText(text=adjacent_word, timestamp=_tstmp(start, end))
            )

    (segment_boundaries,) = torch.where(text_tokens == end_of_padding_id)

    if not segment_boundaries.numel():
        return []

    for i in range(len(segment_boundaries) - 1):
        segment_start = int(segment_boundaries[i]) + 1
        segment_end = int(segment_boundaries[i + 1])
        _decode_segment(segment_start, segment_end)

    last_segment_start = segment_boundaries[-1] + 1
    boundary_token = torch.tensor([tokenizer.eos_id()])
    (end_of_last_segment,) = torch.where(
        torch.isin(text_tokens[last_segment_start:], boundary_token)
    )

    if not end_of_last_segment.numel():
        last_segment_end = min(text_tokens.shape[-1], last_segment_start + frame_rate)
    else:
        last_segment_end = last_segment_start + end_of_last_segment[0]
    _decode_segment(last_segment_start, last_segment_end)
    return sequence_timestamps

import asyncio
from typing import Callable, Awaitable, Any, Dict
from fastapi import WebSocket as websocket
from typing import List
import json


@dataclass
class WebSocketHistory:
    key: str
    webSocket: List[websocket]
    history: List

    def __init__(self, key: str):
        self.key = key
        self.webSocket = []
        self.history = []

    def add_message(self, message: str):
        self.history.append(message)
    
    async def send_message(self, message: str):
        if not self.webSocket:
            print(f"No WebSocket connections for key: {self.key}")
            return

        for ws in self.webSocket:
            asyncio.create_task(ws.send_text(message))

    def add_webSocket(self, webSocket: websocket):
        if webSocket not in self.webSocket:
            self.webSocket.append(webSocket)
        else:
            print(f"WebSocket already exists in the history for key: {self.key}")

    def get_history(self) -> List:
        return self.history
    
    def remove_webSocket(self, webSocket: websocket):
        if webSocket in self.webSocket:
            self.webSocket.remove(webSocket)
        else:
            print(f"WebSocket not found in the history for key: {self.key}")


class DictWebSocketQueue:
    queues: Dict[str, WebSocketHistory]

    def __init__(self, key: str = None):
        self.queues: Dict[str, WebSocketHistory] = {key: WebSocketHistory(key)} if key else {}

    def add_key(self, key: str):
        if key not in self.queues:
            self.queues[key] = WebSocketHistory(key)
        else:
            print(f"Key {key} already exists in the queue.")
    
    async def send_message(self, key: str, message : dict):
        if key in self.queues:
            history = self.queues[key]
            await history.send_message(json.dumps(message))
        else:
            print(f"Key {key} not found in the queue.")

    def add_websocket(self, key: str, webSocket: websocket):
        if key not in self.queues:
            self.add_key(key)
        self.queues[key].add_webSocket(webSocket)

    def remove_websocket(self, key: str, webSocket: websocket):
        if key in self.queues:
            self.queues[key].remove_webSocket(webSocket)
        else:
            print(f"Key {key} not found in the queue.")

    def get_history (self, key: str) -> List:
        if key in self.queues:
            return self.queues[key].get_history()
        else:
            print(f"Key {key} not found in the queue.")
            return []
    
    

@dataclass
class FactCheckingModel() : 
    def __init__(self, model_name):
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",           # Recommended for LLMs
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )


        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )

    
    def call(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        # print("thinking content:", thinking_content)
        return content
    

def build_message_to_server(title : str, content : str, priority:str, fullDescription: str):
    return {
        "title": title,
        "content": content,
        "priority": priority,
        "fullDescription": fullDescription
    }