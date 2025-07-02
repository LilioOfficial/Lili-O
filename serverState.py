
import asyncio
from dataclasses import dataclass
import aiohttp
from aiohttp import web
import numpy as np
import sentencepiece
import sphn
import torch
from moshi.models import MimiModel, LMModel, LMGen
import math

# RxPY imports
import rx
from rx.subject import Subject
from rx.scheduler.eventloop import AsyncIOScheduler
import time
from moshi.client_utils import log

# Custom imports

import numpy as np
from pipelines import ASRPipeline, DiarPipeline
from utils import AudioChunk

@dataclass
class ServerState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds, padding_token_id, audio_silence_prefix_seconds, device: str | torch.device):
        # self.mimi = mimi
        # self.text_tokenizer = text_tokenizer
        # self.lm_gen = LMGen(lm, temp=0, temp_text=0)

        # self.device = device
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)
        self.sample_rate = mimi.sample_rate

        # self.lock = asyncio.Lock()
        # self.audio_delay_seconds = audio_delay_seconds
        # self.padding_token_id = padding_token_id
        # self.mimi.streaming_forever(1)
        # self.lm_gen.streaming_forever(1)
        # self.audio_silence_prefix_seconds = audio_silence_prefix_seconds
        # self.n_prefix_chunks = math.ceil(self.audio_silence_prefix_seconds * mimi.frame_rate)
        
        # RxPY subjects for streaming
        self.audio_subject = Subject()
        self.lock = asyncio.Lock()
        self.audioPipeline = ASRPipeline(
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            audio_delay_seconds=audio_delay_seconds,
            padding_token_id=padding_token_id,
            audio_silence_prefix_seconds=audio_silence_prefix_seconds,
            device=device
        )

        self.diarization_subject = Subject()
        self.diarizationPipeline = DiarPipeline()
        self.schedulerASR = AsyncIOScheduler(asyncio.get_event_loop())
        self.schedulerDiar = AsyncIOScheduler(asyncio.get_event_loop())

    
      

    def warmup(self):
        self.audioPipeline.warmup()

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        event = asyncio.get_running_loop()

        async def recv_loop():
            nonlocal close, opus_reader
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        log("info", "connection closed by client")
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        log("error", f"unexpected message type {message.type}")
                        continue
                    
                    message_data = message.data
                    if not isinstance(message_data, bytes):
                        log("error", f"unsupported message type {type(message_data)}")
                        continue
                    if len(message_data) == 0:
                        log("warning", "empty message")
                        continue
                    
                    kind = message_data[0]
                    if kind == 1:  # audio
                        payload = message_data[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                log("info", "recv_loop closed")

        async def opus_processing_loop():
            """Opus processing in a separate thread to feed the RxPY pipeline"""
            all_pcm_data = None
            
            while not close:
                try :
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    pcm = opus_reader.read_pcm()
                    if pcm.shape[-1] == 0:
                        continue
                        
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk = all_pcm_data[:self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        # Push audio chunk to RxPY pipeline
                        audio_chunk = AudioChunk(
                            data=chunk,
                            timestamp=time.time()
                        )
                        if audio_chunk != None:
                            self.audio_subject.on_next(audio_chunk)
                            vad_threshold = 1e-2  # You can adjust this
                            energy = np.sqrt(np.mean(chunk ** 2))

                            if energy > vad_threshold:
                                self.diarization_subject.on_next(chunk.reshape(1, -1))

                            
                except Exception as e:
                    if not close:
                        log("error", f"Error in opus processing: {e}")
                    break

        log("info", "accepted connection")
        close = False

        async with self.lock:
            opus_reader = sphn.OpusStreamReader(self.sample_rate)
            self.audioPipeline.reset()
            
            # # ⏺ Créer et souscrire au pipeline de transcription (LM)
            pipeline = self.audioPipeline.create_pipeline(
                ws=ws,
                loop=event,
                subject= self.audio_subject,
            )

            subscription = pipeline.subscribe(
                on_next=lambda task: log("info", f"pipeline done"),
                on_error=lambda e: log("error", f"Pipeline subscription error: {e}"),
                on_completed=lambda: log("info", "Pipeline completed"),
                scheduler=self.schedulerASR
            )

            # ⏺ Créer et souscrire au pipeline de diarisation
            diarization_pipeline = self.diarizationPipeline.create_pipeline(ws, loop=event, subject=self.diarization_subject)
            diarization_subscription = diarization_pipeline.subscribe(
                on_next=lambda task: log("info", f"diarization done"),
                on_error=lambda e: log("error", f"Diarization subscription error: {e}"),
                on_completed=lambda: log("info", "Diarization completed"),
                scheduler=self.schedulerDiar
            )


            # 🤝 Envoyer le handshake
            await ws.send_bytes(b"\x00")

            try:
                # Lancer les deux boucles : réception WebSocket + traitement Opus
                await asyncio.gather(opus_processing_loop(), recv_loop())
            finally:
                close = True
                subscription.dispose()
                diarization_subscription.dispose()
            return ws

