import asyncio
from abc import ABC, abstractmethod
from calendar import c
from dataclasses import dataclass
from pydoc import cli
from typing import Optional, List, Any
import json
import librosa
import numpy as np
import sentencepiece
import sphn
import torch
from moshi.models import MimiModel, LMModel, LMGen
from fastapi import WebSocket, WebSocketDisconnect

# RxPY imports
import rx
from rx.subject import Subject
from rx.scheduler.eventloop import AsyncIOScheduler
import time
from moshi.client_utils import log

# Custom imports
import numpy as np
from pipelines import ASRPipeline
from utils import AudioChunk, MapQueue
import base64
from scipy import signal


@dataclass
class BaseServerState(ABC):
    """Abstract base class for server state implementations"""
    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds: float, padding_token_id: int, 
                 audio_silence_prefix_seconds: float, device: str | torch.device):
        
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm = lm
        self.device = device
        
        # Common audio settings
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)
        self.sample_rate = mimi.sample_rate
        
        # RxPY subjects for streaming
        self.audio_subject = Subject()
        self.lock = asyncio.Lock()
        
        # Initialize audio pipeline
        self.audioPipeline = ASRPipeline(
            model_name="Qwen/Qwen3-8B",
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            lm=lm,
            audio_delay_seconds=audio_delay_seconds,
            padding_token_id=padding_token_id,
            audio_silence_prefix_seconds=audio_silence_prefix_seconds,
            device=device
        )
        
        # Common scheduler
        self.scheduler = AsyncIOScheduler(asyncio.get_event_loop())
        
        # Initialize subclass-specific components
        self._init_subclass_components()
    
    @abstractmethod
    def _init_subclass_components(self):
        """Initialize subclass-specific components"""
        pass
    
    def warmup(self):
        """Warmup the model"""
        self.audioPipeline.warmup()
    
    def _create_audio_chunk(self, data: np.ndarray, timestamp: float) -> AudioChunk:
        """Create an AudioChunk object"""
        return AudioChunk(data=data, timestamp=timestamp)
    
    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate energy for VAD purposes"""
        return np.sqrt(np.mean(chunk ** 2))
    
    def _create_pipeline_subscription(self, pipeline) -> Any:
        """Create and return a pipeline subscription"""
        return pipeline.subscribe(
            on_next=lambda task: log("info", f"pipeline done"),
            on_error=lambda e: log("error", f"Pipeline subscription error: {e}"),
            on_completed=lambda: log("info", "Pipeline completed"),
            scheduler=self.scheduler
        )
    
    @abstractmethod
    async def handle_chat_fastapi(self, websocket: WebSocket, clients: MapQueue):
        """Handle FastAPI WebSocket connections - must be implemented by subclasses"""
        pass


@dataclass
class OnlineServerState(BaseServerState):
    """Server state for real-time online processing with binary audio data"""
    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds: float, padding_token_id: int, 
                 audio_silence_prefix_seconds: float, device: str | torch.device):
        super().__init__(mimi, text_tokenizer, lm, audio_delay_seconds, padding_token_id, 
                         audio_silence_prefix_seconds, device)
        
    def _init_subclass_components(self):
        """Initialize components specific to online processing"""
        self.schedulerASR = AsyncIOScheduler(asyncio.get_event_loop())
        self.diarization_subject = Subject()
    
    async def handle_chat_fastapi(self, websocket: WebSocket, key :str,  clients: MapQueue):
        """FastAPI WebSocket handler for binary audio data"""
        event_loop = asyncio.get_running_loop()
        close = False
        opus_reader = None

        async def recv_loop():
            nonlocal close, opus_reader
            try:
                while True:
                    try:
                        # Receive binary data from WebSocket
                        message_data = await websocket.receive_bytes()
                        if len(message_data) == 0:
                            log("warning", "empty message")
                            continue
                        
                        kind = message_data[0]
                        if kind == 1:  # audio
                            payload = message_data[1:]
                            opus_reader.append_bytes(payload)
                        else:
                            log("warning", f"unknown message kind {kind}")
                            
                    except WebSocketDisconnect:
                        log("info", "WebSocket disconnected by client")
                        break
                    except Exception as e:
                        log("error", f"Error in recv_loop: {e}")
                        break
            finally:
                close = True
                log("info", "recv_loop closed")

        async def opus_processing_loop():
            """Opus processing in a separate thread to feed the RxPY pipeline"""
            all_pcm_data = None
            
            while not close:
                try:
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
                        
                        # Create and send audio chunk
                        audio_chunk = self._create_audio_chunk(chunk, time.time())
                        if audio_chunk is not None:
                            self.audio_subject.on_next(audio_chunk)
                            
                            # Optional VAD calculation
                            energy = self._calculate_energy(chunk)
                            
                except Exception as e:
                    if not close:
                        log("error", f"Error in opus processing: {e}")
                    break

        log("info", "accepted FastAPI WebSocket connection")

        async with self.lock:
            opus_reader = sphn.OpusStreamReader(self.sample_rate)
            self.audioPipeline.reset()
            
            # Create and subscribe to transcription pipeline
            pipeline = self.audioPipeline.create_pipeline(
                key=key,
                mapQueue=clients,
                loop=event_loop,
                subject=self.audio_subject,
            )

            subscription = self._create_pipeline_subscription(pipeline)

            # Send handshake
            try:
                await websocket.send_bytes(b"\x00")
            except Exception as e:
                log("error", f"Error sending handshake: {e}")
                return

            try:
                # Launch both loops: WebSocket reception + Opus processing
                await asyncio.gather(opus_processing_loop(), recv_loop())
            except Exception as e:
                log("error", f"Error in main loops: {e}")
            finally:
                close = True
                subscription.dispose()
                try:
                    await websocket.close()
                except:
                    pass  # WebSocket might already be closed


@dataclass
class OfflineServerState(BaseServerState):
    """Server state for offline file-based processing"""
    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds: float, padding_token_id: int, 
                 audio_silence_prefix_seconds: float, device: str | torch.device):
        super().__init__(mimi, text_tokenizer, lm, audio_delay_seconds, padding_token_id, 
                         audio_silence_prefix_seconds, device)
    
    def _init_subclass_components(self):
        """Initialize components specific to offline processing"""
        self.second = 0
        
    def warmup(self):
        """Warmup the model"""
        self.audioPipeline.warmup()
        
    async def handle_chat_fastapi(self, websocket: WebSocket, clients: Optional[MapQueue] = None):
        """Not applicable for offline processing"""
        raise NotImplementedError("Offline processing doesn't support WebSocket connections")
        
    async def process_audio_file_async(self, audio_file_path: str, output_file: Optional[str] = None):
        """Process an audio file asynchronously and return transcription"""
        
        log("info", f"Loading audio file: {audio_file_path}")
        
        # Load audio file
        try:
            audio_data, original_sr = librosa.load(audio_file_path, sr=None, mono=True)
            log("info", f"Audio loaded: {len(audio_data)} samples at {original_sr} Hz")
        except Exception as e:
            log("error", f"Failed to load audio file: {e}")
            return None
            
        # Resample if necessary
        if original_sr != self.sample_rate:
            log("info", f"Resampling from {original_sr} Hz to {self.sample_rate} Hz")
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
        
        # Ensure audio is in the right format (float32)
        audio_data = audio_data.astype(np.float32)
        
        results = []
        
        async with self.lock:
            self.audioPipeline.reset()
            
            # Create pipeline for offline processing
            pipeline = self.audioPipeline.create_offline_pipeline(
                subject=self.audio_subject,
                results_collector=results
            )
            
            # Subscribe to pipeline
            subscription = self._create_pipeline_subscription(pipeline)
            
            try:
                # Process audio in chunks
                await self._feed_audio_chunks(audio_data)
                
                # Wait for processing to complete
                await asyncio.sleep(10.0)  # Allow final processing
                
                # Signal completion
                self.audio_subject.on_completed()
                
                # Wait a bit more for final results
                await asyncio.sleep(1.0)
                
            except Exception as e:
                log("error", f"Error processing audio: {e}")
            finally:
                subscription.dispose()
        
        # Compile results
        transcription = self._compile_results(results)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in transcription:
                        json_line = json.dumps(item, ensure_ascii=False)
                        f.write(json_line + '\n')
                log("info", f"Transcription saved to: {output_file}")
            except Exception as e:
                log("error", f"Failed to save transcription: {e}")
        
        return transcription
    
    async def _feed_audio_chunks(self, audio_data: np.ndarray):
        """Feed audio data to the pipeline in chunks"""
        
        total_samples = len(audio_data)
        processed_samples = 0
        
        log("info", f"Processing {total_samples} samples in chunks of {self.frame_size}")
        
        while processed_samples < total_samples:
            # Get next chunk
            end_sample = min(processed_samples + self.frame_size, total_samples)
            chunk = audio_data[processed_samples:end_sample]
            
            # Pad if necessary
            if len(chunk) < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - len(chunk)), mode='constant')
            
            # Create and send audio chunk
            audio_chunk = self._create_audio_chunk(chunk, processed_samples / self.sample_rate)
            self.audio_subject.on_next(audio_chunk)
            
            processed_samples = end_sample
            
            # Add small delay to simulate real-time processing
            await asyncio.sleep(0.01)
            
            # Log progress
            if processed_samples % (self.frame_size * 100) == 0:
                progress = (processed_samples / total_samples) * 100
                log("info", f"Processing progress: {progress:.1f}%")
    
    def _compile_results(self, results):
        """Compile results into final transcription"""
        if not results:
            return "No transcription generated."
        
        # This depends on your pipeline's output format
        # Adjust based on what your ASRPipeline actually returns
        print(f"Compiling results: {results}")
        return results


@dataclass
class MeetingServerState(BaseServerState):
    """Server state for meeting/conference processing with JSON audio messages"""
    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds: float, padding_token_id: int, 
                 audio_silence_prefix_seconds: float, device: str | torch.device):
        super().__init__(mimi, text_tokenizer, lm, audio_delay_seconds, padding_token_id, 
                         audio_silence_prefix_seconds, device)
    
    def _init_subclass_components(self):
        """Initialize components specific to meeting processing"""
        self.schedulerASR = AsyncIOScheduler(asyncio.get_event_loop())
        self.diarization_subject = Subject()
        self.pcm_buffer = np.array([], dtype=np.float32)

    async def handle_chat_fastapi(self, websocket: WebSocket, clients: MapQueue):
        """FastAPI WebSocket handler for JSON audio messages"""
        event_loop = asyncio.get_running_loop()
        close = False

        async def recv_loop():
            nonlocal close
            try:
                while True:
                    try:
                        # Receive JSON data from WebSocket
                        message_data = await websocket.receive_text()
                        if not message_data:
                            log("warning", "empty message")
                            continue
                        
                        await self._process_json_message(message_data)
                            
                    except WebSocketDisconnect:
                        log("info", "WebSocket disconnected by client")
                        break
                    except Exception as e:
                        log("error", f"Error in recv_loop: {e}")
                        break
            finally:
                close = True
                log("info", "recv_loop closed")

        log("info", "accepted FastAPI WebSocket connection")

        async with self.lock:
            # Initialize PCM buffer for proper frame size handling
            self.pcm_buffer = np.array([], dtype=np.float32)
            self.audioPipeline.reset()
            
            # Create and subscribe to transcription pipeline
            pipeline = self.audioPipeline.create_pipeline(
                mapQueue=clients,
                loop=event_loop,
                subject=self.audio_subject,
            )

            subscription = self._create_pipeline_subscription(pipeline)

            # Send handshake (JSON format)
            try:
                handshake_response = {"event": "handshake", "status": "ready"}
                await websocket.send_text(json.dumps(handshake_response))
            except Exception as e:
                log("error", f"Error sending handshake: {e}")
                return

            try:
                # Only need the WebSocket reception loop now
                await recv_loop()
            except Exception as e:
                log("error", f"Error in recv_loop: {e}")
            finally:
                close = True
                subscription.dispose()
                try:
                    await websocket.close()
                except:
                    pass  # WebSocket might already be closed
    
    async def _process_json_message(self, message_data: str):
        """Process incoming JSON messages"""
        try:
            # Parse JSON message
            json_message = json.loads(message_data)
            
            # Check if this is an audio event
            if json_message.get("event") == "audio_mixed_raw.data":
                await self._process_audio_data(json_message)
            else:
                log("info", f"Received non-audio event: {json_message.get('event')}")
                
        except json.JSONDecodeError as e:
            log("error", f"Error parsing JSON message: {e}")
    
    async def _process_audio_data(self, json_message: dict):
        """Process audio data from JSON message"""
        data = json_message.get("data", {})
        audio_data = data.get("data", {})
        
        # Extract base64 encoded audio buffer
        buffer_b64 = audio_data.get("buffer")
        if not buffer_b64:
            log("warning", "no audio buffer in message")
            return
        
        # Decode and process audio
        try:
            pcm_bytes = base64.b64decode(buffer_b64)
            # Convert bytes to numpy array (16-bit signed little-endian)
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            # Convert to float32 and normalize to [-1, 1] range
            pcm_float = pcm_array.astype(np.float32) / 32768.0
            
            # Resample from 16kHz to 24kHz for Mimi
            pcm_24khz = self._resample_audio(pcm_float)
            
            # Extract timestamp if available
            timestamp_info = audio_data.get("timestamp", {})
            base_timestamp = timestamp_info.get("relative", time.time())
            
            # Process audio chunks
            await self._process_resampled_audio(pcm_24khz, base_timestamp)
            
        except Exception as e:
            log("error", f"Error decoding base64 audio: {e}")
    
    def _resample_audio(self, pcm_float: np.ndarray) -> np.ndarray:
        """Resample audio from 16kHz to 24kHz"""
        # Calculate resampling ratio: 24000/16000 = 1.5
        original_length = len(pcm_float)
        resampled_length = int(original_length * 1.5)
        pcm_24khz = signal.resample(pcm_float, resampled_length)
        
        log("info", f"Resampled {original_length} samples (16kHz) to {len(pcm_24khz)} samples (24kHz)")
        return pcm_24khz
    
    async def _process_resampled_audio(self, pcm_24khz: np.ndarray, base_timestamp: float):
        """Process resampled audio data"""
        # Add to buffer for proper frame size handling
        self.pcm_buffer = np.concatenate((self.pcm_buffer, pcm_24khz))
        
        # Process complete frames (1920 samples at 24kHz = 80ms)
        while len(self.pcm_buffer) >= self.frame_size:
            chunk = self.pcm_buffer[:self.frame_size]
            self.pcm_buffer = self.pcm_buffer[self.frame_size:]
            
            # Create and send audio chunk
            audio_chunk = self._create_audio_chunk(chunk, base_timestamp)
            self.audio_subject.on_next(audio_chunk)
            
            # Optional: Calculate energy for VAD
            energy = self._calculate_energy(chunk)
        
        log("info", f"Buffer length: {len(self.pcm_buffer)}")