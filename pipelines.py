import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
from rx import operators as ops
from transformers import RetriBertConfig
from utils import AudioChunk, DictWebSocketQueue, ProcessedTokens, TimestampedText, build_message_to_server
from moshi.client_utils import log
from utils import tokens_to_timestamped_text
import rx
from moshi.models import LMModel, MimiModel, LMGen
import sentencepiece
import math
from diart import blocks
from pyannote.core import SlidingWindowFeature, Annotation
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import SegmentationModel
import diart.operators as dops
from prompt import prompt_fact_checking
import time
from collections import deque

class Pipeline(ABC):
    @abstractmethod
    def create_pipeline(self, ws, loop, subject):
        pass

    @abstractmethod
    def warmup(self):
        """Warmup method to prepare the pipeline for processing."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the pipeline state."""
        pass

class ASRPipeline(Pipeline):
    def __init__(self, model_name : str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, audio_delay_seconds, padding_token_id, audio_silence_prefix_seconds, device: str | torch.device):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = LMGen(lm, temp=0, temp_text=0)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.audio_delay_seconds = audio_delay_seconds
        self.padding_token_id = padding_token_id
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.audio_silence_prefix_seconds = audio_silence_prefix_seconds
        self.n_prefix_chunks = math.ceil(self.audio_silence_prefix_seconds * mimi.frame_rate)
        self.text_tokens_accum = []
        self.transcription = []
        # self.model = FactCheckingModel(model_name)
        self.time = time.time()
        self.queue = deque(maxlen=10)  # Use deque for efficient FIFO queue
        self.prompt = [] 



    def reset(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.text_tokens_accum = []
        self.transcription = []

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros((1, 1, self.frame_size), dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            tokens = self.lm_gen.step(codes)
            if tokens is None:
                continue
            text = tokens_to_timestamped_text(
                tokens,
                self.text_tokenizer,
                self.mimi.frame_rate,
                end_of_padding_id=0,
                padding_token_id=self.padding_token_id,
                offset_seconds=int(self.n_prefix_chunks / self.mimi.frame_rate) + self.audio_delay_seconds,
            )
            print(f"Test is: {text}")
        torch.cuda.synchronize()

    def encode_audio(self, audio_chunk: AudioChunk) -> Optional[ProcessedTokens]:
        """Encode audio chunk and generate tokens"""
        try:
            # log("info", f"Encoding audio chunk: {audio_chunk.timestamp}")
            chunk = torch.from_numpy(audio_chunk.data)
            chunk = chunk.to(device=self.device)[None, None]
            codes = self.mimi.encode(chunk)
            if codes is None:
                log("error", "Failed to encode audio chunk, returning None")
                return None
            
            text_tokens = self.lm_gen.step(codes)
            if text_tokens is not None:
                return ProcessedTokens(
                    tokens=text_tokens,
                    codes=codes,
                    timestamp=audio_chunk.timestamp
                )

            log("error", "No text tokens generated, returning None")
            return None
        except Exception as e:
            log("error", f"Error encoding audio: {e}")
            return None

    def process_tokens(self, processed: ProcessedTokens) -> List[TimestampedText]:
        """Process tokens and generate timestamped text"""
        try:
            self.text_tokens_accum.append(processed.tokens)
            utterance_tokens = torch.concat(self.text_tokens_accum, dim=-1)
            timed_text = tokens_to_timestamped_text(
                utterance_tokens,
                self.text_tokenizer,
                self.mimi.frame_rate,
                end_of_padding_id=0,
                padding_token_id=self.padding_token_id,
                offset_seconds=int(self.n_prefix_chunks / self.mimi.frame_rate) + self.audio_delay_seconds,
            )
            return timed_text
        
        except Exception as e:
            log("error", f"Error processing tokens: {e}")
            return []
        
    def generate_text(self, key,loop = None, is_get =False, dictWebSocketQueue: DictWebSocketQueue = None):
        # res = self.model.call(prompt)
        # res = res[0]["generated_text"][-1]
        # log("info", f"Generated text: {res}")
        # msg = b"\x02" + bytes(res, encoding="utf8")
        # return res
        if (is_get == False):
            print(f"Sending message with prompt: {self.prompt}")
            asyncio.run_coroutine_threadsafe(self.send_message(key, dictWebSocketQueue), loop)
    
        else:
            if len(self.prompt) == 0:
                return build_message_to_server("Attention", "Not enough data to generate text", "low", "")
            else :
                return build_message_to_server("Fact Checking", self.prompt[1]['content'], "low", "")
            

    async def send_message(self, key, dictWebSocketQueue: DictWebSocketQueue, ):
        """Send a message to the map queue"""
        if len(self.prompt) == 0:
            await dictWebSocketQueue.send_message(key, build_message_to_server("Attention", "Not enough data to generate text", "low", ""))
        else:
            await dictWebSocketQueue.send_message( key,self.prompt[1]['content'])

    
    def build_prompt(self, key, dictWebSocketQueue : DictWebSocketQueue, loop, timed_text: List[TimestampedText]):
        """Build the prompt from the dialogue history"""
        if len(timed_text) <= len(self.transcription) or len(timed_text) <= 1:
            return
        
        text = timed_text[len(self.transcription):-1]  # Exclude the last item
        self.transcription.extend(text)
        text = " ".join([str(m.text) for m in text])
        log("info", f"Building prompt with text: {text}")
        if len(text) == 0:
            return
        self.queue.append(text)
        if len(self.queue) >= self.queue.maxlen:
            old_text = " ".join(self.queue)
            self.queue.clear()
            self.prompt = [{"role": "system", "content": prompt_fact_checking}, {"role": "user", "content": old_text}]
        

    def create_pipeline(self, key, dictWebSocketQueue : DictWebSocketQueue, loop, subject):
        """Create the reactive pipeline for online processing"""
        return (subject.pipe(
                # Filter out None values and errors
                ops.filter(lambda x: x is not None),
                # Encode audio to tokens
                ops.map(self.encode_audio),
                # Filter out None results
                ops.filter(lambda x: x is not None),
                # Process tokens to timestamped text
                ops.map(self.process_tokens),
                # Filter out empty results
                ops.filter(lambda x: len(x) > 0),
                # Send transcription asynchronously
                ops.map(lambda x: self.build_prompt(key,dictWebSocketQueue, loop, x)),
                # If a prompt is built, generate text in the results collector and in a separate coroutine
                ops.do_action(lambda _: self.generate_text(key, dictWebSocketQueue, loop)),
                # Handle errors gracefully
                ops.catch(lambda e, source: rx.empty().pipe(
                    ops.do_action(lambda _: log("error", f"Pipeline error: {e}"))
                ))
        ))
    
    def generate_text_offline(self, prompt, results_collector):
        try:
            if prompt is None:
                return
            """Generate text from the prompt for offline processing"""
            log("info", f"Generating text with prompt")
            res = self.model.call(prompt)
            log ("info", f"Generated text: {res}")
            # res = res[0]["generated_text"][-1]
            results_collector.append({
                'type': 'transcription',
                'prompt' : prompt[1]['content'],
                'text': res,
            })
        except Exception as e:
            log("error", f"Error generating text: {e}")
            return None
        # return res

    def build_prompt_offline(self, timed_text: List[TimestampedText], results_collector):
        """Build the prompt from the dialogue history for offline processing
            Store the transcription in a list until the buffer is full or the end of the stream is reached.
        """
        if len(timed_text) <= len(self.transcription) or len(timed_text) <= 1:
            return None
        text = timed_text[len(self.transcription):-1]  # Exclude the last item
        self.transcription.extend(text)
        text = " ".join([str(m.text) for m in text])
        if len(text) == 0:
            return None
        self.queue.append(text)
        if len(self.queue) >= self.queue.maxlen:
            log("info", f"Queue is full with {len(self.queue)} items, building prompt")
            old_text = " ".join(self.queue)
            self.queue.clear()
            message = [{"role": "system", "content": prompt_fact_checking}, {"role": "user", "content": old_text}]
            return message
        return None

    def create_offline_pipeline(self, subject, results_collector):
        """Create the reactive pipeline for offline processing"""
        return (subject.pipe(
                # Filter out None values and errors
                ops.filter(lambda x: x is not None),
                # Encode audio to tokens
                ops.map(self.encode_audio),
                # Filter out None results
                ops.filter(lambda x: x is not None),
                # Process tokens to timestamped text
                ops.map(self.process_tokens),
                # Filter out empty results
                ops.filter(lambda x: len(x) > 0),
                # Collect transcription results
                # ops.do_action(lambda x: self.collect_transcription_offline(x, results_collector)),
                # Build prompts for offline processing
                ops.do_action(lambda x: self.build_prompt_offline(x, results_collector)),
                # If a prompt is built, generate text in the results collector and in a separate coroutine
                ops.do_action(lambda prompt: self.generate_text(prompt, results_collector)),
                # Handle errors gracefully
                ops.catch(lambda e, source: rx.empty().pipe(
                    ops.do_action(lambda _: log("error", f"Pipeline error: {e}"))
                ))
        ))

class DiarPipeline(Pipeline):
    def __init__(self):
        self.config = SpeakerDiarizationConfig(
            step=0.3,
            tau_active=0.9,
            segmentation=SegmentationModel.from_pyannote(
                "../diarizers/lightning_logs/version_2/checkpoints/epoch=19-step=2940.ckpt"
            ),
            duration=3
        )
        self.diar_pipeline = SpeakerDiarization(
            config=self.config,
        )
        self.last_speaker = None

    def on_diar_result(self, ws, loop, result: List[Tuple[Annotation, SlidingWindowFeature]]):
        annotation, feature = result[0]
        text = ""
        _speaker = None
        for segment, a, speaker in annotation.itertracks(yield_label=True):
            text += f" ---- {speaker} :\n"
            _speaker = speaker
        if _speaker != self.last_speaker and _speaker:
            self.last_speaker = _speaker
            msg = b"\x02" + bytes(text, encoding="utf8")
            asyncio.run_coroutine_threadsafe(ws.send_bytes(msg), loop)

    def create_pipeline(self, ws, loop, subject):
        return (subject.pipe(
            ops.do_action(on_next=lambda results: log("info", f"Received diarization input: {results.shape}")),
            dops.rearrange_audio_stream(
                self.config.duration, self.config.step, self.config.sample_rate
            ),
            ops.map(
                blocks.Resample(
                    16000,
                    self.config.sample_rate,
                    self.config.device,
                )
            ),
            ops.buffer_with_count(count=1),
            ops.map(self.diar_pipeline),
            ops.do_action(on_next=lambda results: self.on_diar_result(ws, loop, results)),
            ops.flat_map(lambda results: rx.from_iterable(results)),
        ))
    
    def warmup(self):
        pass

    def reset(self):
        return super().reset()