import torch
from dataclasses import dataclass
import numpy as np
import torch

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