"""Audio activity and VAD segment pipeline helpers."""

import logging
import queue
import time

import numpy as np

from voice_input_tool.audio_constants import BLOCK_SIZE, SAMPLE_RATE

log = logging.getLogger("voice_input")

AUDIO_ACTIVITY_RMS_THRESHOLD = 0.025
AUDIO_ACTIVITY_RELEASE_RMS_THRESHOLD = 0.015
AUDIO_ACTIVITY_HOLD_SECONDS = 0.35
AUDIO_INITIAL_NOISE_FLOOR = 0.003
AUDIO_NOISE_FLOOR_ALPHA_IDLE = 0.08
AUDIO_NOISE_FLOOR_ALPHA_ACTIVE = 0.01
AUDIO_NOISE_ON_MULTIPLIER = 2.2
AUDIO_NOISE_OFF_MULTIPLIER = 1.4
AUDIO_STATUS_UPDATE_INTERVAL = 0.08
BUSY_STATUSES = {"processing", "correcting", "inserting"}


class AudioActivityTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_status_update = 0.0
        self.noise_floor = AUDIO_INITIAL_NOISE_FLOOR
        self.is_speaking = False
        self.last_active_at = 0.0

    def status_for_samples(self, samples, current_status):
        if current_status in BUSY_STATUSES:
            return None

        now = time.monotonic()
        if now - self.last_status_update < AUDIO_STATUS_UPDATE_INTERVAL:
            return None
        self.last_status_update = now

        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        alpha = AUDIO_NOISE_FLOOR_ALPHA_ACTIVE if self.is_speaking else AUDIO_NOISE_FLOOR_ALPHA_IDLE
        self.noise_floor = max(0.00001, (1.0 - alpha) * self.noise_floor + alpha * rms)

        on_threshold = max(AUDIO_ACTIVITY_RMS_THRESHOLD, self.noise_floor * AUDIO_NOISE_ON_MULTIPLIER)
        off_threshold = max(AUDIO_ACTIVITY_RELEASE_RMS_THRESHOLD, self.noise_floor * AUDIO_NOISE_OFF_MULTIPLIER)

        if rms >= on_threshold:
            self.is_speaking = True
            self.last_active_at = now
        elif self.is_speaking and (
            rms <= off_threshold
            and now - self.last_active_at >= AUDIO_ACTIVITY_HOLD_SECONDS
        ):
            self.is_speaking = False

        new_status = "hearing" if self.is_speaking else "listening"
        if new_status != current_status:
            log.info(
                "音声状態: %s rms=%.4f noise=%.4f on=%.4f off=%.4f",
                new_status,
                rms,
                self.noise_floor,
                on_threshold,
                off_threshold,
            )
        return new_status


class VadAudioHistory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.blocks = []
        self.sample_count = 0

    def accept_block(self, vad, block):
        self.blocks.append((self.sample_count, block.copy()))
        self.sample_count += len(block)
        vad.accept_waveform(block)

    def segment_samples_with_preroll(self, segment, pre_roll_duration, sample_rate=SAMPLE_RATE):
        segment_samples = np.array(segment.samples, dtype=np.float32)
        segment_start = int(getattr(segment, "start", 0))
        segment_end = segment_start + len(segment_samples)
        pre_roll_samples = int(sample_rate * pre_roll_duration)
        padded_start = max(0, segment_start - pre_roll_samples)

        speech_samples = self._audio_range(padded_start, segment_end)
        if len(speech_samples) == 0:
            speech_samples = segment_samples

        added = max(0, min(segment_start, segment_end) - padded_start)
        if added:
            log.info("VAD先頭補完: %.2f秒", added / sample_rate)

        self._prune(max(0, segment_end - pre_roll_samples))
        return speech_samples

    def _audio_range(self, start, end):
        chunks = []
        for block_start, block in self.blocks:
            block_end = block_start + len(block)
            if block_end <= start:
                continue
            if block_start >= end:
                break
            chunk_start = max(0, start - block_start)
            chunk_end = min(len(block), end - block_start)
            if chunk_start < chunk_end:
                chunks.append(block[chunk_start:chunk_end])
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def _prune(self, keep_from):
        self.blocks = [
            (start, block)
            for start, block in self.blocks
            if start + len(block) > keep_from
        ]


def process_audio_queue(audio_queue, is_recording, vad, vad_history, process_segments, block_size=BLOCK_SIZE):
    audio_buffer = np.array([], dtype=np.float32)
    chunk_count = 0

    while is_recording() or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        chunk_count += 1
        if chunk_count == 1:
            log.info("音声データ受信開始")
        audio_buffer = np.concatenate([audio_buffer, chunk])

        while len(audio_buffer) >= block_size:
            block = audio_buffer[:block_size]
            audio_buffer = audio_buffer[block_size:]

            vad_history.accept_block(vad, block)
            process_segments()

    if len(audio_buffer) > 0:
        padded = np.pad(audio_buffer, (0, block_size - len(audio_buffer)))
        vad_history.accept_block(vad, padded)
    vad.flush()
    process_segments()
    vad.reset()
    vad_history.reset()


def drain_vad_segments(
    vad,
    vad_history,
    min_speech_duration,
    pre_roll_duration,
    on_segment,
    sample_rate=SAMPLE_RATE,
):
    while not vad.empty():
        segment = vad.front
        segment_sample_count = len(segment.samples)
        speech_samples = vad_history.segment_samples_with_preroll(
            segment,
            pre_roll_duration,
            sample_rate=sample_rate,
        )
        vad.pop()
        duration = len(speech_samples) / sample_rate
        log.info("VAD検出: %.1f秒", duration)

        if segment_sample_count < sample_rate * min_speech_duration:
            log.info("最小発話長未満、スキップ")
            continue

        on_segment(speech_samples)
