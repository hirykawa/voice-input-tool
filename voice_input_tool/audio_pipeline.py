"""Audio activity helpers for live recording status."""

import logging
import time

import numpy as np

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
