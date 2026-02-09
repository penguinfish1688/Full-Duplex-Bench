#!/usr/bin/env python3
"""
Personaplex Batch Inference Client

This script streams WAV files to a running Personaplex server and captures the
generated responses. Personaplex supports both text and voice prompts, allowing
for flexible conversational interactions.

Usage:
    python inference.py --server_ip localhost:8998
"""

from __future__ import annotations

import argparse
import asyncio
from glob import glob
from pathlib import Path
from typing import List
import json

import numpy as np
import soundfile as sf
import sphn
import torch
import torchaudio.functional as AF
import websockets
import websockets.exceptions as wsex
import yaml


### Configuration ###
root_dir_path = "YOUR_ROOT_DIRECTORY_PATH"
tasks = [
    "YOUR_TASK_NAME",
]
prefix = ""  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files

# Personaplex-specific configuration
voice_prompt = "NATF0.pt"  # Voice prompt (e.g., "NATF0.pt", "NATM1.pt", or custom)
text_prompt = "You are a helpful and friendly assistant."  # Text prompt
#####################

# Load config.yaml (optional) to avoid hard-coded values
try:
    cfg_path = Path(__file__).with_name("config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # Update configuration values if present in YAML
        root_dir_path = cfg.get("root_dir_path", root_dir_path)
        tasks = cfg.get("tasks", tasks) or tasks
        prefix = cfg.get("prefix", prefix)
        overwrite = bool(cfg.get("overwrite", overwrite))
        voice_prompt = cfg.get("voice_prompt", voice_prompt)
        text_prompt = cfg.get("text_prompt", text_prompt)
except Exception as e:
    # If YAML isn't available or parsing fails, continue using defaults
    print(f"[WARN] Config file not found. Using defaults. ({e})")


SEND_SR = 24_000
FRAME_SMP = 1_920
SKIP_FRAMES = 1
FRAME_SEC = FRAME_SMP / SEND_SR


def _patch_sphn():
    """Patch sphn/opus library for compatibility."""
    if not hasattr(sphn.OpusStreamWriter, "read_bytes"):
        for alt in ("get_bytes", "flush_bytes", "read_data"):
            if hasattr(sphn.OpusStreamWriter, alt):
                setattr(
                    sphn.OpusStreamWriter,
                    "read_bytes",
                    getattr(sphn.OpusStreamWriter, alt),
                )
                break
        else:
            setattr(sphn.OpusStreamWriter, "read_bytes", lambda self: b"")
    if not hasattr(sphn.OpusStreamReader, "read_pcm"):
        for alt in ("get_pcm", "receive_pcm", "read_float"):
            if hasattr(sphn.OpusStreamReader, alt):
                setattr(
                    sphn.OpusStreamReader,
                    "read_pcm",
                    getattr(sphn.OpusStreamReader, alt),
                )
                break
        else:
            setattr(
                sphn.OpusStreamReader, "read_pcm", lambda self: np.empty(0, np.float32)
            )


_patch_sphn()


def _mono(x: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono."""
    return x if x.ndim == 1 else x.mean(axis=1)


def _resample(x: np.ndarray, sr: int, tgt: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if sr == tgt:
        return x
    y = torch.from_numpy(x.astype(np.float32) / 32768).unsqueeze(0)
    y = AF.resample(y, sr, tgt)[0].numpy()
    return (y * 32768).astype(np.int16)


def _chunk(sig: np.ndarray) -> List[np.ndarray]:
    """Split audio signal into fixed-size frames."""
    pad = (-len(sig)) % FRAME_SMP
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, sig.dtype)])
    return [sig[i : i + FRAME_SMP] for i in range(0, len(sig), FRAME_SMP)]


class PersonaplexFileClient:
    """WebSocket client for Personaplex streaming inference."""

    def __init__(
        self,
        ws_url: str,
        inp: Path,
        out: Path,
        voice_prompt: str | None = None,
        text_prompt: str | None = None,
    ):
        """
        Initialize Personaplex client.

        Args:
            ws_url: WebSocket URL to Personaplex server
            inp: Input audio file path
            out: Output audio file path
            voice_prompt: Optional voice prompt filename
            text_prompt: Optional text prompt string
        """
        self.url = ws_url
        self.inp = inp
        self.out = out
        self.voice_prompt = voice_prompt
        self.text_prompt = text_prompt

        # Load and prepare input audio
        sig16, sr = sf.read(inp, dtype="int16")
        self.sig24 = _resample(_mono(sig16), sr, SEND_SR)
        self.max_samples = len(self.sig24)  # target output length

        self.writer = sphn.OpusStreamWriter(SEND_SR)
        self.reader = sphn.OpusStreamReader(SEND_SR)
        
        # Store collected text for output
        self.collected_text = []

    # ────────────────────── Sender ──────────────────────
    async def _send(self, ws):
        """Send audio frames to server."""
        # Send initialization message with text/voice prompts if provided
        init_msg = self._create_init_message()
        if init_msg:
            await ws.send(init_msg)
            await asyncio.sleep(0.1)

        # Stream audio frames
        for frame in _chunk(self.sig24):
            pkt0 = self.writer.append_pcm(frame.astype(np.float32) / 32768)
            if isinstance(pkt0, (bytes, bytearray)):
                await ws.send(b"\x01" + pkt0)
            queued = self.writer.read_bytes()
            if queued:
                await ws.send(b"\x01" + queued)
            await asyncio.sleep(FRAME_SEC)

        # Flush remaining audio
        queued = self.writer.read_bytes()
        if queued:
            await ws.send(b"\x01" + queued)
        await asyncio.sleep(0.5)
        await ws.close()

    def _create_init_message(self) -> bytes | None:
        """
        Create initialization message with prompts.
        
        Format: 0x03 (init marker) + JSON payload
        """
        if not self.voice_prompt and not self.text_prompt:
            return None
        
        init_data = {}
        if self.voice_prompt:
            init_data["voice_prompt"] = self.voice_prompt
        if self.text_prompt:
            init_data["text_prompt"] = self.text_prompt
        
        payload = json.dumps(init_data).encode("utf-8")
        return b"\x03" + payload

    # ────────────────────── Receiver ──────────────────────
    async def _recv(self, ws):
        """Receive and decode audio responses from server."""
        samples_written = 0
        first_pcm_seen = False

        with sf.SoundFile(
            self.out, "w", samplerate=SEND_SR, channels=1, subtype="PCM_16"
        ) as fout:
            try:
                async for msg in ws:
                    if not msg or msg[0] not in (1, 2):
                        continue
                    
                    kind, payload = msg[0], msg[1:]

                    if kind == 1:  # Audio bytes (0x01)
                        self.reader.append_bytes(payload)
                        while True:
                            pcm = self.reader.read_pcm()
                            if pcm.size == 0:
                                break

                            if not first_pcm_seen:
                                # Skip initial frames if configured
                                pad = min(SKIP_FRAMES * FRAME_SMP, self.max_samples)
                                fout.write(np.zeros(pad, dtype=np.int16))
                                samples_written += pad
                                first_pcm_seen = True

                            remain = self.max_samples - samples_written
                            if remain <= 0:
                                continue
                            n_write = min(pcm.size, remain)
                            fout.write((pcm[:n_write] * 32768).astype(np.int16))
                            samples_written += n_write

                    elif kind == 2:  # Text output (0x02)
                        try:
                            text = payload.decode(errors="ignore")
                            self.collected_text.append(text)
                            print("[TEXT]", text)
                        except Exception as e:
                            print(f"[WARN] Failed to decode text: {e}")

            except wsex.ConnectionClosedError as e:
                print("[WARN] recv closed:", e)

        # Pad output if needed
        if samples_written < self.max_samples:
            with sf.SoundFile(
                self.out, "a", samplerate=SEND_SR, channels=1, subtype="PCM_16"
            ) as fout:
                fout.write(np.zeros(self.max_samples - samples_written, dtype=np.int16))

    async def _run(self):
        """Run the WebSocket connection."""
        async with websockets.connect(self.url, max_size=None) as ws:
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if not (isinstance(first, (bytes, bytearray)) and first[:1] == b"\x00"):
                    ws._put_message(first)
            except Exception:
                pass

            await asyncio.gather(self._send(ws), self._recv(ws))
        print("[DONE]", self.inp)

    def run(self):
        """Execute inference."""
        try:
            asyncio.run(self._run())
        except wsex.ConnectionClosedError as e:
            print("[WARN] closed:", e)


def _ws_url(addr: str) -> str:
    """Convert address to WebSocket URL."""
    if "://" in addr:
        proto, rest = addr.split("://", 1)
        proto = "ws" if proto in {"http", "ws"} else "wss"
        return f"{proto}://{rest.rstrip('/')}/api/chat"
    if ":" not in addr:
        addr += ":8998"
    return f"ws://{addr}/api/chat"


def _input_files() -> List[Path]:
    """Collect all input WAV files matching the task patterns."""
    files: List[Path] = []
    # Expand ~ and create a Path object for the root directory
    root = Path(root_dir_path).expanduser()
    for t in tasks:
        pattern = str(root / f"{t}/*/{prefix}input.wav")
        files += [Path(p) for p in sorted(glob(pattern))]
    return files


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("personaplex_batch_client")
    ap.add_argument(
        "--server_ip",
        required=True,
        help="Personaplex server address (host[:port] or http(s):// URL)",
    )
    ap.add_argument(
        "--voice-prompt",
        default=None,
        help="Override voice prompt (e.g., 'NATF0.pt')",
    )
    ap.add_argument(
        "--text-prompt",
        default=None,
        help="Override text prompt instruction",
    )
    args = ap.parse_args()

    url = _ws_url(args.server_ip)
    
    # Use CLI overrides if provided, otherwise use config defaults
    voice_prompt_to_use = args.voice_prompt if args.voice_prompt else voice_prompt
    text_prompt_to_use = args.text_prompt if args.text_prompt else text_prompt

    # Debug output
    print(f"[CONFIG] root_dir_path: {root_dir_path}")
    print(f"[CONFIG] tasks: {tasks}")
    print(f"[CONFIG] prefix: {prefix}")
    print(f"[CONFIG] voice_prompt: {voice_prompt_to_use}")
    print(f"[SERVER] url: {url}")
    
    input_files = _input_files()
    print(f"[FILES] Found {len(input_files)} input files")
    
    if not input_files:
        print("[ERROR] No input files found. Check root_dir_path and tasks in config.yaml")
        return

    for inp in input_files:
        out = inp.with_name(inp.name.replace("input.wav", "output.wav"))
        if not overwrite and out.exists():
            print("[SKIP]", out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print("[RUN]", inp)
        try:
            client = PersonaplexFileClient(
                url,
                inp,
                out,
                voice_prompt=voice_prompt_to_use,
                text_prompt=text_prompt_to_use,
            )
            client.run()
        except Exception as e:
            print(f"[ERROR] Failed to process {inp}: {e}")


if __name__ == "__main__":
    main()
