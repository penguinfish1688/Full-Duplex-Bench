import os
import json
import argparse
from glob import glob

import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

MODEL_NAME = ""


def get_time_aligned_transcription(data_path, task, audio_source="output"):
    # Collect input.wav or output.wav files under the root directory
    source_filename = f"{MODEL_NAME}{audio_source}.wav"
    audio_paths = sorted(glob(f"{data_path}/*/{source_filename}"))

    print(f"Found {len(audio_paths)} {audio_source}.wav files for transcription.")

    # Load the pretrained NeMo ASR model and move to GPU
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    ).cuda()

    for audio_path in tqdm(audio_paths):
        print(f"Transcribing: {audio_path}")
        # Read the audio file (waveform and sample rate)
        waveform, sr = sf.read(audio_path)
        # If multichannel audio, convert to mono by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Default offset is zero (no cropping)
        offset = 0.0

        if task == "user_interruption":
            # Load the interrupt metadata to get [start, end] timestamps
            meta_path = os.path.join(os.path.dirname(audio_path), "interrupt.json")
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)

            # We only care about the end of the interruption
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt

            # Compute the sample index to start from, and crop the waveform
            start_idx = int(end_interrupt * sr)
            waveform = waveform[start_idx:]

        import tempfile

        # Skip empty or too-short waveforms (NeMo preprocessor requires > 0 samples)
        if len(waveform) == 0 or (len(waveform) / sr) < 0.1:
            print(f"[SKIP] {audio_path}: waveform too short ({len(waveform)} samples, {len(waveform)/sr:.3f}s)")
            # Write empty result
            result_path = os.path.join(os.path.dirname(audio_path), f"{audio_source}.json")
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                json.dump({"text": "", "chunks": []}, f, indent=4)
            continue

        # [Modification] The issue was that the temporary file wasn't being closed/flushed before the ASR model tried to read it, causing it to get empty audio data.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
            sf.write(tmp_name, waveform, sr)
        
        # File is now closed and flushed, safe to read
        asr_outputs = asr_model.transcribe([tmp_name], timestamps=True)
        
        # remove the temp file so you don't leak disk
        os.unlink(tmp_name)

        # Take the first (and only) result
        result = asr_outputs[0]
        word_timestamps = result.timestamp["word"]

        # Build the output dict, adjusting each timestamp by the offset
        chunks = []
        text = ""
        for w in word_timestamps:
            start_time = w["start"] + offset
            end_time = w["end"] + offset
            word = w["word"]

            text += word + " "
            chunks.append(
                {
                    "text": word,
                    "timestamp": [start_time, end_time],
                }
            )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # Write the JSON result next to the WAV file
        result_path = os.path.join(os.path.dirname(audio_path), f"{audio_source}.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio or only after a user interruption"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder containing subfolders with input.wav/output.wav (and interrupt.json)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["full", "user_interruption"],
        help="Choose 'full' for entire transcript or 'user_interruption' to crop before ASR",
    )
    parser.add_argument(
        "--audio-source",
        type=str,
        default="output",
        choices=["output", "input"],
        help="Choose which wav to transcribe in each sample folder: output.wav (default) or input.wav",
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task, args.audio_source)
