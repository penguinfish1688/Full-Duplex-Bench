# Personaplex Inference Scripts

This folder provides example inference scripts for [Personaplex](https://github.com/penguinfish1688/personaplex) — a full-duplex spoken dialogue model with both text and voice prompt support.

For installation instructions, please refer to the official [Personaplex repo](https://github.com/penguinfish1688/personaplex).

---

## Full-Duplex-Bench v1.5 Inference

For **v1.5**, inference is handled through a **client–server pipeline** using `inference.py`.

### Step 1. Start the Personaplex Server

First, go into the [Personaplex repo](https://github.com/penguinfish1688/personaplex).  
Then, start the Personaplex server using:

```bash
# Via Docker/Apptainer
apptainer run --nv personaplex.sif

# Or direct Python
python -m moshi.server
```

The server will run on port 8998 by default (configurable with `--port` flag).

### Step 2. Configure `inference.py`

Edit the following section at the top of `inference.py`:

```python
root_dir_path = "YOUR_ROOT_DIRECTORY_PATH"
tasks = [
    "YOUR_TASK_NAME",
]
prefix = ""  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files

# Optional: Configure voice and text prompts
voice_prompt = "NATF2.pt"  # Voice prompt filename
text_prompt = "You are a helpful and friendly assistant."  # Text prompt
```

- **`root_dir_path`**: base directory of Full-Duplex-Bench v1.5 (e.g., `data-full-duplex-bench/v1.5/`).  
- **`tasks`**: list of tasks to evaluate (e.g., `user_interruption`, `user_backchannel`, `talking_to_other`, `background_speech`).  
- **`prefix`**:  
  - `""` → raw input (with overlaps)  
  - `"clean_"` → cleaned non-overlap reference files  
- **`overwrite`**: whether existing outputs should be replaced.  
- **`voice_prompt`**: voice prompt filename (e.g., `NATF0.pt`, `NATM1.pt`). If not specified, default voice is used.
- **`text_prompt`**: text instruction prompt for the model. Customizable per task.

### Step 3. Run Inference

Once configured, run:

```bash
python inference.py --server_ip localhost:8998
```

The script will generate output files (`output.wav`) for each evaluated sample.

---

## Supported Voice Prompts

Personaplex supports pre-built voice prompts:
- **Natural Female**: `NATF0.pt`, `NATF1.pt`, `NATF2.pt`, `NATF3.pt`
- **Natural Male**: `NATM0.pt`, `NATM1.pt`, `NATM2.pt`, `NATM3.pt`

You can also create custom voice prompts by encoding WAV files to voice token files.

---

## Protocol Details

Personaplex uses a WebSocket-based protocol similar to Moshi:
- **Message Type 0x00**: Handshake (server → client)
- **Message Type 0x01**: Audio frames (bidirectional, Opus-encoded)
- **Message Type 0x02**: Text output (server → client, UTF-8 encoded)

---

## Citation

If you use this code in your work, please cite:

```bibtex
@article{lin2025full,
  title={Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities},
  author={Lin, Guan-Ting and Lian, Jiachen and Li, Tingle and Wang, Qirui and Anumanchipalli, Gopala and Liu, Alexander H and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2503.04721},
  year={2025}
}

@article{lin2025full,
  title={Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models},
  author={Lin, Guan-Ting and Kuan, Shih-Yun Shan and Wang, Qirui and Lian, Jiachen and Li, Tingle and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2507.23159},
  year={2025}
}
```

