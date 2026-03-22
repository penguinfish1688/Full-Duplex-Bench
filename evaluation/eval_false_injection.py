import argparse
import json
import os
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


def _parse_rating(raw_text: str) -> int:
    m = re.search(r"\b([0-5])\b", raw_text)
    if m is None:
        raise ValueError(f"Cannot parse score 0-5 from judge output: {raw_text}")
    return int(m.group(1))


def _load_answer_text(output_json_path: Path) -> str:
    if not output_json_path.exists():
        raise FileNotFoundError(f"Missing ASR transcript file: {output_json_path}")
    with output_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {output_json_path}, got {type(payload)}")
    return str(payload.get("text", "")).strip()


def _load_question_1(sample_dir: Path) -> str:
    question_path = sample_dir / "user_interrupts_text.json"
    if not question_path.exists():
        raise FileNotFoundError(f"Missing question file: {question_path}")

    with question_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {question_path}, got {type(payload)}")

    question = payload.get("question_1", "")
    question = str(question).strip()
    if not question:
        raise ValueError(f"Missing or empty 'question_1' in {question_path}")
    return question


def _wav_duration_seconds(wav_path: Path) -> float | None:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            nframes = wf.getnframes()
            framerate = wf.getframerate()
        if framerate <= 0:
            return None
        return float(nframes) / float(framerate)
    except wave.Error:
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            if info.samplerate <= 0:
                return None
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return None


def _judge_score(client: OpenAI, question: str, answer: str, model_name: str) -> tuple[int, str]:
    system_msg = (
        "The user asked exactly one question. "
        "Score how related the answer is to that question on this 0-5 integer scale: "
        "0 totally unrelated, 1 not related, 2 slightly related, 3 related, 4 highly related, 5 perfectly related. "
        "Output only one integer from 0 to 5."
    )
    user_msg = f"Question:\n{question}\n\nAnswer:\n{answer}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        seed=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    score = _parse_rating(raw)
    return score, raw


def eval_false_injection(
    root_dir: str,
    client: OpenAI,
    layer: int,
    expectation: float,
    model_name: str = "gpt-4o-mini",
) -> Path:
    root = Path(root_dir)
    expectation_tag = f"{float(expectation):.6f}".rstrip("0").rstrip(".")
    if expectation_tag == "":
        expectation_tag = "0"
    if expectation == -1.0:
        expectation_tag = "-1"

    named_pattern = f"*/output_{int(layer)}_{expectation_tag}.wav"
    output_wavs = [p for p in root.glob(named_pattern) if p.is_file()]
    output_wavs.sort(key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else p.parent.name)
    if not output_wavs:
        raise FileNotFoundError(f"No files matched strict pattern {root_dir}/{named_pattern}")

    print(f"[eval_false_injection] Using strict outputs pattern: {named_pattern}")

    records: list[dict[str, Any]] = []
    scores: list[int] = []

    for output_wav in tqdm(output_wavs, desc="Evaluating false injection"):
        sample_dir = output_wav.parent
        output_json = sample_dir / f"output_{int(layer)}_{expectation_tag}_asr.json"
        answer = _load_answer_text(output_json)
        question = _load_question_1(sample_dir)

        if not answer:
            score = 0
            judge_raw = ""
        else:
            score, judge_raw = _judge_score(
                client=client,
                question=question,
                answer=answer,
                model_name=model_name,
            )

        records.append(
            {
                "sample_id": sample_dir.name,
                "question": question,
                "output_wav": str(output_wav),
                "output_json": str(output_json),
                "duration_sec": _wav_duration_seconds(output_wav),
                "answer": answer,
                "score": int(score),
                "judge_raw": judge_raw,
            }
        )
        scores.append(int(score))

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    payload: dict[str, Any] = {
        "question_source": "root_dir/*/user_interrupts_text.json::question_1",
        "layer": int(layer),
        "expectation": float(expectation),
        "judge_model": model_name,
        "num_samples": len(records),
        "average_score": avg_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": records,
    }

    output_path = root / f"false_injection_{int(layer)}_{expectation_tag}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"Saved false-injection evaluation to {output_path} "
        f"(num_samples={len(records)}, average_score={avg_score:.4f})"
    )
    return output_path


def _build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please run: export OPENAI_API_KEY='...'."
        )
    return OpenAI(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate false injection results")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expectation", type=float, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    client = _build_openai_client()
    eval_false_injection(
        root_dir=args.root_dir,
        client=client,
        layer=int(args.layer),
        expectation=float(args.expectation),
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
