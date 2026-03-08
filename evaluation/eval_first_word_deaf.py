"""
Please implement the following Python function based on the docstring. 
Use the 'openai' library to call the LLM judge, and handle concurrent API calls if possible to speed up the evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path
from typing import Any

import soundfile as sf
from openai import OpenAI

class Q2Data:
    dir_path: str
    interruption_question: str
    expected_answer: str
    model_response: str
    gpt_score: int | None
    gpt_reason: str | None

    def __init__(self, dir_path, interruption_question, expected_answer, model_response):
        self.dir_path = dir_path
        self.interruption_question = interruption_question
        self.expected_answer = expected_answer
        self.model_response = model_response
        self.gpt_score = None
        self.gpt_reason = None


def _normalize_answer(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _list_sample_dirs(root_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir does not exist: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _extract_word_items(transcript: Any) -> list[dict[str, Any]]:
    """Return flattened word items with possible timestamp metadata.

    Expected common formats:
    - [{"speaker": ..., "item": [{"text": ..., "timestamp": [start, end]}, ...]}, ...]
    - [{"text": ..., "timestamp": [start, end]}, ...]
    - {"chunks": [{"text": ..., "timestamp": [start, end]}, ...]}
    """
    items: list[dict[str, Any]] = []

    if isinstance(transcript, dict):
        if isinstance(transcript.get("chunks"), list):
            for chunk in transcript["chunks"]:
                if isinstance(chunk, dict):
                    items.append(chunk)
        elif isinstance(transcript.get("item"), list):
            for item in transcript["item"]:
                if isinstance(item, dict):
                    items.append(item)
        elif isinstance(transcript.get("words"), list):
            for item in transcript["words"]:
                if isinstance(item, dict):
                    items.append(item)

    elif isinstance(transcript, list):
        for entry in transcript:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry.get("item"), list):
                for item in entry["item"]:
                    if isinstance(item, dict):
                        items.append(item)
            else:
                items.append(entry)

    return items


def _item_start_time(item: dict[str, Any]) -> float | None:
    ts = item.get("timestamp")
    if isinstance(ts, (list, tuple)) and len(ts) >= 1 and isinstance(ts[0], (int, float)):
        return float(ts[0])
    start = item.get("start")
    if isinstance(start, (int, float)):
        return float(start)
    return None


def _to_sentence(words: list[str]) -> str:
    text = " ".join(w.strip() for w in words if str(w).strip())
    text = re.sub(r"\s+", " ", text).strip()
    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text

def model_response_for_q2(root_dir, model_response_time=15, buffer_time=1):
    results: list[Q2Data] = []

    for sample_dir in _list_sample_dirs(root_dir):
        user_interrupt_path = sample_dir / "user_interrupts_text.json"
        transcript_path = sample_dir / "output_transcript.json"
        output_wav_path = sample_dir / "output.wav"

        if not user_interrupt_path.exists():
            raise FileNotFoundError(f"user_interrupts_text.json not found in {sample_dir}")
        if not transcript_path.exists():
            raise FileNotFoundError(f"output_transcript.json not found in {sample_dir}")
        if not output_wav_path.exists():
            raise FileNotFoundError(f"output.wav not found in {sample_dir}")

        with user_interrupt_path.open("r", encoding="utf-8") as f:
            user_interrupt_data = json.load(f)

        with transcript_path.open("r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        duration = float(sf.info(str(output_wav_path)).duration)
        start_time = max(0.0, duration - float(model_response_time) - float(buffer_time))

        all_items = _extract_word_items(transcript_data)
        selected_words: list[str] = []

        for item in all_items:
            text = item.get("text") if isinstance(item, dict) else None
            if not isinstance(text, str):
                continue
            word_start = _item_start_time(item)
            if word_start is None or word_start >= start_time:
                selected_words.append(text)

        model_response = _to_sentence(selected_words)
        question = str(user_interrupt_data.get("question_2", "")).strip()
        expected = _normalize_answer(user_interrupt_data.get("question_2_answer", ""))

        results.append(
            Q2Data(
                dir_path=str(sample_dir),
                interruption_question=question,
                expected_answer=expected,
                model_response=model_response,
            )
        )

    return results


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
            pos = text.find("{", end)
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
    raise ValueError("No valid JSON object found in model output")


def _evaluate_one(openai_client: OpenAI, system_prompt: str, item: Q2Data) -> Q2Data:
    user_prompt = (
        f"Interruption Question: {item.interruption_question}\n"
        f"Model Response: {item.model_response}\n"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = _extract_json_object(raw)

    score = parsed.get("score", 0)
    reason = parsed.get("reason", "No reason provided")

    try:
        score_int = int(score)
    except (TypeError, ValueError):
        score_int = 0

    if score_int not in (0, 1):
        score_int = 0

    item.gpt_score = score_int
    item.gpt_reason = str(reason)

    output = {
        "dir_path": item.dir_path,
        "interruption_question": item.interruption_question,
        "expected_answer": item.expected_answer,
        "model_response": item.model_response,
        "gpt_score": item.gpt_score,
        "gpt_reason": item.gpt_reason,
    }
    out_path = Path(item.dir_path) / "q2_evaluation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return item


def evaluate_with_gpt(data_list: list[Q2Data], openai_client: OpenAI, max_workers=10) -> list[Q2Data]:
    SYSTEM_PROMPT = '''
    Evaluate if the AI successfully heard the VERY FIRST WORD of the User's Interruption Question.
    Do NOT infer context. Be strictly literal.

    Score 1: The response EXPLICITLY names the entity/subject of the first word (or a direct synonym).
    Score 0: The response does NOT explicitly name the first word's subject. 

    Automatically assign 0 for any of these failures:
    - Tail-End Catching: Reacts only to the end of the question, missing the first word.
    - Pronoun Dodging: Uses "it/they/that/this" instead of explicitly naming the subject.
    - Self-Referential: Answers with "I/my/me" because it missed the subject.
    - Naked Answers: Just says "Yes/No/True/False" with no subject attached.
    - Unrelated/Gibberish: Fails to address the first word entirely.

    Output ONLY JSON: {"score": <0 or 1>, "reason": "<1-sentence explanation identifying the first word and if it was explicitly said>"}
    '''
    if not data_list:
        return []

    workers = max(1, min(int(max_workers), len(data_list)))
    results: list[Q2Data] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_evaluate_one, openai_client, SYSTEM_PROMPT, item): item
            for item in data_list
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                item.gpt_score = 0
                item.gpt_reason = f"Evaluation failed: {exc}"
                fail_output = {
                    "dir_path": item.dir_path,
                    "interruption_question": item.interruption_question,
                    "expected_answer": item.expected_answer,
                    "model_response": item.model_response,
                    "gpt_score": item.gpt_score,
                    "gpt_reason": item.gpt_reason,
                }
                out_path = Path(item.dir_path) / "q2_evaluation_results.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(fail_output, f, indent=2, ensure_ascii=False)
                results.append(item)

    # Preserve original order for downstream consistency.
    by_dir = {r.dir_path: r for r in results}
    return [by_dir.get(x.dir_path, x) for x in data_list]


def _build_summary(results: list[Q2Data]) -> dict[str, Any]:
    total = len(results)
    success = sum(1 for x in results if x.gpt_score == 1)
    failure = sum(1 for x in results if x.gpt_score == 0)
    return {
        "total": total,
        "success": success,
        "failure": failure,
        "accuracy": (success / total) if total else 0.0,
        "items": [
            {
                "dir_path": x.dir_path,
                "interruption_question": x.interruption_question,
                "model_response": x.model_response,
                "gpt_score": x.gpt_score,
                "gpt_reason": x.gpt_reason,
            }
            for x in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Q2 response text and evaluate with GPT concurrently."
    )
    parser.add_argument("root_dir", help="Root directory containing sample subfolders")
    parser.add_argument(
        "--model-response-time",
        type=float,
        default=15.0,
        help="Expected response duration near the end of output.wav (default: 15)",
    )
    parser.add_argument(
        "--buffer-time",
        type=float,
        default=1.0,
        help="Extra seconds to include before response window (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum concurrent OpenAI requests (default: 10)",
    )
    parser.add_argument(
        "--skip-gpt",
        action="store_true",
        help="Only extract model responses, do not call OpenAI evaluator.",
    )

    args = parser.parse_args()
    data_list = model_response_for_q2(
        root_dir=args.root_dir,
        model_response_time=args.model_response_time,
        buffer_time=args.buffer_time,
    )

    if not data_list:
        print("No valid samples found (need user_interrupts_text.json, output_transcript.json, output.wav).")
        return 1

    print(f"Prepared {len(data_list)} sample(s) from: {args.root_dir}")

    if args.skip_gpt:
        print("Skipping GPT evaluation as requested.")
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it before running GPT evaluation.")
        return 2

    client = OpenAI()
    evaluated = evaluate_with_gpt(
        data_list=data_list,
        openai_client=client,
        max_workers=args.max_workers,
    )

    summary = _build_summary(evaluated)
    summary_path = Path(args.root_dir) / "q2_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"Done. total={summary['total']} success={summary['success']} "
        f"failure={summary['failure']} accuracy={summary['accuracy']:.3f}"
    )
    print(f"Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())