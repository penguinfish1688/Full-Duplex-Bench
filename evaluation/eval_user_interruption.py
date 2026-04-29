import json
import re
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3


def parse_output(data):
    # Regular expression to match each example

    example_pattern = re.compile(
        r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL
    )

    example = {}
    # Parse the examples
    for match in example_pattern.finditer(data):
        analysis = match.group(1).strip()
        rating = match.group(2).strip()

        # Append the parsed example
        example = {"analysis": analysis, "rating": int(rating)}

    return example


MODEL_NAME = "gpt-4-turbo"
SEED = 0
SYSTEM_MSG = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """


def _evaluate_one_file_dir(file_dir, client):
    while True:
        print(f"Processing {file_dir} ...")

        out_after_interrupt_path = os.path.join(file_dir, "output.json")
        if not os.path.exists(out_after_interrupt_path):
            raise FileNotFoundError("Required file 'output.json' not found.")

        with open(out_after_interrupt_path, "r") as f:
            out_after_interrupt = json.load(f)

        metadata_path = os.path.join(file_dir, "interrupt.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Required file 'interrupt.json' not found.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        in_interrupt_text = metadata[0]["interrupt"]
        in_before_interrupt_text = metadata[0]["context"]
        input_end_time = metadata[0]["timestamp"][1]
        out_after_interrupt_text = out_after_interrupt["text"]

        TOR = None
        latency = None
        segments_cw = out_after_interrupt["chunks"]

        # If no transcription from CrisperWhisper, the model does not take turn.
        if len(segments_cw) == 0:
            TOR = 0
        else:
            output_start_time = segments_cw[0]["timestamp"][0]
            duration = segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
            if duration < turn_duration_threshold:
                if len(segments_cw) <= turn_num_words_threshold:
                    TOR = 0
                else:
                    TOR = 1
                    latency = output_start_time - input_end_time
            else:
                TOR = 1
                latency = output_start_time - input_end_time

        result = {
            "file_dir": file_dir,
            "take_turn": TOR,
            "score": None,
            "latency": None,
        }

        if TOR != 1:
            return result

        user_msg = f"""
        - Contextual user turn: {in_before_interrupt_text}
        - User interrupting turn: {in_interrupt_text}
        - AI's response: {out_after_interrupt_text}
        """

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            seed=SEED,
        )

        prediction = response.choices[0].message.content

        print(prediction)
        parsed_output = parse_output(prediction + "\n")

        print(parsed_output)
        if "rating" not in parsed_output:
            continue

        with open(os.path.join(file_dir, "rating.json"), "w") as f:
            json.dump(parsed_output, f)

        result["score"] = parsed_output["rating"]
        if latency < 0:
            result["latency"] = 0
        elif latency >= 0:
            result["latency"] = latency
        return result


def eval_user_interruption(root_dir, client, max_workers=1):
    file_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            file_dirs.append(os.path.join(root, dir))

    score_list = []
    take_turn_list = []
    latency_list = []
    file_dirs = sorted(file_dirs)
    workers = max(1, min(int(max_workers), len(file_dirs))) if file_dirs else 1

    def _collect(result):
        take_turn_list.append(result["take_turn"])
        if result["score"] is not None:
            score_list.append(result["score"])
        if result["latency"] is not None:
            latency_list.append(result["latency"])

    if workers == 1:
        for file_dir in tqdm(file_dirs):
            _collect(_evaluate_one_file_dir(file_dir, client))
    else:
        print(f"Running user_interruption evaluation with max_workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_evaluate_one_file_dir, file_dir, client)
                for file_dir in file_dirs
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                _collect(future.result())

    print("---------------------------------------------------")
    print("[Result]")
    print("Average rating: ", sum(score_list) / len(score_list))
    print("Average take turn: ", sum(take_turn_list) / len(take_turn_list))
    print("Average latency: ", sum(latency_list) / len(latency_list))
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please run: export OPENAI_API_KEY='...'."
        )
    client = OpenAI(api_key=api_key)
    client.models.list()
    eval_user_interruption(args.root_dir, client, max_workers=args.max_workers)
