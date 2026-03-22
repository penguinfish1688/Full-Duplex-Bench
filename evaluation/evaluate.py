import argparse
import os
import importlib.util
from pathlib import Path
from openai import OpenAI

# For OpenAI API key, export before running:
#   export OPENAI_API_KEY="..."
# organization = "YOUR_ORG_ID"
api_key = os.getenv("OPENAI_API_KEY", "")


def _build_openai_client() -> OpenAI:
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please run: export OPENAI_API_KEY='...'."
        )
    return OpenAI(
        # organization=organization,
        api_key=api_key,
    )


def main():
    parser = argparse.ArgumentParser(description="Run evaluation tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
            "false_injection",
            "behavior",
            "general_before_after",
        ],
        help="Evaluation task to perform.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Injection layer for false_injection output filename metadata.",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=None,
        help="Injection probability for false_injection output filename metadata.",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing data for evaluation.",
    )

    args = parser.parse_args()

    if args.task == "backchannel":
        from eval_backchannel import eval_backchannel

        eval_backchannel(args.root_dir)
    elif args.task == "pause_handling":
        from eval_pause_handling import eval_pause_handling

        eval_pause_handling(args.root_dir)
    elif args.task == "smooth_turn_taking":
        from eval_smooth_turn_taking import eval_smooth_turn_taking

        eval_smooth_turn_taking(args.root_dir)
    elif args.task == "user_interruption":
        from eval_user_interruption import eval_user_interruption

        client = _build_openai_client()
        client.models.list()
        eval_user_interruption(args.root_dir, client)

    elif args.task == "false_injection":
        script_dir = Path(__file__).resolve().parent
        module_path = script_dir / "eval_false_injection.py"
        spec = importlib.util.spec_from_file_location("eval_false_injection", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        eval_false_injection = module.eval_false_injection

        if args.layer is None:
            raise ValueError("--task false_injection requires --layer")
        if args.prob is None:
            raise ValueError("--task false_injection requires --prob")

        client = _build_openai_client()
        client.models.list()
        eval_false_injection(
            root_dir=args.root_dir,
            client=client,
            layer=args.layer,
            prob=args.prob,
        )

    elif args.task == "general_before_after":
        from eval_general_before_after import eval_general_all_split

        config = {
            "trim_silence": True,
            "squim": False,
            "utmosv2": True,
            "speaking_rate": True,
            "trim_mode": "silero",
            "agg": {
                "mode": "trim",
                "trim_prop": 0.05,
            },
            "pitch": True,
            "intensity": True,
        }

        aggregate = True
        output = eval_general_all_split(config, args.root_dir, aggregate=aggregate)
        print("General Evaluation Output:", output)
        if aggregate:
            # save to log file
            dir_name = args.root_dir.split("/")[-1]
            log_path = f"{dir_name}_general.log"
            with open(log_path, "w", encoding="utf-8") as f:
                for key, value in output.items():
                    line = f"{key}: {value}\n"
                    print(line, end="")
                    f.write(line)

    elif args.task == "behavior":
        from eval_behavior import eval_behavior_all

        client = _build_openai_client()
        output = eval_behavior_all(
            args.root_dir, client, task=args.task, aggregate=True
        )

        dir_name = args.root_dir.split("/")[-1]
        log_path = f"{dir_name}_{args.task}.log"

        with open(log_path, "w", encoding="utf-8") as f:
            for ax in ["C"]:
                line = f"Ratios ({ax}-axis): {output[ax]}\n"
                print(line, end="")
                f.write(line)


if __name__ == "__main__":
    main()
