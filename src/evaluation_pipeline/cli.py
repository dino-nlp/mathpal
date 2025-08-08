"""CLI for running evaluation with Opik."""

import argparse
from typing import Any

from .config import EvaluationConfig
from .evaluator import Evaluator


def _build_predict_fn(model: Any, tokenizer: Any, device: str = "cuda"):
    from training_pipeline.inference.inference_engine import InferenceEngine

    engine = InferenceEngine(model, tokenizer, device=device, max_new_tokens=64)

    def predict_fn(question: str) -> str:
        return engine.generate(question)

    return predict_fn


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with Opik")
    parser.add_argument("--dataset", type=str, default="ngohongthai/exam-sixth_grade-instruct-dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--experiment-name", type=str, default="gemma3n-math-eval")
    parser.add_argument("--opik-dataset", type=str, default="mathpal-eval")
    parser.add_argument("--numeric-tol", type=float, default=1e-6)
    parser.add_argument("--no-sympy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # Optional: load an already fine-tuned model for prediction
    parser.add_argument("--model-repo", type=str, help="HF repo id of LoRA or merged model")

    args = parser.parse_args()

    cfg = EvaluationConfig(
        dataset_name=args.dataset,
        dataset_split=args.split,
        dataset_subset=args.subset,
        max_samples=args.max_samples,
        experiment_name=args.experiment_name,
        opik_dataset_name=args.opik_dataset,
        numeric_tol=args.numeric_tol,
        sympy_simplify=not args.no_sympy,
    )

    # Build simple predict function
    if args.model_repo:
        # Load with Unsloth for speed
        from unsloth import FastModel, get_chat_template
        model, tok = FastModel.from_pretrained(args.model_repo)
        tok = get_chat_template(tok, "gemma-3n")
        predict_fn = _build_predict_fn(model, tok, device=args.device)
    else:
        # Dummy predictor (for smoke tests)
        def predict_fn(q: str) -> str:
            return "0"

    ev = Evaluator(cfg, predict_fn)
    summary = ev.run()
    print(summary)


if __name__ == "__main__":
    main()

