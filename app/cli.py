import argparse
import sys
from pathlib import Path
from app.core import (
    DEFAULT_MODEL_DIR,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    predict_next_item,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="crop-health-sequence-monitor",
        description="Crop Health Sequence Monitor"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    p_patterns = sub.add_parser("patterns", help="mine and display frequent patterns")
    p_patterns.add_argument("--min-support", type=float, default=0.1, help="minimum support threshold")
    p_patterns.add_argument("--max-size", type=int, default=3, help="maximum size of itemsets")

    p_train = sub.add_parser("train", help="train small language model on agricultural sequences")
    p_train.add_argument("--out-dir", type=str, default=DEFAULT_MODEL_DIR, help="directory to save model")
    p_train.add_argument("--epochs", type=int, default=5, help="number of training epochs")

    p_eval = sub.add_parser("evaluate", help="evaluate next-item predictions on unseen sequences")
    p_eval.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="directory of trained model")

    p_pred = sub.add_parser("predict", help="predict next agricultural observation")
    p_pred.add_argument("observations", nargs="*", help="current observations (e.g. 'Aphid Infestation' 'Leaf Curling')")
    p_pred.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="directory of trained model")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Crop Health Sequence Monitor is ready. Open README.md for your milestones.")
    elif args.command == "patterns":
        baskets = milestone_1()
        patterns = milestone_2(baskets, min_support=args.min_support, max_size=args.max_size)
        sorted_patterns = sorted(patterns.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        print(f"Mined {len(sorted_patterns)} frequent patterns (min_support={args.min_support}):")
        for items, support in sorted_patterns:
            formatted_items = ", ".join(sorted(items))
            print(f"- [{formatted_items}] (support: {support:.2f})")
    elif args.command == "train":
        print(f"Training language model for {args.epochs} epochs...")
        res = milestone_3(out_dir=args.out_dir, epochs=args.epochs)
        print(f"Training completed. Final loss: {res['final_loss']:.4f}. Saved to {res['saved_to']}")
    elif args.command == "evaluate":
        if not Path(args.model_dir).exists():
            print(f"Error: Model directory not found at '{args.model_dir}'. Run train first.", file=sys.stderr)
            sys.exit(1)
        res = milestone_4(model_dir=args.model_dir)
        print(f"Evaluation on unseen test sequences:")
        print(f"Accuracy (Top-1): {res['accuracy'] * 100:.2f}% ({res['correct']}/{res['total_cases']})")
    elif args.command == "predict":
        if not args.observations:
            print("Error: Please provide at least one observation.", file=sys.stderr)
            sys.exit(1)
        if not Path(args.model_dir).exists():
            print(f"Error: Model directory not found at '{args.model_dir}'. Run train first.", file=sys.stderr)
            sys.exit(1)
        try:
            prediction = predict_next_item(args.observations, model_dir=args.model_dir)
            print(f"Predicted next observation: {prediction}")
        except Exception as e:
            print(f"Error during prediction: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
