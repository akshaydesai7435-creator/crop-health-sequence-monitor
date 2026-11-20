from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from thilimarketbasket.mine import frequent_itemsets
from thilimarketbasket.verbalize import verbalize_basket

DATASET_CATALOG = [
    "Aphid Infestation",
    "Leaf Curling",
    "Neem Oil Spray",
    "Powdery Mildew",
    "White Leaf Spots",
    "Fungicide Spray",
    "Nitrogen Deficiency",
    "Yellowing Leaves",
    "Urea Application",
    "Low Soil Moisture",
    "Leaf Wilting",
    "Drip Irrigation",
    "Waterlogging",
    "Root Rot",
    "Drainage Improvement",
    "Stem Borer Attack",
    "Stem Tunneling",
    "Pheromone Traps",
    "Weed Infestation",
    "Stunted Crop Growth",
    "Manual Weeding"
]

DEFAULT_SEQUENCES = [
    ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
    ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"],
    ["Nitrogen Deficiency", "Yellowing Leaves", "Urea Application"],
    ["Low Soil Moisture", "Leaf Wilting", "Drip Irrigation"],
    ["Waterlogging", "Root Rot", "Drainage Improvement"],
    ["Stem Borer Attack", "Stem Tunneling", "Pheromone Traps"],
    ["Weed Infestation", "Stunted Crop Growth", "Manual Weeding"],
    ["Aphid Infestation", "Leaf Curling"],
    ["Aphid Infestation", "Neem Oil Spray"],
    ["Powdery Mildew", "Fungicide Spray"],
    ["Nitrogen Deficiency", "Yellowing Leaves"],
    ["Nitrogen Deficiency", "Urea Application"],
    ["Low Soil Moisture", "Drip Irrigation"],
    ["Waterlogging", "Root Rot"],
    ["Stem Borer Attack", "Pheromone Traps"],
    ["Weed Infestation", "Manual Weeding"],
    ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray", "Low Soil Moisture"],
    ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray", "Nitrogen Deficiency"],
    ["Nitrogen Deficiency", "Yellowing Leaves", "Urea Application", "Drip Irrigation"],
    ["Low Soil Moisture", "Leaf Wilting", "Drip Irrigation", "Aphid Infestation"],
    ["Waterlogging", "Root Rot", "Drainage Improvement", "Yellowing Leaves"],
    ["Stem Borer Attack", "Stem Tunneling", "Pheromone Traps", "Neem Oil Spray"],
    ["Weed Infestation", "Stunted Crop Growth", "Manual Weeding", "Urea Application"],
    ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
    ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"],
    ["Nitrogen Deficiency", "Yellowing Leaves", "Urea Application"],
    ["Low Soil Moisture", "Leaf Wilting", "Drip Irrigation"],
    ["Waterlogging", "Root Rot", "Drainage Improvement"],
    ["Stem Borer Attack", "Stem Tunneling", "Pheromone Traps"],
    ["Weed Infestation", "Stunted Crop Growth", "Manual Weeding"]
]

DEFAULT_MODEL_DIR = "models/crop_health_lm"


def milestone_1() -> List[List[str]]:
    return [list(basket) for basket in DEFAULT_SEQUENCES]


def milestone_2(baskets: List[List[str]] | None = None, min_support: float = 0.1, max_size: int = 3) -> Dict[frozenset, float]:
    if baskets is None:
        baskets = milestone_1()
    patterns = frequent_itemsets(baskets, min_support=min_support, max_size=max_size)
    return patterns


def get_dataset(seed: int = 42) -> Tuple[List[List[str]], List[List[str]]]:
    rng = np.random.default_rng(seed)
    patterns = [
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"],
        ["Nitrogen Deficiency", "Yellowing Leaves", "Urea Application"],
        ["Low Soil Moisture", "Leaf Wilting", "Drip Irrigation"],
        ["Waterlogging", "Root Rot", "Drainage Improvement"],
        ["Stem Borer Attack", "Stem Tunneling", "Pheromone Traps"],
        ["Weed Infestation", "Stunted Crop Growth", "Manual Weeding"],
    ]
    baskets = []
    for _ in range(80):
        pat = list(patterns[int(rng.integers(len(patterns)))])
        rng.shuffle(pat)
        baskets.append(pat)
    rng.shuffle(baskets)
    split = int(len(baskets) * 0.8)
    return baskets[:split], baskets[split:]


def milestone_3(
    train_baskets: List[List[str]] | None = None,
    out_dir: str = DEFAULT_MODEL_DIR,
    model_name: str = "gpt2",
    epochs: int = 5,
    lr: float = 1e-4
) -> dict:
    if train_baskets is None:
        train_baskets, _ = get_dataset()

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.transformer.wte.parameters():
        p.requires_grad = True
    for block in model.transformer.h[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    texts = [verbalize_basket(b) for b in train_baskets]
    losses = []
    for _ in range(epochs):
        for text in texts:
            inputs = tok(text + tok.eos_token, return_tensors="pt")
            out = model(**inputs, labels=inputs["input_ids"])
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"final_loss": float(np.mean(losses[-10:])), "saved_to": str(out_dir)}


def score_candidate_items(
    model,
    tok,
    visible: List[str],
    candidates: List[str]
) -> np.ndarray:
    prefix = "Basket: " + "; ".join(visible) + ";" if visible else "Basket:"
    prefix_enc = tok(prefix, return_tensors="pt")
    prefix_len = prefix_enc["input_ids"].shape[1]

    scores = []
    model.eval()
    with torch.no_grad():
        for cand in candidates:
            cand_text = " " + cand
            full_text = prefix + cand_text
            inp = tok(full_text, return_tensors="pt")
            cand_tokens = inp["input_ids"][0, prefix_len:]
            if len(cand_tokens) == 0:
                scores.append(-1e9)
                continue
            logits = model(**inp).logits[0, prefix_len - 1 : -1]
            log_probs = torch.log_softmax(logits, dim=-1)
            cand_score = sum(log_probs[i, tok_id].item() for i, tok_id in enumerate(cand_tokens))
            scores.append(cand_score)
    return np.array(scores)


def predict_next_item(
    observations: List[str],
    model_dir: str = DEFAULT_MODEL_DIR,
    catalog: List[str] | None = None
) -> str:
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_dir}")

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    if catalog is None:
        catalog = DATASET_CATALOG

    valid_observations = [o.strip() for o in observations if o.strip()]
    scores = score_candidate_items(model, tok, valid_observations, catalog)

    cand_indices = {name: i for i, name in enumerate(catalog)}
    for obs in valid_observations:
        if obs in cand_indices:
            scores[cand_indices[obs]] = -1e9

    best_idx = int(np.argmax(scores))
    return catalog[best_idx]


def milestone_4(
    test_baskets: List[List[str]] | None = None,
    model_dir: str = DEFAULT_MODEL_DIR,
    catalog: List[str] | None = None
) -> dict:
    if test_baskets is None:
        _, test_baskets = get_dataset()
    if catalog is None:
        catalog = DATASET_CATALOG

    cases = []
    for b in test_baskets:
        distinct = sorted(set(b))
        if len(distinct) >= 2:
            target = distinct[-1]
            visible = distinct[:-1]
            cases.append({"visible": visible, "target": target})

    if not cases:
        return {"accuracy": 0.0, "total_cases": 0}

    correct = 0
    for case in cases:
        predicted = predict_next_item(case["visible"], model_dir=model_dir, catalog=catalog)
        if predicted == case["target"]:
            correct += 1

    accuracy = correct / len(cases)
    return {"accuracy": float(accuracy), "correct": correct, "total_cases": len(cases)}


def milestone_5(observations: List[str], model_dir: str = DEFAULT_MODEL_DIR) -> str:
    return predict_next_item(observations, model_dir=model_dir)
