import pytest
from app import core


def test_milestone_1():
    baskets = core.milestone_1()
    assert isinstance(baskets, list)
    assert len(baskets) > 0
    assert all(isinstance(b, list) for b in baskets)
    assert any("Aphid Infestation" in b for b in baskets)
    assert any("Low Soil Moisture" in b for b in baskets)


def test_milestone_2():
    baskets = core.milestone_1()
    patterns = core.milestone_2(baskets, min_support=0.1, max_size=3)
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    for itemset, support in patterns.items():
        assert isinstance(itemset, frozenset)
        assert 0.0 < support <= 1.0


def test_milestone_3(tmp_path):
    train_baskets = [
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"]
    ]
    out_dir = str(tmp_path / "model")
    res = core.milestone_3(train_baskets, out_dir=out_dir, epochs=1)
    assert "final_loss" in res
    assert "saved_to" in res
    assert (tmp_path / "model" / "config.json").exists()


def test_milestone_4(tmp_path):
    train_baskets = [
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"]
    ]
    test_baskets = [
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"]
    ]
    out_dir = str(tmp_path / "model")
    core.milestone_3(train_baskets, out_dir=out_dir, epochs=1)
    eval_res = core.milestone_4(test_baskets=test_baskets, model_dir=out_dir)
    assert "accuracy" in eval_res
    assert eval_res["total_cases"] == 2
    assert 0.0 <= eval_res["accuracy"] <= 1.0


def test_milestone_5(tmp_path):
    train_baskets = [
        ["Aphid Infestation", "Leaf Curling", "Neem Oil Spray"],
        ["Powdery Mildew", "White Leaf Spots", "Fungicide Spray"]
    ]
    out_dir = str(tmp_path / "model")
    core.milestone_3(train_baskets, out_dir=out_dir, epochs=1)
    prediction = core.milestone_5(["Aphid Infestation", "Leaf Curling"], model_dir=out_dir)
    assert isinstance(prediction, str)
    assert len(prediction) > 0
    assert prediction != "Aphid Infestation"
    assert prediction != "Leaf Curling"


def test_cli_patterns(capsys):
    from app.cli import main
    main(["patterns", "--min-support", "0.1"])
    out = capsys.readouterr().out
    assert "Mined" in out
    assert "frequent patterns" in out


def test_cli_missing_model_handled(capsys):
    from app.cli import main
    with pytest.raises(SystemExit):
        main(["predict", "Aphid Infestation", "--model-dir", "non_existent_dir_123"])
    err = capsys.readouterr().err
    assert "Model directory not found" in err


def test_cli_missing_observations_handled(capsys):
    from app.cli import main
    with pytest.raises(SystemExit):
        main(["predict"])
    err = capsys.readouterr().err
    assert "provide at least one observation" in err
