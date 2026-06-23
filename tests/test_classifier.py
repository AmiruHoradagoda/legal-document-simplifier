from pathlib import Path

from src.classifier import (
    build_label_mappings,
    encode_labels,
    load_label_mapping,
    normalize_split,
    save_label_mapping,
)


def test_build_label_mappings_are_stable_and_sorted():
    label2id, id2label = build_label_mappings(["payment", "general", "payment"])

    assert label2id == {"general": 0, "payment": 1}
    assert id2label == {0: "general", 1: "payment"}


def test_encode_labels_uses_mapping():
    label2id, _ = build_label_mappings(["payment", "general"])

    assert encode_labels(["payment", "general"], label2id) == [1, 0]


def test_label_mapping_round_trip(tmp_path: Path):
    label2id = {"general": 0, "payment": 1}
    id2label = {0: "general", 1: "payment"}
    path = tmp_path / "label_mapping.json"

    save_label_mapping(path, label2id, id2label)
    loaded_label2id, loaded_id2label = load_label_mapping(path)

    assert loaded_label2id == label2id
    assert loaded_id2label == id2label


def test_normalize_split_aliases_validation():
    assert normalize_split("val") == "validation"
    assert normalize_split("DEV") == "validation"
    assert normalize_split("train") == "train"
