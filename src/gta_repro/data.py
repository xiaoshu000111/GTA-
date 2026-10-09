"""Dataset metadata matching Table 1 of the GTA paper."""

from dataclasses import dataclass, replace
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSpec:
    hf_id: str
    text_field: str
    label_field: str
    labels: List[str]
    train_size: int
    test_size: int


DATASETS: Dict[str, DatasetSpec] = {
    "sst5": DatasetSpec(
        hf_id="SetFit/sst5",
        text_field="text",
        label_field="label",
        labels=["very negative", "negative", "neutral", "positive", "very positive"],
        train_size=8544,
        test_size=2210,
    ),
    "amazon_massive": DatasetSpec(
        hf_id="mteb/amazon_massive_scenario",
        text_field="text",
        label_field="label",
        labels=[
            "alarm", "audio", "iot", "general", "datetime", "music",
            "takeaway", "weather", "play", "news", "recommendation", "qa",
            "calendar", "social", "transport", "cooking", "email", "lists",
        ],
        train_size=11514,
        test_size=2974,
    ),
    "emotion": DatasetSpec(
        hf_id="dair-ai/emotion",
        text_field="text",
        label_field="label",
        labels=["sadness", "joy", "love", "anger", "fear", "surprise"],
        train_size=16000,
        test_size=2000,
    ),
    "bbc_news": DatasetSpec(
        hf_id="SetFit/bbc-news",
        text_field="text",
        label_field="label",
        labels=["business", "entertainment", "politics", "sport", "tech"],
        train_size=1225,
        test_size=1000,
    ),
}


def load_paper_dataset(name: str, *, trust_remote_code: bool = False):
    """Load a paper dataset and validate fields and documented split sizes."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the training dependencies: pip install -e '.[train]'") from exc
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[name]
    dataset = load_dataset(spec.hf_id, trust_remote_code=trust_remote_code)
    for split in ("train", "test"):
        if split not in dataset:
            raise ValueError(f"{spec.hf_id} has no {split!r} split")
        required = {spec.text_field, spec.label_field}
        missing = required - set(dataset[split].column_names)
        if missing:
            raise ValueError(f"{spec.hf_id}/{split} lacks fields {sorted(missing)}")
    expected_sizes = {"train": spec.train_size, "test": spec.test_size}
    for split, expected in expected_sizes.items():
        if len(dataset[split]) != expected:
            raise ValueError(
                f"{spec.hf_id}/{split} has {len(dataset[split])} rows; "
                f"the paper reports {expected}. Pin or audit the dataset revision."
            )
    label_feature = dataset["train"].features.get(spec.label_field)
    feature_names = getattr(label_feature, "names", None)
    if feature_names:
        if set(feature_names) != set(spec.labels):
            raise ValueError(
                f"label mismatch for {spec.hf_id}: dataset={feature_names}, "
                f"paper mapping={spec.labels}"
            )
        # Integer labels must follow the dataset's declared ClassLabel order,
        # not an assumed order copied from prose.
        spec = replace(spec, labels=list(feature_names))
    return dataset, spec
