import json
import tempfile
import unittest
from pathlib import Path
import sys

from gta_repro.detectrl_zh import (
    build_detectrl_prompt,
    downsample_binary_balance,
    load_detectrl_json,
    parse_detection_completion,
    summarize_records,
)
from gta_repro.evaluation import evaluate_detectrl_predictions

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hf_detectrlzh_predict import _read_progress, _write_confusion_csv
from hf_detectrlzh_microtrain import _training_subset


class DetectRLZHTests(unittest.TestCase):
    def test_prompt_and_completion_mapping(self):
        prompt = build_detectrl_prompt("这是一段待检测文本。")
        self.assertIn("文档级来源分类", prompt)
        self.assertIn("不表示文档中AI内容所占百分比", prompt)
        parsed = parse_detection_completion(
            "<guess>机器生成</guess><think>语言模式较模板化。</think><answer>机器生成</answer>"
        )
        self.assertEqual(parsed.prediction, 1)
        self.assertTrue(parsed.strict_format)
        self.assertEqual(parsed.source, "answer")

    def test_guess_and_fallback_are_reported(self):
        guess = parse_detection_completion("<guess>人类撰写</guess>未完成")
        self.assertEqual((guess.prediction, guess.source), (0, "guess"))
        fallback = parse_detection_completion("无法判断", invalid_fallback=1)
        self.assertEqual((fallback.prediction, fallback.source), (1, "fallback"))

    def test_loader_and_balancing(self):
        rows = [
            {"text": "人类甲", "label": 0, "model": "human", "source": "asap"},
            {"text": "机器甲", "label": 1, "model": "qwen", "source": "asap"},
            {"text": "机器乙", "label": 1, "model": "glm", "source": "csl"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            records = load_detectrl_json(str(path))
        self.assertEqual(summarize_records(records)["labels"]["machine"], 2)
        balanced = downsample_binary_balance(records, seed=7)
        self.assertEqual([record.label for record in balanced].count(0), 1)
        self.assertEqual([record.label for record in balanced].count(1), 1)

    def test_official_scenario_metrics_use_ids(self):
        labels = [0, 1, 0, 1, 0, 1]
        preds = [0, 1, 1, 1, 0, 0]
        ids = [1, 2, 4001, 5001, 7001, 10001]
        result = evaluate_detectrl_predictions(labels, preds, ids=ids)
        main = result["subtask_results"]["main_subtasks"]
        self.assertEqual(main["Normal"]["n_examples"], 2)
        self.assertEqual(main["Attack"]["n_examples"], 2)
        self.assertEqual(main["Varying Lengths"]["n_examples"], 2)
        self.assertIsNotNone(result["subtask_results"]["avg_f1"])

    def test_prediction_progress_alignment_and_confusion_csv(self):
        rows = [
            {"text": "人类甲", "label": 0, "id": 1},
            {"text": "机器甲", "label": 1, "id": 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.json"
            data_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            records = load_detectrl_json(str(data_path))
            progress_path = root / "predictions.jsonl"
            progress_path.write_text(
                json.dumps(
                    {"index": 0, "sample_id": 1, "label": 0, "prediction": 0},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(_read_progress(progress_path, records)), 1)
            csv_path = root / "confusion.csv"
            _write_confusion_csv(csv_path, [[4, 1], [2, 3]])
            csv_text = csv_path.read_text(encoding="utf-8")
        self.assertIn("human_0,4,1", csv_text)
        self.assertIn("machine_1,2,3", csv_text)

    def test_training_subset_can_preserve_raw_distribution(self):
        rows = [
            {"text": f"人类{index}", "label": 0} for index in range(2)
        ] + [
            {"text": f"机器{index}", "label": 1} for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            records = load_detectrl_json(str(path))
        raw = _training_subset(records, 8, 42, "raw")
        balanced = _training_subset(records, 4, 42, "balanced")
        self.assertEqual(sum(record.label == 1 for record in raw), 6)
        self.assertEqual(sum(record.label == 1 for record in balanced), 2)


if __name__ == "__main__":
    unittest.main()
