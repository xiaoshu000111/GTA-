import math
import unittest

import torch

from gta_repro.formatting import parse_gta_output
from gta_repro.gradients import backward_with_paper_conflict_rule
from gta_repro.losses import grpo_loss
from gta_repro.masking import build_rl_mask, build_segment_mask
from gta_repro.rewards import score_completion


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class CoreTests(unittest.TestCase):
    def test_strict_parser_and_rewards(self):
        text = "<guess>joy</guess><think>sounds happy</think><answer>joy</answer>"
        parsed = parse_gta_output(text)
        self.assertEqual(parsed.guess, "joy")
        self.assertEqual(parsed.answer, "joy")
        reward = score_completion(text, "joy", ["sadness", "joy"])
        self.assertEqual(reward.total, 2.0)
        malformed = score_completion("answer: joy", "joy", ["sadness", "joy"])
        self.assertEqual(malformed.total, 0.0)

    def test_segment_masks_exclude_tags(self):
        tokenizer = CharacterTokenizer()
        text = "P<guess>joy</guess><think>why</think><answer>joy</answer>"
        ids = torch.tensor(tokenizer.encode(text))
        guess = build_segment_mask(ids, tokenizer, ["guess"])
        rl = build_rl_mask(ids, tokenizer, [1])
        self.assertEqual(
            "".join(chr(ids[i]) for i in torch.where(guess)[0]), "joy"
        )
        rl_text = "".join(chr(ids[i]) for i in torch.where(rl)[0])
        self.assertEqual(
            rl_text, "<guess></guess><think>why</think><answer>joy</answer>"
        )
        self.assertFalse(torch.any(guess & rl))

    def test_grpo_is_finite_and_masks_tokens(self):
        policy = torch.tensor([[[-0.2, -0.3], [-0.4, -0.5]]], requires_grad=True)
        old = policy.detach().clone()
        ref = policy.detach().clone()
        rewards = torch.tensor([[2.0, 1.0]])
        mask = torch.tensor([[[True, False], [True, False]]])
        loss, _ = grpo_loss(policy, old, ref, rewards, mask)
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertTrue(torch.all(policy.grad[:, :, 1] == 0))
        self.assertTrue(torch.any(policy.grad[:, :, 0] != 0))

    def test_conflicting_sft_gradient_is_discarded(self):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        sft = (parameter - 1.0).pow(2)
        rl = (parameter + 1.0).pow(2)
        stats = backward_with_paper_conflict_rule(sft, rl, [parameter])
        self.assertLess(stats["gradient_dot"], 0)
        self.assertEqual(stats["used_total_loss"], 0.0)
        self.assertTrue(torch.isclose(parameter.grad, torch.tensor(2.0)))


if __name__ == "__main__":
    unittest.main()
