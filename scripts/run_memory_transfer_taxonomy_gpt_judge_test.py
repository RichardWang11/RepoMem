import json
import unittest

from scripts.run_memory_transfer_taxonomy_gpt_judge import (
    SYSTEM_PROMPT,
    parse_judge_json,
    prompt_sha256,
)


class MemoryTransferJudgeTest(unittest.TestCase):
    def test_prompt_declares_oracle_boundary(self):
        self.assertEqual(64, len(prompt_sha256()))
        self.assertIn("oracle-informed", SYSTEM_PROMPT)
        self.assertIn("not a deployable classifier", SYSTEM_PROMPT)
        self.assertIn("T1 exact-component bridge", SYSTEM_PROMPT)
        self.assertIn("repair mechanism need not", SYSTEM_PROMPT)
        self.assertIn("file overlap alone is insufficient", SYSTEM_PROMPT)
        self.assertIn("repository root or broad top-level namespace", SYSTEM_PROMPT)
        self.assertIn("peer-checkable specificity", SYSTEM_PROMPT)
        self.assertIn("exact bare code", SYSTEM_PROMPT)

    def test_valid_applicable_contract(self):
        value = {
            "sample_id": "MTT-1",
            "applicability_label": "Direct-help",
            "transfer_bridge_tags": ["T1", "T3"],
            "primary_transfer_bridge": "T1",
            "contribution_role_tags": ["R1"],
            "nontransfer_reason_tags": [],
            "taxonomy_evidence": "The repair mechanism transfers.",
            "unclear_flag": False,
            "confidence": "high",
        }
        self.assertEqual(value, parse_judge_json(json.dumps(value), "MTT-1"))

    def test_rejects_mixed_applicable_and_nontransfer_tags(self):
        value = {
            "sample_id": "MTT-1",
            "applicability_label": "Direct-help",
            "transfer_bridge_tags": ["T1"],
            "primary_transfer_bridge": "T1",
            "contribution_role_tags": ["R1"],
            "nontransfer_reason_tags": ["S1"],
            "taxonomy_evidence": "Mixed invalid coding.",
            "unclear_flag": False,
            "confidence": "high",
        }
        with self.assertRaises(ValueError):
            parse_judge_json(json.dumps(value), "MTT-1")

    def test_valid_unclear_requires_u_only(self):
        value = {
            "sample_id": "MTT-1",
            "applicability_label": "Unclear",
            "transfer_bridge_tags": [],
            "primary_transfer_bridge": None,
            "contribution_role_tags": [],
            "nontransfer_reason_tags": ["U"],
            "taxonomy_evidence": "The packet lacks the required code context.",
            "unclear_flag": True,
            "confidence": "low",
        }
        self.assertEqual(value, parse_judge_json(json.dumps(value), "MTT-1"))

    def test_rejects_u_on_irrelevant(self):
        value = {
            "sample_id": "MTT-1",
            "applicability_label": "Irrelevant",
            "transfer_bridge_tags": [],
            "primary_transfer_bridge": None,
            "contribution_role_tags": [],
            "nontransfer_reason_tags": ["U"],
            "taxonomy_evidence": "Insufficient evidence is not irrelevance.",
            "unclear_flag": False,
            "confidence": "low",
        }
        with self.assertRaises(ValueError):
            parse_judge_json(json.dumps(value), "MTT-1")


if __name__ == "__main__":
    unittest.main()
