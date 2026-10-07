import json
import unittest

from scripts.run_applicability_gpt_judge import (
    EXPECTED_RESPONSE_MODEL,
    SYSTEM_PROMPT,
    label_complete,
    parse_judge_json,
    prompt_sha256,
)


class JudgeContractTest(unittest.TestCase):
    def valid_value(self):
        return {
            "sample_id": "AVL-1",
            "applicability_label": "Indirect-help",
            "applicability_evidence": "It narrows the correct module.",
            "validity_label": "Partially-valid",
            "validity_evidence": "One symbol moved but still resolves.",
            "confidence": "medium",
        }

    def test_prompt_hash_is_stable(self):
        self.assertEqual(64, len(prompt_sha256()))
        self.assertIn("untrusted research data", SYSTEM_PROMPT)
        self.assertIn("Gold patch", SYSTEM_PROMPT)

    def test_valid_json_contract(self):
        value = self.valid_value()
        self.assertEqual(value, parse_judge_json(json.dumps(value), "AVL-1"))

    def test_rejects_extra_keys_and_bad_sample(self):
        value = self.valid_value()
        value["extra"] = "forbidden"
        with self.assertRaises(ValueError):
            parse_judge_json(json.dumps(value), "AVL-1")
        value.pop("extra")
        with self.assertRaises(ValueError):
            parse_judge_json(json.dumps(value), "AVL-2")

    def test_complete_label_requires_locked_model_and_prompt(self):
        value = {
            **self.valid_value(),
            "model": EXPECTED_RESPONSE_MODEL,
            "prompt_sha256": "frozen",
            "raw_response": "{}",
        }
        self.assertTrue(label_complete(value, "frozen"))
        value["model"] = "gpt-4o"
        self.assertFalse(label_complete(value, "frozen"))


if __name__ == "__main__":
    unittest.main()
