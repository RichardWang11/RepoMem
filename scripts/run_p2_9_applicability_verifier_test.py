import unittest

from scripts.run_p2_9_applicability_verifier import (
    exact_binomial_ci,
    hunk_local_anchors,
    valid_anchor,
    verdict,
)


class P29VerifierTest(unittest.TestCase):
    def test_hunk_anchors_ignore_context_and_rank_scope_first(self):
        patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@ def target(value):
 context_only_call()
-    return old_api(value)
+    return new_api(value.attr)
"""
        anchors = hunk_local_anchors(patch)
        self.assertEqual("target", anchors[0])
        self.assertIn("new_api", anchors)
        self.assertIn("old_api", anchors)
        self.assertNotIn("context_only_call", anchors)

    def test_generic_and_one_character_names_are_excluded(self):
        self.assertFalse(valid_anchor("self"))
        self.assertFalse(valid_anchor("x"))
        self.assertTrue(valid_anchor("resolve_relation"))

    def test_unknown_temporal_and_zero_anchors_pass_through(self):
        features = {
            "temporal_status": "UNKNOWN",
            "memory_file_count": 1,
            "surviving_files": ["a.py"],
            "moved_anchor_count": 0,
            "same_file_anchor_count": 0,
            "anchor_count": 0,
        }
        self.assertEqual((True, []), verdict(features, "V-Struct"))

    def test_struct_rejects_unresolved_anchor(self):
        features = {
            "temporal_status": "BEFORE_BASE",
            "memory_file_count": 1,
            "surviving_files": ["a.py"],
            "moved_anchor_count": 0,
            "same_file_anchor_count": 0,
            "anchor_count": 2,
        }
        retained, reasons = verdict(features, "V-Struct")
        self.assertFalse(retained)
        self.assertEqual(["anchors_unresolved"], reasons)

    def test_three_of_three_exact_interval_is_wide(self):
        lower, upper = exact_binomial_ci(3, 3)
        self.assertAlmostEqual(0.2924017738, lower)
        self.assertEqual(1.0, upper)


if __name__ == "__main__":
    unittest.main()
