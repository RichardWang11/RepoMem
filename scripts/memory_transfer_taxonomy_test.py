import unittest

from scripts.memory_transfer_taxonomy import (
    assert_packet_blinded,
    collapse_case_memory,
    binary_phi,
    compute_discovery_reliability,
    derive_primary_nontransfer,
    multilabel_metrics,
    nominal_metrics,
    sample_holdout,
    split_tags,
    validate_coding_row,
)


def candidate(case, sha, repo, repeat=1, rank=1):
    return {
        "case_id": case,
        "memory_sha": sha,
        "repository": repo,
        "repeat_id": repeat,
        "best_rank": rank,
    }


class MemoryTransferTaxonomyTest(unittest.TestCase):
    def test_tag_parser_and_band_priorities(self):
        self.assertEqual({"S1", "S2"}, split_tags("S1|S2"))
        tags = {"S1", "S2", "S3"}
        self.assertEqual("S1", derive_primary_nontransfer("symbol_only", tags))
        self.assertEqual("S2", derive_primary_nontransfer("parent_only", tags))
        self.assertEqual("S3", derive_primary_nontransfer("exact_file", tags))
        self.assertEqual("U", derive_primary_nontransfer("none", {"U"}))

    def test_collapse_uses_best_rank_and_counts_exposure(self):
        rows = [
            candidate("c1", "abc", "r1", repeat=2, rank=4),
            candidate("c1", "abc", "r1", repeat=1, rank=2),
            candidate("c1", "def", "r1", repeat=1, rank=1),
        ]
        result = collapse_case_memory(rows)
        self.assertEqual(2, len(result))
        first = next(row for row in result if row["memory_sha"] == "abc")
        self.assertEqual(2, first["best_rank"])
        self.assertEqual(2, first["exposure_count"])
        self.assertEqual([1, 2], first["source_repeats"])

    def test_holdout_excludes_discovery_and_obeys_case_cap(self):
        rows = []
        for index in range(20):
            repo = f"repo{index % 4}"
            for memory in range(3):
                rows.append(candidate(f"case{index}", f"sha{index}-{memory}", repo))
        selected, info = sample_holdout(
            rows,
            {("case0", "sha0-0")},
            seed=7,
            sample_size=30,
            attempts=1000,
        )
        self.assertEqual(30, len(selected))
        self.assertGreaterEqual(info["unique_cases"], 15)
        self.assertNotIn(("case0", "sha0-0"), {
            (row["case_id"], row["memory_sha"]) for row in selected
        })
        counts = {}
        for row in selected:
            counts[row["case_id"]] = counts.get(row["case_id"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_packet_forbidden_key_scan(self):
        assert_packet_blinded({"sample_id": "x", "issue": {"text": "ok"}})
        with self.assertRaises(AssertionError):
            assert_packet_blinded({"sample_id": "x", "best_rank": 1})

    def test_coding_contract(self):
        applicable = {
            "applicability_label": "Direct-help",
            "transfer_bridge_tags": "T1|T3",
            "primary_transfer_bridge": "T1",
            "contribution_role_tags": "R1",
            "nontransfer_reason_tags": "",
            "taxonomy_evidence": "Same mechanism and repair pattern.",
            "unclear_flag": "false",
            "confidence": "high",
        }
        self.assertTrue(validate_coding_row(applicable)[0])
        invalid = dict(applicable, nontransfer_reason_tags="S1")
        self.assertFalse(validate_coding_row(invalid)[0])
        irrelevant = dict(
            applicable,
            applicability_label="Irrelevant",
            transfer_bridge_tags="",
            primary_transfer_bridge="",
            contribution_role_tags="",
            nontransfer_reason_tags="S1|S2",
        )
        self.assertTrue(validate_coding_row(irrelevant)[0])

        unclear = dict(
            applicable,
            applicability_label="Unclear",
            transfer_bridge_tags="",
            primary_transfer_bridge="",
            contribution_role_tags="",
            nontransfer_reason_tags="U",
            unclear_flag="1",
            confidence="low",
        )
        self.assertTrue(validate_coding_row(unclear)[0])
        self.assertFalse(
            validate_coding_row(dict(unclear, nontransfer_reason_tags=""))[0]
        )
        self.assertFalse(
            validate_coding_row(dict(irrelevant, nontransfer_reason_tags="U"))[0]
        )

    def test_discovery_reliability_reports_raw_and_unique_gates(self):
        def row(sample, case, sha, label, primary="", transfer="", nontransfer=""):
            return {
                "sample_id": sample,
                "case_id": case,
                "memory_sha": sha,
                "applicability_label": label,
                "primary_transfer_bridge": primary,
                "transfer_bridge_tags": transfer,
                "nontransfer_reason_tags": nontransfer,
            }

        rows = [
            row("s1", "c1", "a", "Direct-help", "T1", "T1"),
            row("s2", "c2", "b", "Irrelevant", nontransfer="S1"),
            row("s3", "c2", "b", "Irrelevant", nontransfer="S1"),
            row("s4", "c3", "c", "Unclear", nontransfer="U"),
        ]
        reliability = compute_discovery_reliability(rows, rows, rows)
        self.assertEqual(4, reliability["raw"]["n"])
        self.assertEqual(3, reliability["unique"]["n"])
        self.assertEqual(0.0, reliability["t3"]["rate"])
        self.assertTrue(reliability["gate_passed"])

    def test_redundancy_and_reliability_helpers(self):
        rows = [
            {
                "transfer_bridge_tags": "T1",
                "contribution_role_tags": "R1",
            },
            {
                "transfer_bridge_tags": "T1",
                "contribution_role_tags": "R1",
            },
            {
                "transfer_bridge_tags": "T3",
                "contribution_role_tags": "R3",
            },
        ]
        redundancy = binary_phi(rows, "T1", "R1")
        self.assertTrue(redundancy["redundant"])
        self.assertEqual(1.0, redundancy["minimum_directional_overlap"])
        metrics = nominal_metrics(["T1", "T2"], ["T1", "T3"])
        self.assertEqual(0.5, metrics["raw_agreement"])
        pairs = [
            (
                {"transfer_bridge_tags": "T1|T3"},
                {"transfer_bridge_tags": "T1"},
            )
        ]
        tags = multilabel_metrics(pairs, "transfer_bridge_tags", {"T1", "T3"})
        self.assertEqual(0.5, tags["mean_row_jaccard"])


if __name__ == "__main__":
    unittest.main()
