import unittest

import pandas as pd

from scripts.analyze_fragmentation_logistic import build_models, fit_logit, get_columns


class FragmentationLogisticTest(unittest.TestCase):
    def test_get_columns_supports_existing_fragmentation_schema(self):
        df = pd.DataFrame(
            [
                {
                    "file_overlap_jaccard_mean": 0.1,
                    "baseline_ok_k1": False,
                    "repomem_ok_k1": True,
                }
            ]
        )

        self.assertEqual(get_columns(df), ("file_overlap_jaccard_mean", "baseline_ok_k1", "repomem_ok_k1"))

    def test_fit_logit_detects_positive_direction(self):
        x = [0.01, 0.02, 0.03, 0.80, 0.90, 1.00]
        y = [0, 0, 0, 1, 1, 1]

        row = fit_logit(x, y)

        self.assertGreater(row["coef_per_1sd"], 0)
        self.assertGreater(row["odds_ratio_per_1sd"], 1)

    def test_build_models_labels_rescue_and_regression(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "file_overlap_jaccard_mean": 0.01 + i * 0.001,
                    "baseline_ok_k1": False,
                    "repomem_ok_k1": False,
                }
            )
            rows.append(
                {
                    "file_overlap_jaccard_mean": 0.8 + i * 0.001,
                    "baseline_ok_k1": False,
                    "repomem_ok_k1": True,
                }
            )
            rows.append(
                {
                    "file_overlap_jaccard_mean": 0.01 + i * 0.001,
                    "baseline_ok_k1": True,
                    "repomem_ok_k1": False,
                }
            )
            rows.append(
                {
                    "file_overlap_jaccard_mean": 0.8 + i * 0.001,
                    "baseline_ok_k1": True,
                    "repomem_ok_k1": True,
                }
            )

        model_rows, meta = build_models(pd.DataFrame(rows))
        by_name = {row["model"]: row for row in model_rows}

        self.assertEqual(meta["analyzed_rows"], 80)
        self.assertGreater(by_name["rescue_given_baseline_wrong"]["coef_per_1sd"], 0)
        self.assertLess(by_name["regress_given_baseline_correct"]["coef_per_1sd"], 0)


if __name__ == "__main__":
    unittest.main()
