import unittest

from auto_search_main import normalize_function_response
from util.process_output import parse_raw_loc_output


class FailureRepairTests(unittest.TestCase):
    def test_normalize_function_response_keeps_unquoted_path_text(self):
        response = "OBSERVATION:\n['django/db/models/base.py']"

        self.assertEqual(normalize_function_response(response), response)

    def test_parse_raw_loc_output_extracts_inline_valid_python_paths(self):
        raw = """
        ### Revised locations
        The fix involves `django/db/migrations/executor.py` and supporting context.
        - django/db/migrations/loader.py: MigrationLoader.check_consistent_history
        """
        valid_files = [
            "django/db/migrations/executor.py",
            "django/db/migrations/loader.py",
        ]

        found_files, _ = parse_raw_loc_output(raw, valid_files)

        self.assertEqual(
            found_files,
            [
                "django/db/migrations/executor.py",
                "django/db/migrations/loader.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
