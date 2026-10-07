"""Prepare a blinded human-adjudication sheet for the MTT holdout cohort."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_OUTPUT = Path(
    "results/paper2_pilot/analysis/memory_transfer_taxonomy"
)
CODING_FIELDS = (
    "applicability_label",
    "transfer_bridge_tags",
    "primary_transfer_bridge",
    "contribution_role_tags",
    "nontransfer_reason_tags",
)
ADJUDICATION_FIELDS = CODING_FIELDS + (
    "taxonomy_evidence",
    "unclear_flag",
    "confidence",
    "coder_notes",
)
RECOMMENDATIONS = {
    "MTT-AB35F07CB0": (
        "`Irrelevant / S4`. The shared word and high-level goal 'dtype "
        "preservation' do not identify the same concrete failure mechanism: "
        "the Memory normalizes a serialization encoding value, whereas the "
        "Gold patch preserves a MultiIndex level dtype in `__array__`. This "
        "fails the frozen v3 specificity requirement for T3."
    ),
    "MTT-A91C33F706": (
        "`Irrelevant / S2`. Both changes lie in the narrow "
        "`sphinx.ext.autodoc` subsystem, but one alters constructor signature "
        "selection and the other repairs mocked-class naming. 'Autodoc "
        "introspection' is too broad to satisfy frozen v3 T3."
    ),
    "MTT-13AA6A9CEE": (
        "`Irrelevant / S3`. The Memory changes a `multiset_partitions` "
        "counting example; the Gold patch changes object-copy semantics in "
        "the distinct `partitions` function. Same file and naming family do "
        "not constitute the same concrete component under frozen v3 T1."
    ),
}
ACCEPTED_RESOLUTIONS = {
    "MTT-AB35F07CB0": ("S4", RECOMMENDATIONS["MTT-AB35F07CB0"]),
    "MTT-A91C33F706": ("S2", RECOMMENDATIONS["MTT-A91C33F706"]),
    "MTT-13AA6A9CEE": ("S3", RECOMMENDATIONS["MTT-13AA6A9CEE"]),
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    output: Path, accept_recommendations: bool = False
) -> tuple[int, list[str]]:
    coder_a = load_csv(output / "holdout_human_coder_a.csv")
    coder_b = load_csv(output / "holdout_human_coder_b.csv")
    if len(coder_a) != len(coder_b):
        raise ValueError("Coder A/B row counts differ")

    b_by_id = {row["sample_id"]: row for row in coder_b}
    if len(b_by_id) != len(coder_b):
        raise ValueError("Coder B contains duplicate sample IDs")

    adjudicated = []
    disagreements = []
    for left in coder_a:
        sample_id = left["sample_id"]
        if sample_id not in b_by_id:
            raise ValueError(f"Coder B is missing {sample_id}")
        right = b_by_id[sample_id]
        changed = [field for field in CODING_FIELDS if left[field] != right[field]]
        row = dict(left)
        if changed:
            disagreements.append((left, right, changed))
            if accept_recommendations:
                nontransfer_tag, evidence = ACCEPTED_RESOLUTIONS[sample_id]
                row.update({
                    "applicability_label": "Irrelevant",
                    "transfer_bridge_tags": "",
                    "primary_transfer_bridge": "",
                    "contribution_role_tags": "",
                    "nontransfer_reason_tags": nontransfer_tag,
                    "taxonomy_evidence": evidence.replace("`", ""),
                    "unclear_flag": "0",
                    "confidence": "medium",
                    "coder_notes": (
                        "adjudication_user_accepted_frozen_v3_recommendation"
                    ),
                })
            else:
                for field in ADJUDICATION_FIELDS:
                    row[field] = ""
        else:
            # Coding is identical. Retain A's independently written evidence;
            # mark this as a mechanical agreement copy, not a new judgment.
            row["coder_notes"] = "adjudication_prefill_exact_A_B_agreement"
        adjudicated.append(row)

    write_csv(output / "holdout_adjudicated.csv", adjudicated)

    lines = [
        "# Holdout Adjudication Worklist",
        "",
        "The 27 exact A/B agreements were mechanically prefilled. The rows below",
        "were the only A/B disagreements and were reviewed against the frozen",
        "v3 codebook. Do not inspect sealed",
        "machine features, trajectories, ranks, actions, or outcomes.",
        "",
    ]
    for left, right, changed in disagreements:
        lines.extend(
            [
                f"## {left['sample_id']} — {left['case_id']}",
                "",
                f"- Differing fields: `{', '.join(changed)}`",
                f"- Coder A: `{left['applicability_label']}`; "
                f"T=`{left['transfer_bridge_tags']}`; "
                f"primary=`{left['primary_transfer_bridge']}`; "
                f"R=`{left['contribution_role_tags']}`; "
                f"S=`{left['nontransfer_reason_tags']}`",
                f"- A evidence: {left['taxonomy_evidence']}",
                f"- Coder B: `{right['applicability_label']}`; "
                f"T=`{right['transfer_bridge_tags']}`; "
                f"primary=`{right['primary_transfer_bridge']}`; "
                f"R=`{right['contribution_role_tags']}`; "
                f"S=`{right['nontransfer_reason_tags']}`",
                f"- B evidence: {right['taxonomy_evidence']}",
                "- Protocol-grounded recommendation: "
                + RECOMMENDATIONS.get(left["sample_id"], "No recommendation."),
                "- Joint decision: **ACCEPTED RECOMMENDATION**"
                if accept_recommendations else "- Joint decision: **PENDING**",
                "",
            ]
        )
    (output / "holdout_adjudication_worklist.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    pending = [] if accept_recommendations else [
        left["sample_id"] for left, _, _ in disagreements
    ]
    return len(adjudicated) - len(disagreements), pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accept-recommendations", action="store_true")
    args = parser.parse_args()
    agreements, disagreements = prepare(
        args.output, accept_recommendations=args.accept_recommendations
    )
    print(f"Prefilled exact agreements: {agreements}")
    print(f"Pending joint adjudication: {len(disagreements)}")
    for sample_id in disagreements:
        print(sample_id)


if __name__ == "__main__":
    main()
