from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chapter4_experiments.chapter4_config import (  # noqa: E402
    ABLATION_ARTIFACT_ROOT,
    CHAPTER4_OUTPUT_ROOT,
    E2E_ARTIFACT_ROOT,
    ONLY_CHOICES,
    SINGLE_OP_ARTIFACT_ROOT,
)
from chapter4_experiments.chapter4_shared import (  # noqa: E402
    SectionResult,
    build_chapter4_draft,
    build_figures_catalog,
    build_run_manifest,
    run_e2e_core,
    run_e2e_sum_baseline,
    run_platform_summary,
    run_single_op_ablation,
    run_single_op_core,
    run_single_op_ood,
    run_timeline_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-click runner for Chapter 4 experiments.")
    parser.add_argument("--output-root", type=Path, default=CHAPTER4_OUTPUT_ROOT)
    parser.add_argument("--single-op-artifact-root", type=Path, default=SINGLE_OP_ARTIFACT_ROOT)
    parser.add_argument("--e2e-artifact-root", type=Path, default=E2E_ARTIFACT_ROOT)
    parser.add_argument("--skip-ood", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-timelines", action="store_true")
    parser.add_argument("--only", choices=ONLY_CHOICES, default="all")
    return parser.parse_args()


def _section_results_to_json(results: list[SectionResult]) -> list[dict[str, object]]:
    return [{"name": result.name, "outputs": result.outputs} for result in results]


def main() -> None:
    args = parse_args()
    sections: list[SectionResult] = []

    sections.append(
        run_platform_summary(
            args.output_root,
            single_op_artifact_root=args.single_op_artifact_root,
            e2e_artifact_root=args.e2e_artifact_root,
        )
    )

    if args.only in {"all", "single_op"}:
        sections.append(
            run_single_op_core(
                args.output_root,
                single_op_artifact_root=args.single_op_artifact_root,
                baseline_model_root=None,
                e2e_artifact_root=args.e2e_artifact_root,
            )
        )
        if not args.skip_ood:
            sections.append(
                run_single_op_ood(
                    args.output_root,
                    single_op_artifact_root=args.single_op_artifact_root,
                    ood_artifact_root=None,
                )
            )
        if not args.skip_ablation:
            sections.append(
                run_single_op_ablation(
                    args.output_root,
                    ablation_artifact_root=ABLATION_ARTIFACT_ROOT,
                )
            )

    if args.only in {"all", "e2e"}:
        sections.append(
            run_e2e_core(
                args.output_root,
                e2e_artifact_root=args.e2e_artifact_root,
            )
        )
        sections.append(
            run_e2e_sum_baseline(
                args.output_root,
                single_op_artifact_root=args.single_op_artifact_root,
            )
        )
        if not args.skip_timelines:
            sections.append(
                run_timeline_cases(
                    args.output_root,
                    single_op_artifact_root=args.single_op_artifact_root,
                )
            )

    sections.append(build_figures_catalog(args.output_root))
    sections.append(build_chapter4_draft(args.output_root))

    manifest_path = build_run_manifest(
        output_root=args.output_root,
        sections=sections,
        single_op_artifact_root=args.single_op_artifact_root,
        e2e_artifact_root=args.e2e_artifact_root,
    )

    payload = {
        "manifest_path": str(manifest_path),
        "sections": _section_results_to_json(sections),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
