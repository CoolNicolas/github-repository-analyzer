"""Command line entry point for the offline-first Phase 5A evaluator."""

import argparse
import asyncio
import csv
import json
import tempfile
from pathlib import Path

from .models import EvaluationReport, HumanReview
from .runner import load_benchmark, run_live, run_offline, summarize, write_report


def _merge_reviews(report_path: Path, review_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    for row in report["records"]:
        review = reviews.get(f"{row['case_id']}:{row['mode']}")
        if review is not None:
            row["human_review"] = HumanReview.model_validate(review).model_dump()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    parsed = EvaluationReport.model_validate(report)
    report_path.with_suffix(".md").write_text(summarize(parsed), encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parsed.records[0].__class__.model_fields))
        writer.writeheader()
        for record in parsed.records:
            row = record.model_dump(mode="json")
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, separators=(",", ":"))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the repository analyzer reproducibly.")
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/reports"))
    parser.add_argument("--context-budget", type=int, default=8_000)
    parser.add_argument("--live", action="store_true", help="Allow requests to a running API.")
    parser.add_argument(
        "--approve-paid-api",
        action="store_true",
        help="Explicitly approve that live evaluation may incur provider charges.",
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--repository-url")
    parser.add_argument("--reviews", type=Path, help="Optional JSON human-review mapping.")
    args = parser.parse_args()

    if args.approve_paid_api and not args.live:
        parser.error("--approve-paid-api is valid only together with --live.")
    if args.live and not args.approve_paid_api:
        parser.error("Live requests require both --live and --approve-paid-api.")
    if args.live and not args.repository_url:
        parser.error("--repository-url is required for live evaluation.")

    benchmark = load_benchmark(args.benchmark)
    if args.live:
        report = run_live(
            benchmark,
            api_base_url=args.api_base_url,
            repository_url=args.repository_url,
            approve_paid_api=args.approve_paid_api,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="repo-agent-evaluation-") as temporary_dir:
            report = asyncio.run(
                run_offline(
                    benchmark,
                    context_budget=args.context_budget,
                    sqlite_path=str(Path(temporary_dir) / "index.sqlite3"),
                )
            )
    json_path, markdown_path, csv_path = write_report(report, args.output_dir)
    if args.reviews:
        _merge_reviews(json_path, args.reviews)
    print(f"Evaluation mode: {report.mode}")
    print(f"JSON report: {json_path}")
    print(f"Markdown summary: {markdown_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
