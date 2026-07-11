from __future__ import annotations

from sophons.evals.runner import EvalRun


def render_report(run: EvalRun) -> str:
    """Render an EvalRun as a plain-text report: the two headline rates,
    per-dimension averages, and every failure with its reason."""
    lines = [
        f"dataset: {run.dataset_name} ({run.dataset_version})   "
        f"cases: {len(run.case_results)}   trials per case: {run.num_trials}",
        "",
        f"pass rate (per trial):  {run.pass_rate:.0%}",
        f"pass^{run.num_trials} (per case):    {run.pass_hat_k:.0%}",
        "",
    ]

    averages = run.dimension_averages()
    if averages:
        lines.append("dimension averages:")
        for dimension, average in averages.items():
            lines.append(f"  {dimension:<14} {average:.2f}")
        lines.append("")

    failures = run.failures()
    if failures:
        lines.append(f"failures ({len(failures)}):")
        for case, trial, score in failures:
            lines.append(
                f"  [{case.id} trial {trial.trial}] {score.dimension}: "
                f"{score.reason}"
            )
    else:
        lines.append("failures: none")

    return "\n".join(lines)
