# Experiment analysis

Each completed experiment already writes throughput, timing, cost, and
ablation metadata to `aggregate_summary.csv`.

The analysis helper only collects completed runs and averages repetitions. It
does not recompute pricing and it does not generate plots.

```bash
python -m experiments.analysis.summarize_results \
  --results-dir exp_results \
  --output-dir analysis_results
```

Outputs:

```text
analysis_results/run_summary.csv
analysis_results/group_summary.csv
```

Normal runs are grouped by workload and system. Ablation runs are additionally
grouped by ablation stage so the four BatchFlow stages remain separate.

Cost efficiency is computed by the experiment reporter when each run finishes:

```text
batches/$ = aggregate_batches_per_sec * 3600 / hourly_cost_usd
```

The fixed pricing inputs and reusable resource profiles are stored in
`experiments/config/pricing/aws.yaml`. These values are reporting inputs only;
they do not launch or configure AWS resources.
