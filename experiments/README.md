# Experiments

The experiment framework keeps dataset, workload, system, BatchFlow topology,
and BatchFlow policy separate.

```text
batchflow/config/dataset/<dataset>.yaml
    Dataset location, batch size, transforms, and shuffle settings.

batchflow/config/deployment/<deployment>.yaml
    Coordinator/worker topology and optional Redis endpoint.

batchflow/config/policy/<policy>.yaml
    BatchFlow scheduling and reuse behavior.

experiments/config/workload/<workload>.yaml
    Models/jobs and training-batch settings.

experiments/config/system/<system>.yaml
    Trainer-side settings for the evaluated system.

experiments/config/ablation/<stage>.yaml
    Metadata describing the deployment/policy pair for a paper ablation stage.
```

## W1: ImageNet

Single-job sanity check:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=imagenet_1j_resnet18
```

Four-job workload:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=imagenet_4j_mixed
```

For BatchFlow, start the service with `dataset=imagenet` first, then use the
same workload with `system=batchflow`.

## W2: Open Images

W2 uses flat Open Images image objects plus image-level annotations rather than
an ImageFolder class-directory layout.

```text
s3://<bucket>/
├── train/<ImageID>.jpg
└── annotations/
    ├── train-annotations-human-imagelabels-boxable.csv
    └── class-descriptions-boxable.csv
```

The dataset builder creates one logical classification sample for each positive
`(ImageID, LabelName)` annotation. The generated canonical dataset index is
used by every system.

Single-job sanity check:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=openimages_1j_vit_b_32
```

Four-job workload:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=openimages_4j_mixed
```

For BatchFlow, launch the service with `dataset=openimages`.

## W3: COCO + ALBEF

W3 uses the ALBEF/Karpathy COCO retrieval annotations and a common image root
containing both `train2014/` and `val2014/`.

```text
s3://<bucket>/coco/
├── train2014/<image>.jpg
├── val2014/<image>.jpg
└── annotations/coco_train.json
```

The dataset builder creates one logical sample per caption and gives all
captions for the same `image_id` the same retrieval index.

Single-job sanity check:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=coco_1j_albef_2
```

Four-job workload:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=coco_4j_albef
```

For BatchFlow, launch the service with `dataset=coco`.

## Systems

The same workload can be selected with:

```text
system=pytorch
system=batchflow
system=tensorsocket
system=coordl
```

Each training job runs in its own process. With `device: cuda`, jobs are mapped
by workload order:

```text
job 0 -> cuda:0
job 1 -> cuda:1
job 2 -> cuda:2
job 3 -> cuda:3
```

## Output layout

```text
exp_results/<workload>/<system>/<run_id>/
├── resolved_config.yaml
├── per_batch_metrics_<job>.csv
├── job_summary.csv
└── aggregate_summary.csv
```

Warmup batches remain in the per-batch files but are excluded from the measured
summary statistics. `aggregate_summary.csv` also records the run-level hourly
resource cost, batches-per-dollar cost efficiency, and ablation metadata when
applicable. Cost is reported at the aggregate level because the infrastructure
is shared by all concurrent jobs.

# BatchFlow ablation

The ablation does not use separate deployment files for each stage. It composes
one of two topologies with one of three BatchFlow policies:

```text
Stage                  Deployment          Policy
-----------------------------------------------------
Colocated-Static       colocated           static
+ Disaggregation      aws_disaggregated   static
+ Coordination        aws_disaggregated   coordination
+ Caching             aws_disaggregated   full
```

The corresponding small metadata configs live under
`experiments/config/ablation/`. They are recorded in `resolved_config.yaml` so
the analysis code can keep the four stages separate.

## 1. Colocated-Static

Start BatchFlow:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=colocated \
  policy=static \
  role=all \
  dataset=imagenet
```

Run the workload:

```bash
python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed \
  ablation=colocated_static
```

`colocated.yaml` has 24 workers. `policy=static` divides each worker host evenly
across four jobs, giving six workers/job.

## 2. + Disaggregation

Edit the private addresses in
`batchflow/config/deployment/aws_disaggregated.yaml`.

Training machine, coordinator terminal:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=static \
  role=coordinator \
  dataset=imagenet
```

Training machine, worker terminal:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=static \
  role=worker \
  worker_host_id=training-node \
  dataset=imagenet
```

Data-worker machine:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=static \
  role=worker \
  worker_host_id=data-node \
  dataset=imagenet
```

Then on the training machine:

```bash
python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed \
  ablation=disaggregation
```

Each 24-worker host is divided into four six-worker partitions, so every job
receives six workers from the training host and six from the data host.

## 3. + Coordination

Use the same `aws_disaggregated` topology, changing only the BatchFlow policy:

```text
policy=coordination
```

Launch all three service roles with that policy, then run:

```bash
python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed \
  ablation=coordination
```

This policy uses one shared dynamic worker pool but keeps cross-job prepared
batch reuse disabled.

## 4. + Caching

Use the same `aws_disaggregated` topology with:

```text
policy=full
```

The Redis endpoint in `aws_disaggregated.yaml` is now used. Launch all three
service roles with `policy=full`, then run:

```bash
python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed \
  ablation=caching
```

Repeat the same four stages with the W2 and W3 workload/dataset selections.

# Cost efficiency and repeated runs

Hourly prices and reusable resource profiles are stored in:

```text
experiments/config/pricing/aws.yaml
```

The experiment reporter computes cost efficiency when the run finishes:

```text
cost efficiency = aggregate batches/s * 3600 / hourly resource cost
```

The result is written directly to `aggregate_summary.csv` together with:

```text
pricing_name
cost_resource_profile
hourly_cost_usd
cost_efficiency_batches_per_dollar
ablation_stage
ablation_label
ablation_deployment
ablation_policy
```

For normal end-to-end runs, the system config selects the resource profile.
For ablation runs, the ablation config overrides the profile where necessary.

After collecting repeated runs, use the small aggregation helper:

```bash
python -m experiments.analysis.summarize_results \
  --results-dir exp_results \
  --output-dir analysis_results
```

This writes:

```text
analysis_results/run_summary.csv
analysis_results/group_summary.csv
```

No plotting code is required by the experiment framework. The CSV outputs can
be visualized with the paper plotting workflow or any external analysis tool.

Before final artifact release, verify the fixed hourly prices in `aws.yaml`
against the exact prices used for the submitted paper results.
