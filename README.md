# BatchFlow

BatchFlow is a distributed data-loading and batch-orchestration service for
concurrent machine-learning training jobs.

## Configuration layout

Each kind of setting has one owner:

```text
batchflow/config/config_types.py
    Canonical BatchFlow runtime defaults.

batchflow/config/dataset/
    Dataset location, batch size, transforms, shuffle, and metadata.

batchflow/config/deployment/
    Where BatchFlow runs: coordinator/worker hosts, ports, and Redis endpoint.

batchflow/config/policy/
    How BatchFlow behaves: static allocation, dynamic coordination, and reuse.

experiments/config/workload/
    Concurrent jobs, models, training batches, and learning rates.

experiments/config/system/
    Trainer-side settings for each evaluated system.

experiments/config/ablation/
    Small metadata configs that map each ablation stage to a deployment/policy.

experiments/config/pricing/
    Fixed analysis-time cloud prices and resource profiles.
```

The important separation is:

```text
deployment = where services run
policy     = how BatchFlow schedules/reuses batches
system     = which training data-loading system is evaluated
```

## Local validation

Run from the repository root.

```bash
python examples/pytorch_smoke_test.py
```

For a local multi-job synthetic run:

```bash
python examples/run_local_multi_pytorch_jobs.py --num-jobs 4 --num-workers 2
```

These examples do not require S3 or Redis.

## Launching BatchFlow

The default is a colocated deployment with the full BatchFlow policy:

```bash
python -m batchflow.deployment.launch_batchflow dataset=imagenet
```

The policy can be selected independently:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=colocated \
  policy=static \
  dataset=imagenet
```

Available policies are:

```text
static        fixed worker partitions, no cross-job reuse
coordination  shared/dynamic worker pool, no cross-job reuse
full          shared/dynamic worker pool with reuse/cache enabled
```

### AWS disaggregated deployment

Edit the addresses in:

```text
batchflow/config/deployment/aws_disaggregated.yaml
```

On the training machine, launch the coordinator:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=full \
  role=coordinator \
  dataset=imagenet
```

On the same training machine, launch its local workers in another terminal:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=full \
  role=worker \
  worker_host_id=training-node \
  dataset=imagenet
```

On the data-worker machine:

```bash
python -m batchflow.deployment.launch_batchflow \
  deployment=aws_disaggregated \
  policy=full \
  role=worker \
  worker_host_id=data-node \
  dataset=imagenet
```

`aws_disaggregated.yaml` contains the Redis/ElastiCache endpoint, but the
launcher connects to Redis only when the selected policy has caching enabled.
Thus `policy=static` and `policy=coordination` do not use Redis even though the
same deployment topology is reused.

## Running experiments

Example W1 commands:

```bash
python -m experiments.run_experiment \
  system=pytorch \
  workload=imagenet_4j_mixed

python -m experiments.run_experiment \
  system=batchflow \
  workload=imagenet_4j_mixed
```

The same runner supports W2 and W3:

```bash
python -m experiments.run_experiment system=pytorch workload=openimages_4j_mixed
python -m experiments.run_experiment system=pytorch workload=coco_4j_albef
```

For BatchFlow, launch the service first with the workload's dataset config.

Each training job runs in a separate process. With `device: cuda`, job 0 uses
`cuda:0`, job 1 uses `cuda:1`, and so on.

Results are written to:

```text
exp_results/<workload>/<system>/<run_id>/
├── resolved_config.yaml
├── per_batch_metrics_<job>.csv
├── job_summary.csv
└── aggregate_summary.csv
```

See `experiments/README.md` for W1--W3 details, the ablation workflow, and
analysis commands.

## Shared cache

Reusable BatchFlow payloads use Redis/ElastiCache when both conditions are
true:

```text
deployment.redis.enabled = true
policy.cache_enabled      = true
```

If the selected policy disables caching, the launcher disables the Redis
connection for that run. Transient payloads continue to use worker-local memory
and the worker gRPC endpoint.

## Dataset preparation

See:

```text
datasets/prepare_datasets.md
```

for S3 layout and preparation instructions for the experiment datasets.
