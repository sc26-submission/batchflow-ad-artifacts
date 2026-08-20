# BatchFlow Deployment

BatchFlow uses a **topology** file to describe where the coordinator and workers run.

Each machine running BatchFlow is represented as a node. The launcher uses `node_id` to determine what should run on the current machine.

## Local Deployment

The local topology runs the coordinator and workers on the same machine.

```yaml
# config/topology/local.yaml

nodes:
  local:
    host: 127.0.0.1
    worker_count: 4
    worker_port_start: 60061

coordinator:
  node: local
  port: 50051

redis:
  host: 127.0.0.1
  port: 6379
  ssl: false

startup_timeout_seconds: 30.0
```

Start BatchFlow with:

```bash
python -m batchflow.deployment.launch_batchflow
```

The default configuration uses:

```text
topology=local
node_id=local
```

This starts:

```text
Coordinator
+ 4 workers
```

## AWS / Multi-Node Deployment

A topology can contain multiple machines.

For example:

```yaml
# config/topology/aws.yaml

nodes:
  node-0:
    host: 10.0.1.10
    worker_count: 8
    worker_port_start: 60061

  node-1:
    host: 10.0.2.10
    worker_count: 24
    worker_port_start: 61061

coordinator:
  node: node-0
  port: 50051

redis:
  host: batchflow-cache.example.amazonaws.com
  port: 6379
  ssl: false

startup_timeout_seconds: 60.0
```

Here:

* `node-0` runs the coordinator and 8 workers.
* `node-1` runs 24 workers.
* All workers connect to the coordinator at `10.0.1.10:50051`.
* Redis is shared between the coordinator and workers.

On `node-0`, run:

```bash
python -m batchflow.deployment.launch_batchflow topology=aws node_id=node-0
```

On `node-1`, run:

```bash
python -m batchflow.deployment.launch_batchflow topology=aws node_id=node-1
```

The launcher automatically determines whether the current node should start the coordinator, workers, or both.

## Configuration

The main configuration selects the default topology and node:

```yaml
defaults:
  - _self_
  - dataset: cifar10
  - topology: local
  - policy: full

node_id: local
```

The different configuration groups have separate responsibilities:

```text
dataset   → what data BatchFlow serves
policy    → scheduling, caching, and reuse behaviour
topology  → where the coordinator, workers, and Redis are located
node_id   → which topology node is being launched on this machine
```

There is no separate `co-located` or `disaggregated` mode. The topology itself determines the deployment.

For example, a node with both the coordinator and workers is naturally co-located, while workers on another node are naturally disaggregated.

## Redis

The topology only specifies how Redis can be reached:

```yaml
redis:
  host: 127.0.0.1
  port: 6379
  ssl: false
```

Whether BatchFlow actually uses caching is controlled by the selected policy:

```yaml
cache_enabled: true
```

For local development, Redis can be started with:

```bash
sudo systemctl start redis-server
```

and checked with:

```bash
redis-cli ping
```

which should return:

```text
PONG
```
