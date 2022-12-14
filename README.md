# opcc-baselines

Code for baselines presented in research paper "Offline Policy Comparison with Confidence:
Baseline and Benchmarks"

## Installation

#### 1. Setup [opcc](https://github.com/opcciclr/opcc)
   _(We recommend familarizing with [opcc usage](https://github.com/opcciclr/opcc#usage) before using baselines)_

#### 2. Python dependencies could be installed using:

```bash
git clone https://github.com/opcciclr/opcc-baselines.git
cd opcc-baselines
python3 -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage

| Required Arguments                                         | Description         |
|:-----------------------------------------------------------|:--------------------|
| `--job {train-dynamics,evaluate-queries,uncertainty-test}` | Job to be performed |

| Optional Arguments     | Description                                            |
|:-----------------------|:-------------------------------------------------------|
| `--no-cuda`            | no cuda usage (default: False)                         |
| `--result-dir`         | directory to store results (default: pwd)              |
| `--wandb-project-name` | name of the wandb project (default: opcc-baselines)    |
| `--use-wandb`          | use Weight and bias visualization lib (default: False) |

#### Train dynamics:

Example: 
```bash
python main.py --job train-dynamics --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```

| Optional Arguments                              | Description                                                                    |
|:------------------------------------------------|:-------------------------------------------------------------------------------|
| `--env-name`                                    | name of the environment (default: HalfCheetah-v2)                              |
| `--dataset-name`                                | name of the dataset (default: random)                                          |
| `--dynamics-type {feed-forward,autoregressive}` | type of dynamics model (default: feed-forward)                                 |
| `--deterministic`                               | if True, we use deterministic model otherwise<br/> stochastic (default: False) |
| `--dynamics-seed`                               | seed for training dynamics (default: 0)                                        |
| `--log-interval`                                | log interval for training dynamics (default: 1)                                |
| `--dynamics-checkpoint-interval`                | update interval to save dynamics checkpoint (default:1)                        |
| `--hidden-size`                                 | hidden size for Linear Layers (default: 200)                                   |
| `--update-count`                                | total batch update count for training (default: 100)                           |
| `--dynamics-batch-size`                         | batch size for Dynamics Learning (default: 256)                                |
| `--dynamics-checkpoint-interval`                | update interval to save dynamics checkpoint (default:1)                        |
| `--reward-loss-coeff`                           | reward loss coefficient for training (default: 1)                              |
| `--observation-loss-coeff`                      | obs. loss coefficient for training (default: 1)                                |
| `----num-ensemble`                              | number of dynamics for ensemble (default: 1)                                   |
| `--constant-prior-scale`                        | scale for constant priors (default: 0)                                         |


#### Query Evaluation:

Example:
- Restoring dynamics locally:
```bash
python main.py --job evaluate-queries --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```
- Restoring dynamics from wandb(if used in train-dynamics phase):
```bash
python main.py --job evaluate-queries --restore-dynamics-from-wandb --wandb-dynamics-run-path <username>/<project-name>/<run-id>
```

| Optional Arguments              | Description                                                                                             |
|:--------------------------------|:--------------------------------------------------------------------------------------------------------|
| `--restore-dynamics-from-wandb` | restore model from wandb run (default: False)                                                           |
| `--wandb-dynamics-run-path `      | wandb run id if restoring model (default: None)                                                         |
| `--mixture`                     | If enabled, randomly select ensemble models at<br/> each step of query evaluation<br/> (default: False) |
| `--eval-runs`                   | run count for each query evaluation (default: 1)                                                        |
| `--eval-batch-size`             | batch size for query evaluation (default: 128)                                                          |
| `--clip-obs`                    | clip the observation space with bounds for <br/>query evaluation (default: False)                       |
| `--clip-reward`                 | clip the reward with dataset bounds for <br/>query evaluation (default: False)                          |

#### Uncertainty-test :


Example:
- Restoring query evaluation data locally:
```bash
python main.py --job uncertainty-test --env-name d4rl:maze2d-open-v0 --dataset 1m --num-ensemble 10
```
- Restoring query evaluation data from wandb(If used in query-evaluation phase):
```bash
python main.py --job uncertainty-test --restore-query-eval-data-from-wandb --wandb-query-eval-data-run-path <username>/<project-name>/<run-id>
```

| Optional Arguments                                                                                                      | Description                                           |
|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| `--uncertainty-test-type`<br/>`{paired-confidence-interval,`<br/>`unpaired-confidence-interval,`<br/>`ensemble-voting}` | type of uncertainty test  (default:ensemble-voting)   |
| `--restore-query-eval-data-from-wandb`                                                                                  | get query evaluation data from wandb (default: False) |
| `--wandb-query-eval-data-run-path`                                                                                        | wandb run id having query eval data (default: None)   |

