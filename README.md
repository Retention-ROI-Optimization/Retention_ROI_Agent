# Retention ROI Project

## Image

<img src="assets/dashboard.png" width="600"/>

<img src="assets/chatbot.png" width="600"/>


## Installment

```bash
pip install -r requirements.txt
```

## Docker Implementation

```bash
docker compose up --build
```

## Simulation Implementation

```bash
python src/main.py --mode simulate
```

## Feature Engineering Implementation

```bash
python src/main.py --mode features
```

Output:

- `data/feature_store/customer_features.csv`
- `data/feature_store/customer_features_metadata.json`
- `results/feature_engineering_summary.json`

## Churn modeling

```bash
python src/main.py --mode train
```

Output:

- `models/churn_model_<best_model>.joblib`
- `results/churn_auc_roc.png`
- `results/churn_precision_recall_tradeoff.png`
- `results/churn_shap_summary.png`
- `results/churn_shap_local.png`
- `results/churn_threshold_analysis.json`
- `results/churn_top10_feature_importance.json`
- `results/churn_metrics.json`

## Survival Ananlysis

```bash
python src/main.py --mode survival
```

Runs the survival analysis pipeline to estimate when each customer is likely to churn, not just whether they will churn.
This step produces time-aware churn signals such as expected churn timing, hazard-related outputs, and intervention windows that can later be integrated into the optimization and recommendation stages.

## Uplift + CLV/ Segmentation / Optimization

```bash
python src/main.py --mode uplift
python src/main.py --mode clv
python src/main.py --mode segment
python src/main.py --mode optimize --budget 50000000
```
you're allowed to write whatever budget you have in your mind.

## Personalization / Recommendation

```bash
python src/main.py --mode recommend --budget 5000000 --threshold 0.5 --max-customers 1000
```
you can change the figures (threshod, max-customers)

## AB Test

```bash
python src/main.py --mode abtest

```

## Realtime Bootstrap

```bash
python src/main.py --mode realtime-bootstrap
```
Initializes the real-time scoring environment before replaying or consuming streaming events.
This step typically prepares the required state, caches, intermediate artifacts, or message-stream resources so that the real-time pipeline can start from a consistent baseline.

## Realtime Replay

```bash
python src/main.py --mode realtime-replay --stream-limit 20000 --stream-max-events 20000
```

Replays simulated or stored customer events through the real-time pipeline so the system can update churn-risk-related outputs as if events were arriving live.

Parameter meaning:

--stream-limit 20000
Limits how many customers or records are taken into the replay process.

--stream-max-events 20000
Limits the total number of streamed events processed during replay.

## Detached Docker Run

```bash
docker compose up -d --build
```
Builds the Docker images and starts the services in detached mode, which means the containers run in the background.

Difference from:
```bash
docker compose up --build
```
up --build: runs in the foreground and shows logs directly in the terminal

up -d --build: runs in the background so you can continue using the terminal

## Implementation Order

```bash
python src/main.py --mode simulate 
python src/main.py --mode features
python src/main.py --mode train
python src/main.py --mode survival
python src/main.py --mode uplift
python src/main.py --mode clv
python src/main.py --mode segment
python src/main.py --mode optimize --budget 50000000
python src/main.py --mode recommend --budget 5000000 --threshold 0.5 --max-customers 1000
python src/main.py --mode abtest
docker compose up -d --build
python src/main.py --mode realtime-bootstrap
python src/main.py --mode realtime-replay --stream-limit 20000 --stream-max-events 20000
```

## When You Want To Reimplement

```bash
python3 src/main.py --mode simulate --force --randomize
python3 src/main.py --mode features
python3 src/main.py --mode train
python3 src/main.py --mode survival
python3 src/main.py --mode uplift
python3 src/main.py --mode clv
python3 src/main.py --mode segment
python3 src/main.py --mode optimize --budget 50000000
python3 src/main.py --mode recommend --budget 5000000 --threshold 0.5 --max-customers 1000
python3 src/main.py --mode abtest
docker compose up -d --build
python3 src/main.py --mode realtime-bootstrap
python3 src/main.py --mode realtime-replay --stream-limit 20000 --stream-max-events 20000
```

## Reimplementation Flags

```bash
python src/main.py --mode simulate --force --randomize
```

This command is used when you want to regenerate the simulation data from scratch.

Parameter meaning:

--force
Overwrites existing generated files or reruns the simulation even if prior outputs already exist.


--randomize
Generates a new randomized simulation instead of reusing the same deterministic data configuration.