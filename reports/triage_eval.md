# Triage Evaluation Report

Events: **35**  
Class mix: `{'normal': 8, 'low': 6, 'medium': 0, 'high': 21}`

Cost-aware comparison (lower Expected Cost / High FN is better):

| Model | Accuracy | Macro F1 | Expected Cost | High FN | Alert Volume |
|---|---:|---:|---:|---:|---:|
| Threshold | 0.657 | 0.411 | 0.629 | 0 | 17 |
| Cost-sensitive Tree | 0.971 | 0.713 | 0.029 | 0 | 21 |
| RandomForest | 1.000 | 0.750 | 0.186 | 0 | 21 |
| XGBoost optional | — | — | — | — | — |

## Alert precision (P(true=high | action=alert))

- **Threshold**: 1.000
- **Cost-sensitive Tree**: 1.000
- **RandomForest**: 1.000

## Holdout mean expected cost (leave-one-group-out, fresh tree)

| Holdout | Expected Cost |
|---|---:|
| node | 0.300 |
| fault_type | 0.429 |
| time | — |

## Class confusion matrices (rows=true, cols=pred)

Class order: `['normal', 'low', 'medium', 'high']`

**Threshold**
```
        normal     low  medium    high
normal       0       4       4       0
   low       0       6       0       0
medium       0       0       0       0
  high       0       0       4      17
```

**Cost-sensitive Tree**
```
        normal     low  medium    high
normal       8       0       0       0
   low       1       5       0       0
medium       0       0       0       0
  high       0       0       0      21
```

**RandomForest**
```
        normal     low  medium    high
normal       8       0       0       0
   low       0       6       0       0
medium       0       0       0       0
  high       0       0       0      21
```

## Notes

- Expected Cost is the operational objective; accuracy is secondary.
- High FN = a `high` case routed to suppress/wait (most expensive error).
- `—` = not available (model absent or holdout group not evaluable).
