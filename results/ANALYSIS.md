# eBPF 이상 탐지 — Fault 강화 실험 + 모델 평가 (A+B)

생성: 2026-06-01 · 모델: `models/isolation_forest.pkl`

> **갱신 (C 섹션 참조)**: cpu_utilization(14번째 feature) 추가 + 하이브리드 IF+z-tail
> 탐지로 **5/5 fault 전부 100% recall** 달성. cpu-stress 탐지 불가 → 해결됨.
> 아래 A/B는 13-feature 모델 기준 초기 분석(맥락 보존), 최종 결과는 **C 섹션**.

## A. 강화된 Fault Injection

v1 약점(bash 루프 조기 종료, curl 성공으로 retransmit 0)을 stress-ng/fio 300초 연속 부하로 교체.
각 fault를 전용 노드에 고정, collector CSV에서 fault pod row만 추출해 라벨링.

라벨 데이터: `results/labeled/fault_<type>.csv` (positive), `normal_sample.csv` (4124 negative).

### Fault별 feature 프로필 (평균)

| Fault | 노드 | 지배 feature (실측 평균) |
|-------|------|----------------------|
| cpu-stress | worker-1 | context_switch=**236** (정상 7376보다 낮음), 나머지 0 |
| fork-bomb | worker-2 | process_fork=**18046** (정상 22의 824×), context_switch=37080 |
| disk-stress | worker-3 | disk_read=**1.20GB**, disk_write=65MB, context_switch=294038 |
| network-flood | infra-1 | network_rx=**13.7GB**, tx=13.6GB, syscall_open=26104 |
| syscall-flood | infra-2 | syscall_open=**5101**, read=3061, write=2039, ctxsw=27989 (순수 syscall) |

## B. 모델 평가 (정상 4124 vs fault 119)

```
ROC AUC = 0.7539
PR  AUC = 0.5129
```

### Threshold 비교

| Threshold | 값 | Recall(TPR) | FPR | Precision |
|-----------|-----|------|-----|-----------|
| 현재(모델) | 0.6230 | 0.563 | **0.013** | 0.549 |
| Youden-J | 0.3309 | **0.773** | 0.091 | 0.197 |
| max-F1 | 0.6596 | 0.563 | 0.007 | **0.705** |

max F1 = 0.626 @ 0.6596

### Fault별 탐지율(recall)

| Fault | @0.623(현재) | @0.331(Youden) | 비고 |
|-------|------|------|------|
| disk-stress | **1.00** | 1.00 | 완벽 |
| network-flood | **1.00** | 1.00 | 완벽 |
| syscall-flood | **1.00** | 1.00 | 완벽 (순수 syscall 신호) |
| fork-bomb | **0.00** | **1.00** | threshold 민감 (score ~0.4) |
| cpu-stress | 0.00 | 0.00 | **탐지 불가** |

## 핵심 발견

1. **cpu-stress는 현 feature set으로 본질적 탐지 불가.** stress-ng `--cpu matrixprod`는
   순수 유저스페이스 FP 연산 → syscall/IO/network 0, CPU-bound라 context switch도
   정상 이하(236 < 7376). 13개 eBPF feature(syscall/net/disk/sched-switch)는 커널
   이벤트 기반이라 순수 연산 부하를 못 봄. → **CPU utilization feature 추가 필요**
   (cgroup cpuacct usage 또는 sched runtime delta).

2. **fork-bomb는 threshold 민감.** fork_count가 824배인데도 0.62에선 미탐, 0.33에서
   100%. IsolationForest가 fork 축을 과소 격리. → 임계값 하향 또는 fork-rate 전용 룰.

3. **disk/network는 near-perfect** (해당 차원 GB 단위 폭증).

4. **Trade-off**: 0.623 유지 시 FPR 1.3%(오탐 적음)지만 fork 미탐. 0.331은 fork 잡지만
   FPR 9.1%. 운영 권장: **0.62 유지 + fork/exec rate 보조 룰**.

## SHAP Feature Attribution

`detector/explain.py` (shap.TreeExplainer) — fault별 최대 기여 feature:

| Fault | Top SHAP driver (Δ vs normal) |
|-------|------------------------------|
| fork-bomb | **process_fork_count (+3.96)**, context_switch (+0.95) |
| disk-stress | **disk_write_bytes (+3.35)**, disk_read (+2.54), ctxsw (+2.53) |
| network-flood | **syscall_open_rate (+2.90)**, network_tx (+2.72), rx (+2.67) |
| syscall-flood | **syscall_open_rate (+2.07)**, disk_write (+2.14), process_fork (+1.62) |
| cpu-stress | 전부 음수 (정상 이하) → 탐지 불가 재확인 |

SHAP가 각 fault의 원인 feature를 정확히 지목 → 라이브 API의 z-score proxy보다 해석력 우수.
단 200-tree IF에 per-request로는 느려 **오프라인 분석 전용**(`requirements-analysis.txt`).

## 산출물

- `results/labeled/` — 라벨 feature 데이터
- `results/threshold_tuning_*.txt` — 전체 메트릭
- `results/roc_pr_points.csv` — ROC/PR 곡선 점 (plot용)
- `results/shap_attribution.csv` — feature별 SHAP 값
- `detector/threshold_tune.py`, `detector/explain.py` — 평가 도구

## 다음 단계

- [x] CPU utilization feature 추가 → cpu-stress 탐지 + 재학습 (→ C 섹션)
- [x] fork/exec rate 보조 룰 또는 threshold 재보정 (→ C 섹션 하이브리드 z-rule이 해결)
- [x] syscall-flood 순수 syscall 데이터로 재캡처 (stress-ng OOM → 시간제한 bash 루프, mem~0, recall 0.78→1.00)

---

## C. CPU feature + 하이브리드 탐지 (14-feature, 최종)

A/B에서 드러난 cpu-stress 탐지 불가를 해결.

### 1) cpu_utilization feature 추가 (커널 신호 확보)
sched_switch에서 per-CPU 마지막 전환 ts → 다음 전환 delta = 나간 cgroup on-CPU 시간 적립.
`cpu_utilization = on-CPU ns / wall-clock ns` (1코어 분율; 4코어 pegging = ~4.0).
→ cpu-stress가 cpu_utilization=**4.0** (train mean 0.012, **z=210σ**)로 드디어 관측됨.

### 2) 핵심 발견: feature 추가만으론 불충분
IsolationForest는 **단일축 이상에 약함**. cpu-stress는 14축 중 cpu 1개만 이상,
나머지 13축은 idle pod처럼 보임(학습셋에 흔함) → 200트리 중 cpu로 일찍 분기하는
트리만 격리 → 평균 path 길어짐 → **IF score 0.255 (정상 판정)**. fork-bomb도 동일.

```
            cpu_utilization    IF score   IF 판정
cpu-stress  4.0 (z=210σ)       0.255      정상 ❌
```

### 3) 해법: 하이브리드 (IF + z-tail 사이드룰)
`is_anomaly = IF_score ≥ threshold  OR  max(per-feature z) ≥ sigma_threshold(10)`
- IF = 다변량/상관 이상 담당
- z-tail = 단일축 극단 담당 (IF 맹점)

분리도 (max-z):
```
정상 pod:  max 7.3 (p99.9=6.2)      ← 임계 10 아래
전 fault:  cpu 202, fork 403, syscall 1.3만, net 12만, disk 90만  ← 전부 10 위
```

### 4) 재학습 (clean baseline)
- 30분 × 9노드 = 52,423 window 수집 → **node/system idle 행 제거** (cpu_util 최대 80 →
  포함하면 "high cpu=normal" 학습) → **33,296 pod 행**으로 학습.

### 5) 최종 결과

| 지표 | 13-feat (B) | **14-feat 하이브리드 (C)** |
|------|------|------|
| ROC AUC | 0.754 | **0.948** |
| cpu-stress recall | 0.00 | **1.00** (z-rule) |
| fork-bomb recall | 0.00 | **1.00** (z-rule) |
| disk/network/syscall | 1.00 | **1.00** (both) |
| 정상 FP | 1.3% | **1.3%** (전부 IF발, z-rule FP=0) |

z-tail 룰은 **오탐 0 추가**하면서 IF 맹점 2개(cpu/fork)를 메움 = 순이득.

### 6) Live 검증 (배포된 서비스)
```
POST /detect  cpu_utilization=4.0, 나머지 idle
 → is_anomaly=True, score=0.255, trigger=sigma, top=[cpu_utilization]
POST /detect  cpu_utilization=0.02 (idle)
 → is_anomaly=False, trigger=none
```
배포 detector(14-feat, hybrid)가 IF 단독으론 놓칠 cpu-stress를 실시간 탐지 확인.

### 산출물 (C)
- `models/isolation_forest.pkl` — 14-feat + sigma_threshold=10
- `results/labeled14/` — 14-feature 라벨 데이터 (cpu-stress 포함)
- `results/threshold_tuning_14feat.txt`
- `scripts/harvest_baseline.sh` — baseline 수집
- 코드: `detect.py` 하이브리드, `model_utils.py` sigma_threshold

### 남은 한계
- z-tail 룰은 학습 분포 기반 → 합법적 부하 급증(GPU batch 등)에 FP 가능. 현 baseline은
  9노드 전체라 어느정도 커버하나, 워크로드 다양성 늘면 재보정 필요.
- 대안 모델 ECOD/COPOD는 per-dimension tail을 native 집계 → 단일축 이상에 강함. 향후 비교.
