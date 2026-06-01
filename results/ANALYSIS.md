# eBPF 이상 탐지 — Fault 강화 실험 + 모델 평가 (A+B)

생성: 2026-06-01 · 모델: `models/isolation_forest.pkl` (54.5k window 학습, threshold=0.6230)

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
| syscall-flood | infra-2 | syscall_open=**7648**, disk_write=87MB, disk_io_latency=high |

## B. 모델 평가 (정상 4124 vs fault 113)

```
ROC AUC = 0.7330
PR  AUC = 0.4504
```

### Threshold 비교

| Threshold | 값 | Recall(TPR) | FPR | Precision |
|-----------|-----|------|-----|-----------|
| 현재(모델) | 0.6230 | 0.522 | **0.013** | 0.518 |
| Youden-J | 0.3309 | **0.752** | 0.091 | 0.184 |
| max-F1 | 0.6596 | 0.522 | 0.007 | **0.678** |

max F1 = 0.59 @ 0.6596

### Fault별 탐지율(recall)

| Fault | @0.623(현재) | @0.331(Youden) | 비고 |
|-------|------|------|------|
| disk-stress | **1.00** | 1.00 | 완벽 |
| network-flood | **1.00** | 1.00 | 완벽 |
| syscall-flood | 0.78 | 0.89 | 양호 |
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
| syscall-flood | disk_io_latency (+2.23), disk_write (+1.80) |
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

- [ ] CPU utilization feature 추가 → cpu-stress 탐지 + 재학습
- [ ] fork/exec rate 보조 룰 또는 threshold 재보정
- [ ] syscall-flood 더 깨끗한 데이터 (이번엔 iomix가 disk도 침 → 저메모리 stressor로 교체 완료)
