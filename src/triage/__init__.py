"""
src.triage
==========
Cost-sensitive supervised triage pipeline (NOT reinforcement learning).

Pipeline:
    eBPF window features
      -> case_bundle  (group into a labelable unit + model votes)
      -> weak_labeler (first-pass labels from rules + fault metadata)
      -> llm_adjudicator (RTX 5090; ONLY ambiguous/conflicting cases)
      -> label_store  (multi-source labels, priority-resolved)
      -> train_cost_sensitive (tree / RandomForest / optional XGBoost)
      -> expected-cost decision over {suppress, wait, triage, alert}
      -> production detector: lightweight CPU inference (POST /triage)

Why not RL: the action does not change cluster state (non-MDP), there is no
delayed-reward loop, and offline labels are available -> supervised learning is
correct. See docs/why_not_rl.md.
"""
