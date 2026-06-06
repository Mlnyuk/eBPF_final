"""Offline-trained runtime policy layer (Policy v0).

The detector pipeline is::

    Isolation Forest -> Noise Filter -> Policy Filter -> final_action

The Policy Filter decides *how to escalate* an anomaly that survived the noise
filter (wait / triage / alert), using a small cost-sensitive DecisionTree trained
offline on labelled fault-injection data. No LLM and no RL run at detection time —
the tree only estimates P(fault); the action is a deterministic expected-cost rule.
"""
