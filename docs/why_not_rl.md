# Why this is NOT reinforcement learning

The triage layer chooses one of `suppress | wait | triage | alert` for each
anomaly event. It is tempting to frame this as RL ("learn a policy that maximises
reward"). That framing is **wrong for this problem**. This document records why,
and the conditions under which RL would become justified.

## 1. The problem is non-MDP

An MDP requires that the action change the environment state and that future
observations depend on the action taken. Here:

* `suppress / wait / triage / alert` are **alert-routing** decisions. They notify
  (or don't notify) a human. **They do not change the cluster state.** The next
  window's syscall rate, disk latency, and TCP retransmits are identical whether
  we paged or stayed silent.
* With no state transition, there is **no sequential credit assignment** to
  learn. There is nothing for `Q(s, a)` or a policy gradient to bootstrap from.

So DQN / PPO / Q-learning have no MDP to optimise. At most this is a
**contextual bandit** (one-shot decision per context, no transitions).

## 2. There is no delayed-reward loop

RL/bandits earn their keep when you **cannot get labels** but **can observe the
outcome of an action** (delayed reward) and must explore to learn. We are not in
that regime:

* There is no wired feedback signal — no operator clicking "good page / bad page",
  no auto-resolution outcome feeding back a reward.
* Exploration would mean **deliberately taking the wrong action** on live alerts
  to gather reward signal. That is operationally unacceptable for a paging system.

## 3. Offline labels are available → supervised is strictly better

We have offline labels from three sources (see `src/triage/label_store.py`):

* `fault_metadata` — injected faults with known type/severity (ground truth),
* `llm_adjudicator` — high-quality LLM verdicts on ambiguous cases,
* `weak_rule` — rule-based first-pass labels + hard-negative mining.

When labels exist, a **cost-sensitive supervised classifier** directly minimises
expected operational cost (see `src/triage/cost_matrix.py`) with **zero
exploration regret and lower variance** than any bandit/RL estimator. It also
decouples the cost grid from the model: retune costs without retraining.

## 4. When RL *would* become justified

Re-evaluate only when **both** of these change:

1. **Delayed reward replaces labels.** If explicit labels disappear and the only
   signal becomes post-hoc action feedback (operator disposition, auto-resolution
   outcome) that cannot be turned into a label any other way → a **contextual
   bandit** becomes reasonable.
2. **Closed-loop remediation is added.** If actions start to *change the cluster*
   — pod restart, throttle, scale, cordon, evict — then the action affects the
   next observation and a real **MDP** exists. That is when full RL (and only
   then) is on the table.

Until then: **cost-sensitive supervised learning, first.**

See also: `docs/triage_pipeline.md`, `docs/heterogeneous_gpu_label_factory.md`.
