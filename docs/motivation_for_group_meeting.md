# HTER Motivation and Innovation Notes for Group Meeting

## One-Sentence Motivation

HTER studies **how to allocate global semantic evidence and local behavioral evidence when text, audio, and visual modalities disagree**.

The key claim is not that we invent a new text/audio/video backbone from scratch. The key claim is:

> Strong global-local MSA backbones can extract useful evidence, but they still lack an explicit mechanism for deciding which evidence should be trusted under different disagreement patterns.

## Why This Problem Matters

In multimodal sentiment analysis, disagreement is common:

- Text can be clear while audio/visual cues are weak or noisy.
- Audio and visual behavior may contain sentiment cues that are not obvious from text.
- Some examples are easy: all modalities point in the same direction.
- Some examples are hard: modalities conflict, and a fixed fusion gate may allocate evidence incorrectly.

Existing global-local models mainly ask:

> How do we extract global and local multimodal evidence?

HTER asks a follow-up question:

> After global and local evidence are extracted, how should the model allocate them under heterogeneous disagreement?

This is the thesis of the paper.

## How To Explain the Three Modules

### SAE: Semantic Anchor Evidence

**Problem.** Text is usually the strongest modality, so global semantic evidence is powerful. But if we use text-centered evidence without calibration, the model may over-trust text.

**Idea.** Use audio/visual context to calibrate global semantic evidence.

**Implementation.**

```text
semantic_input = [text, average(audio, visual), |text - average(audio, visual)|]
semantic_support = tanh(MLP(semantic_input))
global_evidence = global_evidence * (1 + scale * semantic_support)
```

**How to say it.**

SAE does not replace the global path. It turns the global path into a **semantic anchor** whose reliability is adjusted by nonverbal support.

### LBE: Local Behavioral Evidence

**Problem.** Audio and visual cues are weak on average, but they are important in some samples. During cross-modal fusion, nonverbal behavior may be diluted by text-dominant representations.

**Idea.** Preserve a pure audio/visual behavior residual inside the local evidence branch.

**Implementation.**

```text
behavior_input = [audio, visual, |audio - visual|]
behavior_residual = tanh(MLP(behavior_input))
local_evidence = local_evidence + scale * behavior_residual
```

**How to say it.**

LBE gives the model a non-text behavioral channel. It is not a second semantic branch; it specifically protects local nonverbal evidence.

### HDE-RER: Heterogeneous Disagreement Estimation and Routing

**Problem.** A normal global-local gate treats all disagreement similarly. But different disagreement patterns require different evidence allocation.

**Idea.** Build a disagreement observation vector and apply a bounded correction to the original global-local gate.

**Implementation.**

```text
v = [
  text, audio, visual,
  |text-audio|, |text-visual|, |audio-visual|,
  text*audio, text*visual, audio*visual
]
delta = budget * tanh(MLP(v))
gate = softmax(log(base_gate) + delta)
```

**How to say it.**

HDE-RER does not overwrite the backbone. It only gives the model a small evidence allocation budget. This is important because DPDF-style global-local evidence is already strong.

## Relationship to DPDF-LQ

DPDF-LQ provides a strong global-local evidence backbone:

- Global path extracts semantic/global evidence.
- Local path extracts fine-grained/local evidence.
- Dynamic gate fuses the two.

HTER builds on this evidence space and asks a different question:

> DPDF-LQ extracts global and local evidence; HTER studies how to calibrate and route that evidence under heterogeneous disagreement.

This is the safest way to write the contribution. Do not claim the whole global-local backbone is new.

## Relationship to ECA

ECA distinguishes easy and hard/conflict samples. It also uses a conflict-specific hard branch.

HTER extends the same general motivation but uses a different formulation:

- ECA: easy vs hard conflict.
- HTER: evidence allocation under heterogeneous disagreement.
- ECA emphasizes conflict intensity.
- HTER emphasizes whether the model should trust semantic/global evidence or local/nonverbal evidence.

Important writing rule:

Because MOSI/MOSEI do not provide unimodal sentiment labels, avoid claiming that HTER fully identifies "pragmatic conflict." Use safer terms such as **text-outlier-like disagreement** and **nonverbal-uncertain disagreement**.

## Relationship to DBR and TSD

DBR and TSD are stronger representation-level papers:

- DBR studies shared/private branch imbalance.
- TSD studies common, pairwise-shared, and private subspaces.
- TSDA studies temporal-spatial decoupling.

Their contribution is deeper representation restructuring.

HTER should not pretend to be the same kind of paper. HTER is better positioned as:

> a lightweight evidence allocation framework that improves a strong global-local backbone and provides interpretable routing behavior.

This is still a valid paper story, especially with strong MOSI results and clear ablations.

## Main Experimental Story

### MOSI

MOSI is the main performance highlight.

Current best:

```text
Non0 Acc-2: 87.20
F1:         87.01
Acc-7:      48.54
MAE:        0.6948
Corr:       0.8002
```

A/V-TCN is important on MOSI:

```text
Full HTER + A/V-TCN: 87.20
w/o A/V-TCN:         84.45
```

This supports the claim that preserving nonverbal temporal evidence matters.

### MOSEI

MOSEI is the main generalization dataset.

Current best:

```text
Non0 Acc-2: 86.38
F1:         86.42
Acc-7:      53.40
MAE:        0.5292
Corr:       0.7731
```

MOSEI is larger and more stable; gains are smaller. This can be written as:

> HTER achieves strong improvements on MOSI and competitive performance on MOSEI. The smaller MOSEI gain suggests that heterogeneous disagreement is more salient in smaller/high-variance samples, while large-scale MOSEI benefits more from stable semantic evidence.

## Analyses To Add

### 1. Module Ablation

Use the MOSI table because the drops are clear:

- Full HTER.
- Base gate.
- SAE only.
- LBE only.
- HDE only.
- SAE + LBE without HDE-RER.
- w/o A/V-TCN.

### 2. Sensitivity Analysis

Show `hde_max_gate_shift`:

```text
0.08 / 0.10 / 0.12 / 0.15
```

Expected interpretation:

- Too small: model cannot adapt.
- Too large: model disturbs the strong backbone.
- A bounded budget is necessary.

### 3. Disagreement Bucket Analysis

Split samples by disagreement score:

- Low disagreement.
- Medium disagreement.
- High disagreement.
- Text-outlier-like top 30%.

Compare Full HTER vs Base Gate.

Expected conclusion:

HTER should help more in high-disagreement or text-outlier-like buckets.

### 4. Gate Visualization

Plot:

- x-axis: disagreement score.
- y-axis: local evidence gate or gate shift.

Expected conclusion:

The router does not randomly change evidence allocation; its shifts become larger when disagreement is stronger.

### 5. Case Study

Pick successful examples:

- Low conflict: HTER barely shifts the gate.
- Nonverbal uncertain: HTER preserves global semantic evidence.
- Text-outlier-like: HTER increases local/nonverbal evidence.

Do not choose only failure cases.

## How To Answer "What Is Yours?"

Backbone:

- The dual-path global-local evidence extraction backbone is inherited/inspired by DPDF-LQ-style global-local fusion.

Our contributions:

- SAE: semantic evidence calibration with nonverbal support.
- LBE: local behavioral evidence preservation.
- HDE-RER: bounded disagreement-aware evidence routing.
- A/V-TCN: nonverbal temporal refinement that strengthens LBE.
- The overall problem formulation: heterogeneous disagreement-induced evidence allocation mismatch.

## Conservative Claim

HTER is not a huge new representation-learning paradigm like DBR/TSD. It is a lightweight, interpretable evidence allocation framework.

The strongest claim is:

> HTER shows that, on top of a strong global-local MSA backbone, explicitly calibrating and routing evidence under heterogeneous cross-modal disagreement improves performance and provides interpretable behavior.
