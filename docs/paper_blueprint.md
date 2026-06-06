# HTER Paper Blueprint

## Working Title

**Heterogeneous Disagreement-Aware Evidence Routing for Multimodal Sentiment Analysis**

Chinese working title:

**面向多模态情感分析的异质冲突感知证据路由**

## Core Thesis

Existing global-local multimodal sentiment models can already extract strong semantic and behavioral evidence. The remaining problem is not only how to extract evidence, but how to allocate evidence when modalities disagree.

HTER studies **heterogeneous disagreement-induced evidence allocation mismatch**:

- Some disagreement is noise-like: audio/visual cues are weak or unstable, so global semantic evidence should remain dominant.
- Some disagreement is text-outlier-like: text diverges from nonverbal cues, so local behavioral evidence should be preserved.
- Low-disagreement samples should not be over-corrected.

Therefore, HTER keeps a strong dual-evidence backbone and adds bounded disagreement-aware calibration and routing.

## Architecture

### Input Encoders

The model receives three aligned modality sequences:

- Text: BERT-based token representation.
- Audio: pre-extracted acoustic descriptors.
- Visual: pre-extracted facial/visual descriptors.

HTER does not require raw audio or raw video. The optional A/V-TCN module refines pre-extracted audio and visual sequences before fusion.

### SAE: Semantic Anchor Evidence

**Motivation.** Text is the strongest modality in MOSI/MOSEI, but it should not be blindly trusted when nonverbal evidence disagrees. The global path should be a semantic anchor, not a pure text shortcut.

**Mechanism.**

1. Obtain global semantic evidence `g` from the global path.
2. Pool text, audio, and visual modality summaries.
3. Build an A/V-supported semantic signal:

```text
semantic_input = [text, average(audio, visual), |text - average(audio, visual)|]
semantic_support = tanh(MLP(semantic_input))
g' = g * (1 + scale_sae * semantic_support)
```

**Interpretation.** SAE asks whether nonverbal context supports the text-centered semantic evidence. It does not replace the global path; it calibrates the global evidence with a small residual budget.

**Ablation expectation.** Removing SAE should weaken global semantic calibration and slightly reduce binary/multi-class performance, especially when text and nonverbal signals are not perfectly aligned.

### LBE: Local Behavioral Evidence

**Motivation.** Local evidence should preserve nonverbal behavioral cues. In many samples, audio/visual information is weak, but when it is useful, it should not be diluted by text-dominant fusion.

**Mechanism.**

1. Obtain local evidence `l` from the local path.
2. Build a pure nonverbal behavior signal without text:

```text
behavior_input = [audio, visual, |audio - visual|]
behavior_residual = tanh(MLP(behavior_input))
l' = l + scale_lbe * behavior_residual
```

**Interpretation.** LBE is not another text gate. It explicitly reserves an A/V-only channel to compensate for local behavioral evidence that may be weakened during cross-modal fusion.

**Ablation expectation.** Removing LBE should reduce the model's ability to exploit nonverbal cues. This is most visible in high-disagreement or text-outlier-like samples.

### HDE-RER: Heterogeneous Disagreement Estimation and Regime-Aware Evidence Routing

**Motivation.** A single global-local gate treats all disagreement similarly. HTER instead estimates the topology of disagreement and applies a bounded gate shift.

**Observation vector.**

HTER uses modality summaries from global and local paths and constructs:

```text
v_hde = [
  text, audio, visual,
  |text - audio|, |text - visual|, |audio - visual|,
  text * audio, text * visual, audio * visual
]
```

This is not a hard rule that says cosine distance equals conflict. It is a learnable disagreement observation vector.

**Routing.**

1. A base gate first predicts the original global/local evidence ratio:

```text
alpha_base = softmax(MLP([g', l']))
```

2. HDE encodes the disagreement observation:

```text
h = MLP(v_hde)
delta = budget * tanh(Linear(h))
```

3. RER updates the gate in a constrained way:

```text
alpha_hter = softmax(log(alpha_base) + delta)
```

**Interpretation.** RER does not overwrite the backbone gate. It only shifts the evidence allocation within a small budget. This is important because the global-local backbone is already strong.

**Ablation expectation.**

- Base gate: tests whether normal global-local fusion is enough.
- HDE-only / SAE-only / LBE-only: tests whether each evidence cue is individually useful.
- Full HTER: tests whether semantic calibration, behavior preservation, and bounded routing work together.

### A/V-TCN: Nonverbal Temporal Refinement

**Motivation.** Audio and visual features are pre-extracted sequences. A lightweight temporal module can refine local nonverbal dynamics without needing raw audio/video.

**Mechanism.**

```text
a' = a + s_a * TCN_a(a)
v' = v + s_v * TCN_v(v)
```

The scale is small, so the module starts close to the original representation. In the paper, A/V-TCN can be described as an input-level nonverbal temporal refinement inside the LBE line.

## Recommended Experiment Tables

### Main Comparison

Main table should report MOSI and MOSEI. SIMS is not included unless later tuning improves it.

Use the strongest available numbers:

- MOSI: Non0 Acc-2 87.20.
- MOSEI: Non0 Acc-2 86.38.

When comparing to DPDF-LQ and related methods, clearly state the encoder and selection protocol.

### Main Ablation

| Variant | What It Proves |
|---|---|
| Full HTER | Complete SAE + LBE + HDE-RER + A/V-TCN |
| Base gate | Normal global-local fusion is not enough |
| HDE only | Disagreement-aware routing contributes but is not sufficient alone |
| SAE only | Semantic anchor calibration contributes |
| LBE only | Nonverbal behavior preservation contributes |
| SAE + LBE, w/o HDE-RER | Evidence calibration without routing |
| w/o A/V-TCN | Tests whether temporal nonverbal refinement is useful |

Current MOSI ablation evidence:

| Variant | Non0 Acc-2 | F1 | Acc-7 | MAE | Corr |
|---|---:|---:|---:|---:|---:|
| Full HTER + A/V-TCN | 87.20 | 87.01 | 48.54 | 0.6948 | 0.8002 |
| Base gate | 85.06 | 84.97 | 47.96 | 0.7163 | 0.7849 |
| HDE only | 85.67 | 85.57 | 48.10 | 0.7132 | 0.7940 |
| SAE only | 86.13 | 85.92 | 48.83 | 0.7062 | 0.7986 |
| LBE only | 86.43 | 86.24 | 47.67 | 0.7059 | 0.8007 |
| SAE + LBE, w/o HDE-RER | 86.43 | 86.31 | 48.69 | 0.6886 | 0.8064 |

This table supports the claim that each component helps, but the full model gives the best binary result.

## Writing Position

Do not claim that HTER invents the entire global-local backbone. A safer and stronger position is:

> HTER builds on a strong dual-path global-local evidence space and studies how to calibrate and route semantic and behavioral evidence under heterogeneous cross-modal disagreement.

This makes the contribution boundary clear:

- Backbone: dual-path evidence extraction.
- Our focus: semantic evidence calibration, behavioral evidence preservation, and bounded disagreement-aware evidence routing.

## Why This Is Worth Studying

MSA is not only about higher benchmark scores. It studies how language, speech, and visual behavior jointly express affect. In real applications, modalities often disagree because of noise, weak nonverbal signals, sarcasm, hesitation, or mismatched facial/voice cues. A model that explicitly diagnoses disagreement and controls evidence allocation is more interpretable than a black-box fusion gate.

HTER's value is therefore:

1. It gives a concrete mechanism for "when to trust text and when to preserve nonverbal evidence".
2. It keeps the backbone stable with bounded routing instead of aggressive feature rewriting.
3. It provides analyzable intermediate quantities: gate shift, route weights, and evidence calibration strength.
