# Innovation Positioning

This note positions HTER against DBR, TSD/TSDA, ECA, and DPDF-LQ for paper writing.

## Compared With DBR

DBR studies **shared-private branch imbalance**. Its motivation is structural:

- The shared branch can accumulate redundant and modality-biased evidence.
- Private branches can be polluted by leaked shared information.
- The final shared/private complementarity becomes weaker.

Its modules are tightly matched to this problem:

- **TSF** reduces redundancy in the shared branch.
- **AGPR** preserves discriminative private patterns with controlled cross-modal borrowing.
- **BRF** reunifies regularized shared/private branches.

HTER should learn this writing style: define one clear failure mode, then make every module answer part of it.

Our corresponding failure mode should be:

> Existing global-local evidence fusion suffers from heterogeneous disagreement-induced evidence allocation mismatch: the model uses nearly the same global/local allocation rule for noise-like, text-outlier-like, and low-conflict samples.

Our module mapping:

- **SAE**: calibrates global semantic evidence with nonverbal support.
- **LBE**: preserves audio/visual behavioral evidence that may be diluted by text-dominant fusion.
- **HDE-RER**: diagnoses disagreement topology and applies bounded evidence reallocation.
- **A/V-TCN**: strengthens the nonverbal temporal evidence available to LBE.

DBR is stronger in theoretical structure because it rebalances a full shared/private representation system. HTER is lighter, but it is more directly tied to global/local evidence allocation and is easier to integrate into a strong backbone.

## Compared With TSD / TSDA

TSD proposes tri-subspace disentanglement:

- Common subspace: globally shared evidence.
- Pairwise-shared subspaces: modality-pair evidence.
- Private subspaces: modality-specific evidence.

TSDA proposes temporal-spatial decoupling before cross-modal interaction:

- Temporal dynamics and spatial context are separated for each modality.
- Cross-modal alignment happens within matched factors.
- Gated recoupling restores task-level representation.

These papers are stronger than HTER in representation-level novelty because they explicitly reshape the feature space before fusion.

HTER's distinction:

- It does not claim to rebuild the whole representation space.
- It starts from global/local evidence and asks a different question: **when should each evidence type be trusted?**
- Its novelty is decision-level and evidence-allocation-level, not full disentanglement.

Therefore, do not write HTER as "we propose a more advanced disentanglement framework." Write it as:

> We complement disentanglement-style representation learning by studying bounded evidence allocation over a strong global-local evidence space.

## Compared With ECA

ECA studies emotional consistency and conflict. It has two important ideas:

- A unimodal network estimates emotional conflict intensity.
- Easy and hard branches are fused adaptively; the hard branch includes decoupled attention and contrastive learning.

ECA is stronger than a simple gate because the hard branch has its own conflict-processing mechanism.

HTER should borrow the writing logic but not copy the structure:

- ECA: easy vs hard conflict.
- HTER: low-disagreement, nonverbal-uncertain, and text-outlier-like evidence allocation.
- ECA handles conflict with a dedicated hard branch.
- HTER handles disagreement with bounded global/local evidence reallocation and nonverbal evidence preservation.

Important limitation:

ECA is naturally supported by CH-SIMS because it has unimodal labels. MOSI/MOSEI do not provide unimodal ground-truth sentiment labels, so HTER should avoid overclaiming that it identifies true "pragmatic conflict." Use conservative terms such as **text-outlier-like** and **nonverbal-uncertain**.

## Compared With DPDF-LQ

DPDF-LQ defines two important challenges:

- Balancing global and fine-grained sentiment contributions.
- Reducing over-reliance on text.

Its architecture already provides a strong dual-path evidence space:

- Global path: cross-modal dependency and semantic evidence.
- Local path: fine-grained local representation.
- Dynamic gate: global/local fusion.

HTER should be positioned as an extension over this evidence space:

> DPDF-LQ learns strong global and local evidence, but its allocation mechanism does not explicitly model heterogeneous disagreement topology. HTER adds semantic evidence calibration, behavioral evidence preservation, and bounded disagreement-aware routing.

This is the safest contribution boundary. Do not claim the entire global/local backbone as new.

## Recommended Paper Claim

The cleanest claim is:

> HTER is a disagreement-aware evidence allocation framework built on a strong global-local evidence backbone. It improves multimodal sentiment prediction by calibrating semantic evidence, preserving local behavioral evidence, and applying bounded routing under heterogeneous cross-modal disagreement.

## Qualitative Analyses To Add

### 1. Gate Shift Visualization

Plot:

- x-axis: disagreement score or text-outlier-like score.
- y-axis: local evidence gate weight or gate shift.

Expected story:

- Low disagreement: gate shift stays small.
- Text-outlier-like samples: local weight increases.
- Nonverbal-uncertain samples: global weight is preserved.

### 2. Conflict Bucket Analysis

Split test samples into:

- Low disagreement.
- Medium disagreement.
- High disagreement.
- Text-outlier-like top 30%.

Compare:

- Base gate.
- SAE + LBE without HDE-RER.
- Full HTER.

Expected story:

Full HTER should gain more on high-disagreement or text-outlier-like buckets than on low-disagreement buckets.

### 3. Case Study

Choose successful cases, not only worst errors:

- A low-conflict example where HTER barely shifts the gate.
- A nonverbal-uncertain example where global evidence remains dominant.
- A text-outlier-like example where local evidence receives more weight.

For each case show:

- Text snippet.
- Ground-truth sentiment.
- Base prediction vs HTER prediction.
- Global/local gate weights.
- Short explanation.

### 4. Sensitivity Analysis

For MOSEI and MOSI, plot or table:

- `hde_max_gate_shift = 0.08, 0.10, 0.12, 0.15`
- A/V-TCN enabled/disabled.

The main point is not only best score. It is to show that routing must be bounded; too small cannot adapt, too large can disturb the strong backbone.

## Practical Writing Strategy

For AAAI-style writing, HTER is lighter than DBR/TSD. The paper should not oversell as a new universal representation paradigm.

For a strong CCF-B / Q1-style paper, this framing is credible:

- Strong MOSI result.
- Competitive MOSEI result.
- Clear ablation with large drops from A/V-TCN and base gate.
- Qualitative evidence that routing behavior matches the motivation.

The paper should emphasize interpretability and controlled evidence allocation rather than claiming large universal SOTA.
