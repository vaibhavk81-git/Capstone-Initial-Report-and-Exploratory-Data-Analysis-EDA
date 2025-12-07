
# Adaptive Context Selection for English-Hindi Chat Translation
**Author:** Vaibhav Khare

---

## Executive Summary

This project addresses the challenge of **adaptive context selection** for machine translation of chat conversations from English to Hindi. Unlike document translation, chat translation requires understanding conversational context - pronouns, references, and topic continuity depend on previous turns.

Our EDA reveals that **about 65% of conversation turns require context** from previous messages for accurate translation (based on a heuristic proxy label). We developed a **two-stage machine learning pipeline**:
- **Stage 1:** Logistic Regression classifier to predict IF a turn needs context (~98.4% F1 score)
- **Stage 2:** Ridge Regression model to score WHICH specific history turns to include (~97.4% selection F1 at threshold 0.3)

Key findings show that **sentence-embedding similarity** (MiniLM) and **pronoun detection** are the strongest predictors of context dependency, enabling intelligent, adaptive context selection rather than fixed-window approaches.

---

## Rationale

**Why should anyone care about this question?**

1. **Cost Efficiency:** LLM-based translation APIs charge per token. Including unnecessary context wastes money; excluding necessary context produces poor translations. Adaptive selection optimizes both.

2. **Quality vs. Efficiency Trade-off:** Fixed-context approaches (e.g., always include last 3 turns) either:
   - Include irrelevant context (wasting tokens, potentially confusing the model)
   - Miss relevant context (producing ambiguous translations)

3. **Scalability:** For enterprise chat translation systems processing millions of messages, even small efficiency gains translate to significant cost savings.

4. **Translation Quality:** Pronouns like "it," "they," and "this" are common in chat but ambiguous without context. Hindi translation requires proper noun/pronoun agreement that depends on antecedents from previous turns.

---

## Research Question

**Primary Question:**  
*Can we build an ML model that adaptively selects which previous conversation turns to include as context for English-to-Hindi chat translation, optimizing the trade-off between translation quality and token efficiency?*

**Sub-questions:**
1. What features best predict whether a turn needs context from conversation history?
2. Given that context is needed, which specific history turns are most relevant?
3. Can adaptive context selection achieve comparable quality to fixed-context approaches while using fewer tokens?

---

## Data Sources

| Dataset | Description | Size | Usage |
|---------|-------------|------|-------|
| **PRESTO (English)** | Multi-domain task-oriented dialogues | ~95K conversations | Training context selector |
| **X-RiSAWOZ (EN-HI)** | Parallel English-Hindi dialogues | ~37K turns | Future: Reference-based evaluation |

**Current Implementation:** PRESTO English dataset (19,423 turns from 3,000 conversations sampled for EDA)

**Data Structure:**
- Conversation ID, Turn ID, Split (train/dev/test)
- Current turn text and role (user/assistant)
- Full conversation history as list
- Metadata: domain, linguistic phenomena

---

## Methodology

### Two-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: English conversation turn + history                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  STAGE 1: Context Needed Classifier   │
        │  Model: Logistic Regression           │
        │  Output: needs_context (0/1)          │
        └───────────────────────────────────────┘
                            │
                            ▼ (if needs_context = 1)
        ┌───────────────────────────────────────┐
        │  STAGE 2: Context Turn Selector       │
        │  Model: Ridge Regression              │
        │  Output: relevance scores per turn    │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  OUTPUT: Selected history turn indices│
        │  (turns with score ≥ threshold)       │
        └───────────────────────────────────────┘
```

### Feature Engineering (14 Features)

| Category | Features |
|----------|----------|
| **Lexical** | Sentence-embedding similarity, TF-IDF cosine similarity, Bigram overlap |
| **Linguistic** | Pronoun detection, Question type (temporal/entity/yes_no), Entity count (spaCy NER) |
| **Structural** | Turn position, Recency score, Speaker role, Speaker match, History length, Word count |

### Models Used

1. **Stage 1 - Logistic Regression:** Binary classification with balanced class weights
2. **Stage 2 - Ridge Regression:** Continuous relevance scoring (0-1 scale)

### Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Regression:** RMSE, MAE, R²
- **Selection:** F1 score for turn selection at threshold = 0.3

---

## Results

### Stage 1: Context Needed Classifier

| Metric            | Value      |
|-------------------|------------|
| Test Accuracy     | 97.97%     |
| Test Precision    | 99.76%     |
| Test Recall       | 97.12%     |
| **Test F1 Score** | **98.42%** |
| Test ROC-AUC      | 99.85%     |

**Top Predictive Features:**
1. Embedding Similarity (largest positive coefficient)
2. TF-IDF Similarity
3. Has Pronoun

### Stage 2: Context Turn Selector

| Metric            | Value      |
|-------------------|------------|
| Test RMSE         | 0.0208     |
| Test MAE          | 0.0138     |
| Test R²           | 0.9828     |
| **Selection F1**  | **97.38%** |

**Top Predictive Features:**
1. Embedding Similarity
2. Recency
3. Current Has Pronoun

### Key Findings

1. **~65% of turns need context** - The majority of chat turns benefit from including conversation history (according to our heuristic proxy label).
2. **Semantic similarity matters most** - Sentence-embedding similarity consistently dominates simpler lexical overlap features.
3. **Pronouns are strong signals** - 25% of turns contain pronouns requiring antecedent resolution
4. **Recency is important for selection** - More recent turns are generally more relevant
5. **Two-stage approach works** - Separating "if" from "which" improves interpretability and performance

### Visualizations Generated

1. `conversation_lengths.png` - Distribution of turns per conversation
2. `turn_length_distribution.png` - Word count distribution
3. `context_features.png` - Feature distributions by context need
4. `feature_correlations.png` - Correlation heatmap
5. `class_balance.png` - Target variable distribution
6. `linguistic_phenomena.png` - Special phenomena distribution

---

## Next Steps

- Model comparison: Random Forest, XGBoost vs current baseline
- Ensemble methods exploration
- Cross-validation for more robust evaluation
- Integrate LLM translation pipeline (Tower-7B or similar)
- Implement CometKiwi reference-free quality scoring
- Compare adaptive vs fixed-context translation quality

### Final Evaluation
- [ ] Measure token savings: adaptive vs fixed 3-turn context
- [ ] Error analysis: when does adaptive selection help/hurt?
- [ ] Production-ready model deployment considerations

---

## Notebooks

- [notebooks/01_eda_and_baseline.ipynb](notebooks/01_eda_and_baseline.ipynb)
  Main analysis notebook for this submission. Covers data loading, cleaning, feature engineering, EDA plots, and training/evaluating the two-stage baseline models.

- [notebooks/02_demo_adaptive_context.ipynb](notebooks/02_demo_adaptive_context.ipynb)  
  Lightweight demo notebook that loads the trained models from `outputs/models/` and shows how the system selects context turns and how many tokens it saves on a few representative conversations.

---

## Contact and Further Information

**Author:** Vaibhav Khare
**Program:** UC Berkeley ML/AI Professional Certificate  
**Course:** Capstone Project  
**Submission:** Module 20.1 - Initial Report and EDA

---
