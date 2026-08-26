# AI Development Log

## Purpose

This log records how AI tools were used during development of the Intain FinTech Challenge 2026 project.

The log is maintained throughout development rather than reconstructed at the end.

## Entries

| 2026-08-26 | Explainability | ChatGPT | Designed grounded loan-level explanations from model risk evidence. | Evidence-based reasons and explicit model-only signalling. | Accepted | Adjusted wording/tests to ensure explanations remain grounded. | Avoid unsupported borrower claims. | ~80% | Explainability must remain tied to observable evidence. |
| 2026-08-26 | LLM Copilot | ChatGPT | Designed a reviewer prompt using structured risk context. | Grounded reviewer context and anti-hallucination constraints. | Accepted | Validated prompt with unit tests. | Prevent unsupported financial/borrower claims. | ~80% | LLM should explain model output, not invent facts. |
| 2026-08-26 | Dashboard | ChatGPT | Designed a Streamlit reviewer dashboard. | Risk review interface with model outputs and explanations. | Accepted | Tested locally and verified frontend execution. | Provide decision-ready reviewer access. | ~70% | Dashboard should expose model reasoning clearly. |
| 2026-08-26 | Transition Modeling | ChatGPT | Designed next-state prediction across CURRENT, DELINQUENT, DEFAULT, PREPAID, CLOSED. | Multiclass transition model. | Accepted | Validated predictions and probabilities. | Capture loan lifecycle transitions. | ~80% | Macro F1 is important because state frequencies are imbalanced. |
| 2026-08-26 | Anomaly Intelligence | ChatGPT | Designed unsupervised anomaly detection. | Isolation Forest with normalized anomaly scores. | Accepted | Added tests and integrated into submission. | Surface unusual loan observations without fabricated causes. | ~80% | Anomaly scores should be treated as signals, not explanations. |
| 2026-08-26 | Submission Generation | ChatGPT | Designed final challenge submission pipeline. | 12-column submission generated from model outputs. | Accepted | Validated 17,392 rows and zero actual NaNs. | Produce reproducible challenge-ready output. | ~80% | Validate schema, row count, duplicates and probability ranges before submission. |

## Review Principles

- AI-generated suggestions are reviewed by the developer.
- Incorrect or unsuitable suggestions are documented rather than hidden.
- AI output does not replace ML evaluation or human judgment.
- Results are never fabricated.