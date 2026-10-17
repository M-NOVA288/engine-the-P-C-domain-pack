# Governance and responsible AI

Insurance is regulated, and many states have adopted the NAIC Model Bulletin on
the use of AI systems by insurers. It asks carriers to govern AI that affects
consumers: documented controls, testing for unfair discrimination, third-party
oversight and auditability. This project is built so those questions have answers.

## Design decisions

| Concern | How the copilot handles it |
|---|---|
| Who decides? | The copilot recommends. A licensed underwriter approves or overrides, and that action is logged (`Orchestrator.approve`). The model alone can only refer, never decline (guideline 7.1). |
| Can we explain it? | Accept/refer/decline comes from deterministic rules tied to numbered guideline sections. The risk model is a logistic regression whose per-feature contributions are printed in every memo. |
| Can the LLM make things up? | The LLM only rephrases a draft built from computed, cited findings. It is told not to add facts. Local mode uses no LLM at all, and the eval decisions are identical in both modes. |
| Can we trace it? | Every finding carries a citation to a document line or guideline section. Every step and every human decision is written to an append-only, hashed audit log. |
| What if the documents are bad? | Values that are missing, repaired from OCR, or taken from free text get low confidence and force a referral instead of a silent guess. Evals track this ("messy cases handled safely"). |
| Does quality regress? | CI regenerates the golden set, re-runs evals and blocks the build if accuracy drops or any unsafe accept or wrongful decline appears. |

## Fairness

The model uses property, loss and operational features only. It does not use
protected characteristics or obvious proxies such as owner demographics or ZIP
code. Before production, a carrier would add:

- Disparate-impact testing of referral and decline rates across geography and
  business-owner segments where data allows.
- A documented review of each feature for proxy risk.
- Monitoring of underwriter override rates by segment; a rising override rate
  is an early signal that the model or rules drifted.

## Data handling

All data in this repo is synthetic. For real submissions: keep documents in the
carrier's tenant (Azure region-pinned storage), use private endpoints for Azure
OpenAI and AI Search, redact PII before logging, and do not use customer data to
train third-party models.
