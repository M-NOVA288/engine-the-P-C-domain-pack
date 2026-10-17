# Deploying to Azure

Target architecture (not provisioned by this repo yet; next milestone is Bicep):

| Component | Azure service |
|---|---|
| API (`api/main.py`) | Azure Container Apps, behind API Management |
| Memo narrative | Azure OpenAI (via Azure AI Foundry), private endpoint |
| Guideline retrieval | Azure AI Search, hybrid keyword + vector index |
| Document parsing | Azure AI Document Intelligence (layout or custom ACORD model) |
| Case store and audit log | Cosmos DB or Azure SQL; audit stream to Log Analytics |
| Portfolio analytics | Microsoft Fabric Lakehouse + Power BI over eval and decision tables |
| Secrets | Key Vault with managed identity, no keys in code |
| CI/CD | GitHub Actions: tests and eval gate, then container build and deploy |

## Minimal manual deploy

```bash
az group create -n rdc-demo -l southcentralus
az containerapp up -n risk-decision-copilot -g rdc-demo \
  --source . --ingress external --target-port 8000 \
  --env-vars COPILOT_LLM=local COPILOT_RETRIEVER=local
```

Add a Dockerfile with `uvicorn api.main:app --host 0.0.0.0 --port 8000` as the
command. Switch `COPILOT_LLM` and `COPILOT_RETRIEVER` to `azure` once the
OpenAI and Search resources exist and their settings are in Key Vault.
