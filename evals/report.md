# Eval report

**Release gate: PASS** on 60 golden cases

| Metric | Result |
|---|---|
| Decision accuracy vs. senior underwriter | 98.3% |
| Unsafe accepts (should have been referred/declined) | 0 |
| Wrongful declines | 0 |
| Over-referrals (safe, costs reviewer time) | 1 |
| Risk capture rate (problem accounts not accepted) | 100.0% |
| Field extraction accuracy | 98.0% |
| Loss ratio mean absolute error | 0.0 pts |
| Guideline retrieval hit@1 / hit@3 | 100.0% / 100.0% |
| Free-text guideline Q&A hit@1 / hit@3 (15 questions) | 66.7% / 86.7% |
| Warnings with a citation | 100.0% |
| Consistency (identical memo on re-run) | 100.0% |
| Messy-document cases handled safely | 11 of 11 |
| Pipeline latency, median / p95 | 2.4 ms / 3.9 ms |

## Misses

| Case | Truth | Copilot | Kind | Document issues |
|---|---|---|---|---|
| CASE-0030 | accept | refer | over-referral (safe) | revenue_in_remarks |
