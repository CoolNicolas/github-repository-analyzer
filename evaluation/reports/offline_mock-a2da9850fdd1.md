# Evaluation summary: github-ai-agent-phase5a-2026-11

- Mode: `offline_mock`
- Ground-truth commit: `a2da9850fdd1fe8a9bb4ae016a30b3f662f90fa2`
- Cases: 13

## Offline pipeline reliability

| Mode | Runs | Completed | Sufficiency match | Model calls | Tool calls | Unique chunks | Duplicate chunks | Est. tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| analysis | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 8000 |
| rag | 6 | 6 | 6 | 6 | 0 | 60 | 0 | 46805 |
| agent | 6 | 6 | 6 | 12 | 6 | 18 | 0 | 21797 |

## Citation integrity and mechanical relevance

Relevance excludes cases with no expected sources (shown as N/A in the denominator). Integrity checks that citations resolve only to request-supplied evidence.

| Mode | Runs | Integrity passed | Relevant / applicable |
|---|---:|---:|---:|
| analysis | 1 | 1/1 | 1/1 |
| rag | 6 | 6/6 | 3/5 |
| agent | 6 | 6/6 | 3/5 |

## Actual LLM accuracy

Not measured. Human review is pending; offline mocks do not establish model accuracy.

## Failure and termination categories

- Error categories: agent_insufficient_evidence (1)
- Termination reasons: insufficient_repository_evidence (1)
Correctness requires human review against each benchmark rubric. Citation and retrieval metrics are mechanical checks against pinned references; they do not establish semantic correctness.
