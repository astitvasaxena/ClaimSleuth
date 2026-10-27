# RAG in ClaimSleuth

Retrieval-augmented generation (RAG) gives the investigation the **policy wording** behind each
decision, with real source references. It adds evidence and caution. It never decides eligibility
or money: those stay with the deterministic rules, the VerifierAgent and the human reviewer.

## Architecture

```
data/rag/policies/*.md ──chunk by "## " section──► embed (all-MiniLM-L6-v2, 384-d) ──► data/rag/index/
claim dossier (*.txt)  ──chunk by 6 lines────────► embed at query time (that claim only)
                                                         │
ADK pipeline:  Intake → Evidence → PolicyRetrievalAgent → Anomaly → Warranty → Settlement → Verifier → Report
                                         │ bb:rag                                              ▲
                                         └─ citations + stated terms ─────── cross-check ──────┘
```

| Part | File | What it does |
|---|---|---|
| Corpus | `src/rag/corpus.py` | Generates one wording document per policy from `configs/policies.json`, plus claims-handling guidelines. Chunks Markdown by section and claim documents by line range. Skips binary, empty, non-UTF-8 and unsupported files and reports them. |
| Retriever | `src/rag/retriever.py` | Embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2` (offline, from the local cache), saves an exact cosine index (NumPy) and returns the top-k passages above a minimum score. It filters to the claim's own policy and documents, and scans every passage with the prompt-injection guardrail. |
| Policy evidence | `src/rag/policy_evidence.py` | Asks fixed questions (coverage period, deductible, limit, depreciation, exclusions for the incident cause, required documents, and claim support). It keeps a **citation** for each topic (file, section, chunk id, score, verbatim quote) and parses the numbers the wording **states**. |
| ADK agent | `src/adk_app/steps.py` `policy_retrieval_agent` | Runs in every pattern: after EvidenceAgent (sequential), as a third branch beside Anomaly/Warranty (parallel), and as a `retrieval` route chosen by ClaimManager (hierarchical). It writes `bb:rag`. |
| Verifier check | `steps.verifier_agent` + `check_policy_citations` | Compares the stated terms with the structured terms the settlement used. |
| Model tool | `src/adk_app/tools.py` `search_policy_documents` | In LLM mode the investigator can search the policy. Results are marked `untrusted_evidence`, and quarantined passages are never returned. |
| Report | `FinalReport.rag`, `policy_citations`, `retrieved_evidence` | The Markdown report has a "Policy evidence retrieved (RAG)" table. The dashboard status line shows the citation count, and hovering it shows the sources. |

**Why no FAISS:** it is not installed, and the corpus has 25 chunks. An exact NumPy dot product
gives the same results with no approximation, and it is easier to explain. For thousands of
documents, FAISS or a vector database would replace `vectors.npy` and nothing else would change.

**Why this embedding model:** it was already in the local Hugging Face cache, it is small (90 MB,
CPU-only) and widely used. Nothing is downloaded at run time (`HF_HUB_OFFLINE=1`,
`local_files_only=True`). If it cannot load, a TF-IDF retriever (scikit-learn) is used, and every
result is labelled with it (`retriever: "tfidf:scikit-learn (fallback)"`).

## Safety rules

| Situation | What happens | Resulting status (iPhone sample) |
|---|---|---|
| Wording matches the structured terms | Citations are added to the report | unchanged: $251 proposed, pending human approval |
| Wording states a different term (e.g. deductible $129) | Conflict `Policy wording (…) states deductible=129.0 …`, flag `POLICY_WORDING_MISMATCH` | `CONFLICT_ESCALATED`, not payable |
| Passage contains instruction-like text ("ignore previous instructions…") | The passage is quarantined (kept in the record but never used), flag `RAG_PASSAGE_QUARANTINED` | `CONFLICT_ESCALATED`, not payable |
| Retrieval fails, corpus is missing, or a term has no wording | Discrepancy `Missing policy wording citation`, flag `RAG_CITATION_MISSING` | `NEEDS_EVIDENCE`, not payable |
| Direct messaging drops PolicyRetrievalAgent → Verifier | The verifier treats the missing evidence as a failure | `NEEDS_EVIDENCE`, not payable |
| Off-topic question | Empty result with a warning, never an invented passage | n/a |

RAG can only **block** a payout, never create or change one. The settlement amount always comes
from `math_engine` using `configs/policies.json`, which is checked by the test
`test_tampered_wording_escalates_and_never_changes_the_money`. Rejections (for example an expired
warranty) are not affected by retrieval failures.

## How to run

```powershell
python -m src.rag.retriever --rebuild                       # regenerate wording + rebuild the index
python -m src.rag.retriever --query "What deductible applies?" --policy POL-TECH-CARE-01
python -m src.rag.evaluate                                  # retrieval metrics (embedding vs TF-IDF)
python -m pytest tests/test_rag.py -q                       # RAG tests
```

RAG is on by default in the ADK engine. `run_adk_investigation(..., rag=False)` turns it off.
Settings (environment variables): `CLAIMSLEUTH_RAG_CORPUS_DIR` (default `data/rag/policies`),
`CLAIMSLEUTH_RAG_INDEX_DIR` (default `data/rag/index`, gitignored), and
`CLAIMSLEUTH_RAG_RETRIEVER=tfidf` to force the fallback.

**Rebuilding the index:** you rarely need to do this by hand. The index stores a SHA-256 hash of
the chunked corpus, and the retriever rebuilds it automatically when the hash changes, the files
are missing or unreadable, or the retriever type is different. After editing
`configs/policies.json`, run `--rebuild` so the wording documents are regenerated too. If the
wording is not regenerated, the verifier will (correctly) report a mismatch.

**First-time setup on a new machine:** the model must be in the Hugging Face cache once. While
online, run
`python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"`.
Without it, ClaimSleuth still runs on the labelled TF-IDF fallback.

## Evaluation results (measured 2026-10-27, this machine)

**Retrieval** (`python -m src.rag.evaluate`): 34 questions written for the evaluation, separately
from the 24 development questions used to choose the score threshold. There are 10 questions per
policy × 3 policies, plus 4 guideline questions and 5 off-topic questions.

| Retriever | hit@1 | hit@3 | MRR | Off-topic rejected |
|---|---|---|---|---|
| Embedding (all-MiniLM-L6-v2) | 0.53 | **1.00** | 0.75 | 5/5 |
| TF-IDF fallback | 0.82 | 0.82 | 0.82 | 5/5 |

- **Embedding rank-1 misses:** 16 of 34. Most lose to section "7. Settlement and approval",
  which mentions deductible, limit and depreciation in one sentence. The expected section was
  still always in the top 3. The agent cites the first top-3 passage *from the matching section*,
  so the pipeline's citations are correct (hit@3 is the number that matters here).
- **TF-IDF:** it is better at rank 1 when the words overlap, but it finds nothing for paraphrases
  such as "How much do I pay myself before the insurer pays?" (deductible). That is why it is only
  the fallback.
- **The questions the pipeline asks:** the four money terms rank 1 for all three policies. The
  required-documents question ranks 2nd for `POL-LAPTOP-STD-02` (short section).

**Pipeline** (`tests/test_rag.py`, 23 tests):
- The three samples keep their outcomes in sequential, parallel and hierarchical patterns, with 6
  citations each: iPhone $251 `PROPOSED_PENDING_APPROVAL`, laptop `NOT_PAYABLE`, fridge
  `CONFLICT_ESCALATED`.
- Every citation quote is found verbatim in the cited file and section.
- Cost: loading the model takes about 9 s once per process; retrieval takes about 135–175 ms per claim.

**Regression benchmark** (`python -m src.eval.benchmark --splits dev,val`, RAG on, deterministic planner):
384/384 runs correct (96 synthetic claims × 4 patterns), with 0 unsafe proposals and 0 payments
authorized without a human, the same as before RAG. PolicyRetrievalAgent's mean latency is
286 ms (sequential) and 174 ms (parallel branch). The held-out test split was **not** rerun.

## Limitations

- The corpus is small and **synthetic**: the wording is generated from the same JSON the rules use.
  RAG here proves the evidence chain (retrieve → cite → cross-check). It does not prove that it
  understands real insurance contracts.
- Only Markdown/text is supported. PDFs and scanned documents are skipped and reported, not read
  (no OCR). Claim documents have no page numbers, so citations use line ranges.
- The term cross-check uses regular expressions on the expected sentence pattern. Reworded
  documents produce "missing citation" (fail closed), not a parsed value.
- The exclusions text is cited but not compared: exclusion decisions remain with `policy_rules`.
- The 0.15 score threshold was tuned on this corpus. A different corpus needs re-measuring.
- Injection detection uses the existing regex guardrail. Paraphrased instructions that avoid its
  patterns would not be quarantined. They still cannot change the money, but they would appear as
  evidence text.
- These evaluation questions have now been used once. Any further tuning should use a new set.
