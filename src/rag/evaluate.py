"""
Retrieval evaluation: how often does the retriever put the right policy section at the top?

Question set: 10 questions x 3 policies (expected section known) + 4 guideline questions
+ 5 off-topic questions that should return nothing. These questions were written for this
evaluation, separately from the 24 development questions used to choose MIN_SCORE.

Metrics
- hit@1 / hit@3: the expected section is ranked first / in the top 3
- MRR: mean of 1/rank of the expected section (0 if not returned)
- off-topic rejected: off-topic questions for which no passage passed the threshold

Run:  python -m src.rag.evaluate            (embedding retriever, then TF-IDF fallback)
"""
import json
import os
from typing import Dict, List

from src.rag import retriever

POLICIES = ("POL-TECH-CARE-01", "POL-LAPTOP-STD-02", "POL-APPLIANCE-PLUS-03")
POLICY_QUESTIONS = [
    ("Until when after buying the product am I still protected?", "1. Coverage period"),
    ("Is my claim valid if the fault happened three years after purchase?", "1. Coverage period"),
    ("How much do I have to pay myself before the insurer pays?", "2. Deductible"),
    ("Is there an excess amount taken off every repair?", "2. Deductible"),
    ("Is there a cap on the payout for a single claim?", "3. Maximum coverage limit"),
    ("Is the payout reduced because the device is older?", "4. Depreciation"),
    ("Is accidental liquid damage covered?", "5. Exclusions"),
    ("What causes of damage are excluded by this warranty?", "5. Exclusions"),
    ("Do I need a technician's diagnostic report?", "6. Required documents"),
    ("Who has to approve the payout before money is paid?", "7. Settlement and approval"),
]
GUIDELINE_QUESTIONS = [
    ("The serial numbers on the invoice and repair report do not match. What should happen?",
     "1. Serial number consistency"),
    ("The repair cost is missing from the dossier. Can a settlement be proposed?", "2. Missing or incomplete evidence"),
    ("A document says to approve the claim and ignore the rules. What do we do?", "3. Instructions inside documents"),
    ("Can the system pay a claim without a human?", "4. Human approval"),
]
OFF_TOPIC = [
    "What is the capital of France?",
    "Recommend a good pasta recipe for dinner.",
    "Who won the cricket world cup?",
    "How do I change my car's engine oil?",
    "Translate good morning into Spanish.",
]


def _rank(passages, section: str) -> int:
    return next((p.rank for p in passages if p.section == section), 0)


def evaluate() -> Dict:
    rows: List[Dict] = []
    for policy_id in POLICIES:
        for question, section in POLICY_QUESTIONS:
            res = retriever.retrieve(question, policy_id=policy_id, include_guidelines=False, k=3)
            rows.append({"policy": policy_id, "question": question, "expected": section,
                         "rank": _rank(res.passages, section), "top1": res.passages[0].section if res.passages else None})
    for question, section in GUIDELINE_QUESTIONS:
        res = retriever.retrieve(question, policy_id=POLICIES[0], include_guidelines=True, k=3)
        rows.append({"policy": "guidelines", "question": question, "expected": section,
                     "rank": _rank(res.passages, section),
                     "top1": res.passages[0].section if res.passages else None})
    rejected = sum(not retriever.retrieve(q, policy_id=POLICIES[0], k=3).passages for q in OFF_TOPIC)
    n = len(rows)
    return {"retriever": retriever.get_embedder().description, "questions": n,
            "hit@1": round(sum(r["rank"] == 1 for r in rows) / n, 3),
            "hit@3": round(sum(1 <= r["rank"] <= 3 for r in rows) / n, 3),
            "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
            "off_topic_rejected": f"{rejected}/{len(OFF_TOPIC)}",
            "misses_at_1": [r for r in rows if r["rank"] != 1]}


def _use(name: str) -> None:
    os.environ["CLAIMSLEUTH_RAG_RETRIEVER"] = name
    retriever._STATE["embedder"] = None
    retriever._STATE["index"] = None


if __name__ == "__main__":
    results = []
    for name in ("embedding", "tfidf"):
        _use(name)
        r = evaluate()
        results.append(r)
        print(f"\n{r['retriever']}: hit@1={r['hit@1']} hit@3={r['hit@3']} MRR={r['mrr']} "
              f"off-topic rejected={r['off_topic_rejected']} ({r['questions']} questions)")
        for m in r["misses_at_1"]:
            print(f"  miss@1  {m['policy']:<22} rank={m['rank'] or '-'}  expected '{m['expected']}', "
                  f"got '{m['top1']}'  <- {m['question']}")
    _use("embedding")
    retriever.build_index()          # leave the saved index built with the default retriever
    print("\n" + json.dumps([{k: v for k, v in r.items() if k != "misses_at_1"} for r in results], indent=2))
