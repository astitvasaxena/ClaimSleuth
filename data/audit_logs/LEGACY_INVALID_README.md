# Legacy audit ledgers — INVALID hash chains (preserved as evidence)

The four `*_audit_trail.jsonl` files in this folder were committed in `850d757`.
They were written by the original `AuditLogger`, which restarted the hash chain at
`GENESIS` every time a new logger was created for an existing claim (defect D-3).
**Their hash chains do not verify.** They are kept byte-for-byte unchanged as
evidence of the defect and of the pre-fix runs; they must not be presented as a
valid tamper-evident record.

Status measured on 27 Oct 2026 with `AuditLogger.verify_chain()` (read-only):

| File | Lines | Entries restarting at GENESIS | verify_chain() |
|---|---|---|---|
| CLM-2024-001_audit_trail.jsonl | 1,826 | 113 | invalid: hash mismatch at entry 4 |
| CLM-2024-002_audit_trail.jsonl | 538 | 77 | invalid: hash mismatch at entry 4 |
| CLM-2024-003_audit_trail.jsonl | 964 | 71 | invalid: hash mismatch at entry 4 |
| CLM-CONFLICT-DATE_audit_trail.jsonl | 26 | 25 | invalid: hash mismatch at entry 2 |

SHA-256 of the files as preserved (working copy, 27 Oct 2026):

```
b60a68225d72fd2571cc9d1338f337aae95131975c8676fcbc74e31fa6e98f7b  CLM-2024-001_audit_trail.jsonl
3d1c790c95bb9064f5d05da5e0d32750ca9c7b99cb94b2a65cec7ab560735855  CLM-2024-002_audit_trail.jsonl
ad6b988b221abd8cb0ece2c7827442ac7d338f6838b56de8c61044f7d9782a8e  CLM-2024-003_audit_trail.jsonl
ef065d5ed1757906515cf9686b3c5d202ebe9ec87ae725da3516b9451ee1077e  CLM-CONFLICT-DATE_audit_trail.jsonl
```

The application no longer reads or writes this folder. New ledgers are written by
the fixed logger to `data/runtime/audit_logs/` (gitignored), or to
`CLAIMSLEUTH_AUDIT_LOG_DIR` if set. `GET /api/audit-legacy` re-runs the
verification on these files read-only and reports the result.
