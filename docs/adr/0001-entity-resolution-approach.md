# Entity Resolution: Fuzzy Matching + Human Review

**Status:** accepted

**Date:** 2026-01-01

**Decision Maker(s):** Daniel Zivkovic (with Claude Code assistance)

## Context

After extracting 1052 contacts from 740 Claude conversations using Gemini 3 Flash ($4.45), we faced entity resolution challenges:

| Problem | Example | Impact |
|---------|---------|--------|
| Name variants | "Daniel" / "Daniel Z." / "Daniel Zivkovic" | ~50-80 duplicate clusters |
| Relationship explosion | 4764 chars of concatenated roles | Unreadable data |
| Celebrity contamination | "Gordon Ramsay" mixed with real contacts | False positives |
| Category errors | Business contacts in "Family" | Miscategorization |

**Constraint:** Solution must align with project philosophy - "less is more", human-in-the-loop, zero additional API cost.

## Decision

Implement **Fuzzy Matching (Jaro-Winkler) + Human Review** via `scripts/deduplicate_contacts.py`.

**Parameters chosen:**
- Fuzzy threshold: 0.88 (balances precision/recall)
- Auto-merge threshold: User-configurable (default: none, all human-reviewed)
- Relationship limit: Top 5 items per contact
- Last name typo tolerance: 0.80 Jaro-Winkler similarity

**What we built:**
1. Celebrity detection (context-aware removal)
2. Self-reference detection (conservative, avoids false positives)
3. LinkedIn URL exact matching
4. Fuzzy name matching with human review CLI
5. Relationship field deduplication and priority sorting
6. Category validation with rule-based correction

## Consequences

### Positive

- **$0 cost** - All local processing, no API calls
- **Auditable** - Every decision logged, human confirms merges
- **Effective** - 17 auto-removed (celebrities + self), 262 candidates for review
- **Quality** - Relationship fields: 4764 → 219 chars max
- **Generic** - Works for any user via `--user-name` / `--user-last-name` flags

### Negative

- **Manual effort** - 262 candidates require ~15-30 min human review
- **Not fully automated** - By design (human-in-the-loop philosophy)
- **Threshold sensitivity** - 0.88 may miss some edge cases or include false positives

## Alternatives Considered

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| **SBERT Embeddings** | Semantic context distinguishes same-name different-people | +50 LOC, new dependency, non-deterministic | Deferred |
| **Graph Transitive Closure** | Auto-merges A=C when A=B and B=C confirmed | Complexity, order-dependent results | Deferred |
| **Dedupe Library (ML)** | Sophisticated ML-based matching | Overkill for ~1000 contacts, training overhead | Rejected |
| **LLM-based Resolution** | Nuanced understanding | $5-50 cost, non-deterministic, slow | Rejected |
| **Manual Review Only** | Perfect accuracy | 1000+ contacts = hours of work | Rejected |

**Why alternatives were deferred, not rejected:** SBERT and Graph TC could add value for larger datasets (10K+ contacts). Current solution handles 1000 contacts efficiently. Can revisit if scale increases.

## Testing Results

Three critical bugs caught during defensive testing:

1. **Self-detection false positives** - "Daniel Stone" (different person) was being removed
   - Fix: Conservative matching, requires both name match AND self-indicator in relationship

2. **Last name typo edge cases** - "Zikovic" (typo) wasn't caught
   - Fix: Jaro-Winkler similarity at 0.80 threshold for last names

3. **Sequential label false positives** - "Speaker_01" matched "Speaker_02" at 0.96
   - Fix: Added `is_sequential_label()` filter

## Research References

- [work/2025-12-31/02-entity-resolution-plan.md](../../work/2025-12-31/02-entity-resolution-plan.md) - Initial research
- [work/2026-01-01/01-entity-resolution-testing.md](../../work/2026-01-01/01-entity-resolution-testing.md) - Testing notes
- [work/2026-01-01/03-implementation-review-ultrathink.md](../../work/2026-01-01/03-implementation-review-ultrathink.md) - Full analysis
- [Fuzzy Matching Algorithms](https://tilores.io/fuzzy-matching-algorithms) - Algorithm selection
- [Entity Matching using LLMs](https://arxiv.org/pdf/2310.11244) - Why we didn't use LLMs

## Notes

**"Should I Stay or Should I Go?"** - We stopped here because:
1. All 6 planned phases implemented and tested
2. Solution handles current scale (1000 contacts) efficiently
3. Adding SBERT/Graph TC would violate "less is more" for marginal gain
4. Human review (262 candidates, ~20 min) is the intended design, not a limitation

**Next steps** (if needed):
- Run interactive review: `python scripts/deduplicate_contacts.py private/contacts-clean.json -o private/contacts-final`
- Future: Consider SBERT if dataset grows to 10K+ contacts
