# claude-mining Roadmap

## Completed

### Multi-Provider LLM Extraction (December 2025)

**Problem**: Extract meaningful contacts from 900+ Claude conversations without data loss.

**Solution Implemented**:
- Tool/function calling (not JSON parsing) for zero data loss
- Gemini 3 Flash integration: 29% more contacts at 60x lower cost than Claude Opus
- Checkpoint/resume for long extractions
- Per-conversation processing with incremental saves

**Results**: 740 conversations processed for $4.45, yielding ~300 raw contacts.

---

## In Progress

### Entity Resolution & Deduplication

**Problem**: Raw extraction produces duplicates and quality issues:
- Name variants: "Daniel" vs "Daniel Z." vs "Daniel Zivkovic" (same person)
- Relationship explosion: 100+ concatenated values per contact
- Celebrity contamination: "Gordon Ramsay" mixed with real contacts
- Category errors: Business contacts in "Family"

**Research Completed**: Evaluated 5 approaches for this dataset:

| Approach | Suitability | Cost | Why/Why Not |
|----------|-------------|------|-------------|
| **Fuzzy Matching (RapidFuzz)** | ⭐⭐⭐⭐⭐ | $0 | Handles name variants, typos, nicknames |
| Embedding Similarity (SBERT) | ⭐⭐⭐⭐ | $0 | Adds semantic context |
| Graph Transitive Closure | ⭐⭐⭐ | $0 | Auto-clusters after human confirms pairs |
| Dedupe Library (ML) | ⭐⭐⭐ | $0 | Overkill for ~300 contacts |
| LLM-based Resolution | ⭐⭐ | $5-50 | Expensive, non-deterministic |

**Selected Approach**: Fuzzy (Jaro-Winkler) + Human Review

- **Phase 1**: Pre-filter celebrities and exact URL matches
- **Phase 2**: Fuzzy matching at 0.88 threshold for candidate pairs
- **Phase 3**: CLI presents candidates: `[M]erge [S]kip [Q]uit`
- **Phase 4**: Relationship cleanup (dedupe, limit to top 5)
- **Phase 5**: Category validation (rule-based correction)

**Deliverable**: `scripts/deduplicate_contacts.py`

**Expected Results**:

| Metric | Before | After |
|--------|--------|-------|
| Total contacts | ~300 | ~180-200 |
| Duplicate clusters | 50-80 | 0 (merged) |
| Celebrity false positives | 10-15 | 0 (removed) |
| Relationship field | 100+ items | ≤5 items |

---

## Future Ideas

### Export to CRM
- Generate VCF (vCard) for import to Google Contacts, Apple Contacts
- CSV export with custom field mapping
- Direct API integration with popular CRMs

### Incremental Updates
- Merge new Claude exports with existing contact database
- Track "last seen" timestamps
- Highlight new vs. updated contacts

### Project Summary Extraction
- Summarize what you worked on across conversations
- Extract project names, technologies, outcomes
- Generate portfolio/resume content

### Knowledge Graph
- Build relationship graph between people and topics
- Visualize professional network
- Identify connection patterns

---

## Design Philosophy

> "Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

This project follows these principles:

1. **Less is more**: Simple, auditable solutions over complex abstractions
2. **Human-in-the-loop**: AI assists, human decides on ambiguous cases
3. **Zero data loss**: Tool calling over JSON parsing, per-item saves
4. **Cost awareness**: Gemini for bulk extraction, Claude for nuanced reasoning
5. **Privacy first**: Scripts only in repo, data stays private

---

## Research References

Entity resolution research that informed the deduplication approach:

- [Fuzzy Matching Algorithms for Data Deduplication](https://tilores.io/fuzzy-matching-algorithms)
- [Pre-trained Embeddings for Entity Resolution](https://www.vldb.org/pvldb/vol16/p2225-skoutas.pdf)
- [Entity Matching using Large Language Models](https://arxiv.org/pdf/2310.11244)
- [Can LLMs be used for Entity Resolution?](https://medium.com/tilo-tech/can-llms-be-used-for-entity-resolution-68053e357bae)
- [Dedupe Library](https://github.com/dedupeio/dedupe)
