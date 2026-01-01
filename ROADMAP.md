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

### Entity Resolution & Deduplication (January 2026)

**Problem**: Raw extraction produces duplicates (name variants, celebrities, self-references) and relationship field explosion.

**Solution**: Fuzzy matching (Jaro-Winkler) + Human-in-the-loop review.

**Results** (1052 contacts):

- 8 celebrity references removed
- 9 self-references removed
- 63 category fixes applied
- Relationship fields: 4764 → 219 chars max
- 262 fuzzy candidates identified for human review

**Deliverable**: `scripts/deduplicate_contacts.py`

See [ADR-0001](docs/adr/0001-entity-resolution-approach.md) for decision rationale, alternatives considered, and "why we stopped here."

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
