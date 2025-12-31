# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**claude-mining** is an LLM-powered tool for extracting insights from Claude conversation history exports. The core innovation is using Claude API to intelligently understand relationships, context, and sentiment—not just pattern matching with regex.

### Hybrid Approach
This project combines **friendly, approachable documentation** with **professional code quality**:
- Type hints throughout for better IDE support
- Structured logging with emoji-enhanced messages
- Comprehensive error handling
- Argparse CLI with helpful flags
- Community-friendly tone with privacy-first design

### Key Privacy Constraint
This repo contains **scripts only**. User export data stays in private locations (Google Drive, local folders) and must **NEVER** be committed. The [.gitignore](.gitignore) file enforces this with patterns blocking `*.json`, `*export*`, output files, and data directories.

## Development Commands

### Setup
```bash
# Install dependencies
pip install anthropic --break-system-packages

# Set API key (required for intelligent extraction)
export ANTHROPIC_API_KEY="your-key-from-console.anthropic.com"

# Quick setup script
./setup.sh
```

### Running Scripts

```bash
# Main intelligent extraction (LLM-powered)
python scripts/intelligent_contacts.py ~/path/to/claude_export.json

# With custom output path
python scripts/intelligent_contacts.py ~/path/to/export.json -o my_contacts

# With custom batch size (for very large exports)
python scripts/intelligent_contacts.py ~/path/to/export.json --batch-size 5

# Verbose mode for detailed logging
python scripts/intelligent_contacts.py ~/path/to/export.json -v

# Fallback regex-based extraction (no API key needed)
python scripts/holiday_contacts.py ~/path/to/export.json
```

### Testing
No formal test suite exists. Test scripts manually with sample exports.

## Architecture

### Core Processing Flow

1. **Load Export** ([common.py](scripts/common.py))
   - Handles multiple Claude export formats (list, dict with 'conversations' or 'chats')
   - Normalizes structure to consistent format
   - Export structure: `{conversations: [{uuid, title, messages: []}]}`

2. **Chunk Conversations** ([intelligent_contacts.py](scripts/intelligent_contacts.py:84-93))
   - Groups 10-15 conversations per chunk to stay within API limits
   - Each chunk sent to Claude API for parallel processing

3. **Extract with LLM** ([intelligent_contacts.py](scripts/intelligent_contacts.py:96-183))
   - Sends conversation text to Claude API with structured prompt
   - Prompt asks for: name, relationship, organization, context, importance, sentiment, contact_info
   - Returns JSON with people array

4. **Merge & Deduplicate** ([intelligent_contacts.py](scripts/intelligent_contacts.py:185-221))
   - Combines results from all chunks
   - Uses normalized name matching (`name.lower().replace('.', '')`)
   - Merges data: keeps most detailed info, upgrades importance if higher found

5. **Categorize** ([intelligent_contacts.py](scripts/intelligent_contacts.py:224-248))
   - Groups by relationship keywords: Family, Work Colleagues, Recruiters, Clients, etc.
   - Uses keyword matching in relationship field

6. **Generate Report** ([intelligent_contacts.py](scripts/intelligent_contacts.py:251-331))
   - Human-readable text report with categories
   - JSON file with structured data
   - Holiday greeting checklist (high + medium importance)

### Shared Utilities ([common.py](scripts/common.py))

**Export Handling:**
- `load_claude_export()`: Normalizes different export formats
- `get_conversations()`, `get_messages()`, `get_message_content()`: Extract data
- Message content can be string OR array of content blocks (handle both)

**Claude API:**
- `get_anthropic_client()`: Initialize client if API key exists
- `call_claude()`: Wrapper with error handling
- Default model: `claude-sonnet-4-20250514`

**Regex Patterns:**
- `PATTERNS` dict: greeting_name, email, url, phone, etc.
- `FALSE_POSITIVE_NAMES`: Common words to filter (days, months, companies)
- Used by fallback regex script, not main intelligent extraction

### Message Content Structure

Claude exports have two content formats:

```python
# Simple string
message['content'] = "Hello, how are you?"

# Content blocks (code, text, etc.)
message['content'] = [
    {"type": "text", "text": "Here's the code:"},
    {"type": "code", "language": "python", "code": "print('hello')"}
]
```

Always use `get_message_content()` helper to handle both.

### API Usage Pattern

The intelligent extraction uses chunking to avoid context limits:

```python
# Process 10-15 conversations at once
chunks = chunk_conversations(conversations, chunk_size=10)

# Each chunk analyzed independently
for chunk in chunks:
    result = analyze_chunk_with_claude(client, chunk, user_context)
    all_results.append(result)

# Then merge to deduplicate
all_people = merge_people(all_results)
```

This allows processing thousands of conversations without hitting token limits.

## Adding New Scripts

When creating a new mining script (e.g., `project_summary.py`, `knowledge_graph.py`):

1. Import utilities from `common.py`:
   ```python
   from common import (
       load_claude_export, get_conversations, get_messages,
       get_message_content, get_anthropic_client, call_claude
   )
   ```

2. Follow the established pattern:
   - Load export → Chunk → Process with LLM → Merge → Generate report
   - Save both `.txt` (human-readable) and `.json` (structured data)
   - Use base filename from input: `{input_stem}_output.txt`

3. Handle API key gracefully:
   ```python
   api_key = os.environ.get('ANTHROPIC_API_KEY')
   if not api_key:
       print("ERROR: Set ANTHROPIC_API_KEY for AI extraction")
       sys.exit(1)
   ```

4. Add to [README.md](README.md) table of use cases

5. **NEVER** create example data files with real export data—use sanitized examples only in [examples/](examples/)

## Security & Privacy

**Critical Rules:**
- NEVER commit files with user data (exports, outputs, reports)
- NEVER modify [.gitignore](.gitignore) to allow data files
- NEVER create test files with real conversation data
- When adding features, preserve the privacy-first design

**Protected Patterns in .gitignore:**
- `*.json` (except package.json, tsconfig.json)
- `*export*`, `*Export*`, `*EXPORT*`
- `*_contacts.*`, `*_report.*`, `*_summary.*`
- `data/`, `exports/`, `private/`, `personal/`

## File Organization

```
scripts/
├── common.py               # Shared utilities - USE THIS
├── intelligent_contacts.py # Main LLM-powered extraction
└── holiday_contacts.py     # Regex fallback (no API)

docs/
└── data_format.md         # Claude export structure reference

examples/
└── sample_output.txt      # Sanitized example output only
```

Keep scripts in `scripts/`, documentation in `docs/`, sanitized examples in `examples/`.
