# claude-mining ⛏️

**LLM-powered** extraction of insights from your Claude conversation history.

> **⚠️ PRIVACY FIRST - READ THIS:** This repo contains **scripts only**. Your exported data should stay in a private location (Google Drive, local folder) and **NEVER** be committed to this repository. All examples in this README use obviously fake data.

> **This is NOT regex matching.** Uses LLMs to **intelligently understand** relationships, context, and importance - just like a human assistant reading your conversations.

## 🎯 Use Cases

| Script | Purpose | Method |
|--------|---------|--------|
| `intelligent_contacts.py` | Extract & classify people for holiday greetings | **Multi-LLM** 🧠 |
| `holiday_contacts.py` | Basic name extraction (fallback) | Regex |
| `project_summary.py` | Summarize what you worked on (coming soon) | LLM API |
| `knowledge_graph.py` | Build a graph of people & topics (coming soon) | LLM API |

## 📊 Model Comparison (December 2025)

Contact extraction is a **needle-in-haystack** problem - finding names, relationships, and context scattered across hundreds of conversations. After extensive benchmarking, we discovered that **Gemini 3 Flash** dramatically outperforms Claude models for this specific task at a fraction of the cost.

### Benchmark Results (Same 10 Conversations)

| Model | Contacts Found | Time | Cost (10 convs) | Cost (740 convs) |
|-------|---------------|------|-----------------|------------------|
| Claude Sonnet | 7 | ~2 min | ~$1.50 | ~$111 |
| Claude Opus | 45 | ~5 min | ~$6.00 | **~$444** |
| **Gemini 3 Flash** | **58** | **74 sec** | **~$0.10** | **~$7.40** |

### Key Findings

- **Gemini 3 Flash found 29% more contacts than Opus** (58 vs 45)
- **60x cheaper** than Claude Opus ($7.40 vs $444 for full extraction)
- **4x faster** processing time
- Gemini excels at needle-in-haystack tasks (a pattern I first observed during my time at NASDAQ)

### Why Gemini Wins Here

This task requires scanning large amounts of text to find scattered mentions of people - exactly the "needle in haystack" problem Gemini models are optimized for. Claude's strength in nuanced reasoning doesn't provide much advantage when the task is fundamentally about recall and pattern detection.

### Model Pricing (December 2025)

| Model | Input | Output | Best For |
|-------|-------|--------|----------|
| Gemini 3 Flash | $0.10/1M | $0.40/1M | **Recommended** - best value |
| Gemini 3 Pro | $0.50/1M | $2.00/1M | Higher quality, still cheap |
| Claude Haiku | $0.25/1M | $1.25/1M | Not recommended for this task |
| Claude Sonnet | $3.00/1M | $15.00/1M | Good general purpose |
| Claude Opus | $15.00/1M | $75.00/1M | Best reasoning, overkill here |

## 🧠 What Makes This Different

**Regex approach (dumb):**
```
Found: "Pat", "Uma", "Mir"
```

**LLM approach (intelligent):**
```
• Alexandra Chen (TechVentures Inc) ⭐
  Relationship: Engineering Manager - key project sponsor
  Context: Collaborated on microservices migration
  Sentiment: positive

• Robert Martinez (DataFlow Systems)
  Relationship: Former colleague, now at different company
  Context: Occasional meetups to discuss industry trends
  Sentiment: neutral

• Dr. Sarah Williams (Academic Consulting)
  Relationship: Mentor - career guidance advisor
  Context: Quarterly check-ins on professional development
  Importance: high
```

## 🚀 Quick Start

### 1. Export Your Claude Data

1. Go to [claude.ai](https://claude.ai)
2. Click profile → **Settings** → **Privacy**
3. Click **Export data**
4. Save the JSON to your **private** location (Google Drive, etc.)

### 2. Clone This Repo

```bash
git clone https://github.com/dzivkovi/claude-mining.git
cd claude-mining
```

### 3. Install Dependencies & Set API Key

```bash
# For Gemini (recommended - best value)
pip install google-genai
export GOOGLE_API_KEY="your-key-from-aistudio.google.com"

# For Claude (optional - if you prefer Anthropic models)
pip install anthropic
export ANTHROPIC_API_KEY="your-key-from-console.anthropic.com"
```

### 4. Run Intelligent Extraction

```bash
# Recommended: Gemini 3 Flash (best value, ~$7 for 740 conversations)
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json -m gemini-3-flash

# Preview what will be processed (no API calls)
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json --dry-run

# Claude Opus (best quality, ~$444 for 740 conversations)
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json -m opus

# Test on a small batch first
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json --limit 10
```

### 5. Review Output

```
✅ Report saved: claude_export_contacts_report.txt
✅ JSON saved: claude_export_contacts.json
```

## 📁 Project Structure

```
claude-mining/
├── README.md                    # You are here
├── .gitignore                   # Protects your data from commits
├── setup.sh                     # Quick setup script
├── scripts/
│   ├── intelligent_contacts.py  # 🧠 LLM-powered extraction (main script)
│   ├── holiday_contacts.py      # Regex fallback (no API needed)
│   └── common.py                # Shared utilities
├── docs/
│   └── data_format.md           # Claude export format reference
└── examples/
    └── sample_output.txt        # What results look like
```

## 💡 How It Works

1. **Load** your Claude export (JSON with all conversations)
2. **Filter** conversations likely to contain contact mentions
3. **Extract** contacts one conversation at a time using tool/function calling
4. **LLM understands** context, relationships, sentiment (Gemini or Claude)
5. **Deduplicate** and categorize contacts
6. **Output** both human-readable report (.txt) and structured data (.json)

## 🔒 Security Model

```
┌─────────────────────────────────────────────────────────┐
│  PUBLIC (GitHub)              PRIVATE (Google Drive)    │
│  ─────────────────            ──────────────────────    │
│  • Python scripts             • claude_export.json      │
│  • Documentation              • Output reports          │
│  • .gitignore                 • Personal data           │
└─────────────────────────────────────────────────────────┘
```

**The `.gitignore` file prevents accidental commits of:**
- `*.json` (your export files)
- `*_contacts.txt` (output with names)
- `data/` folder
- Any file with "export" in the name

## 🛠️ Development

### Adding a New Script

1. Create `scripts/your_script.py`
2. Use shared utilities from `common.py`
3. Follow the pattern in existing scripts
4. Update this README

### Using Claude Code

```bash
# Start Claude Code in the repo
cd claude-mining
claude

# Ask Claude to help
> Add a script that extracts all the technical topics I discussed
```

## 📊 Example Output

> **Note:** The examples below use obviously fake data. Your actual output will contain contacts extracted from your conversations.

```
============================================================
🎄 INTELLIGENT CONTACT EXTRACTION REPORT 🎄
Generated: 2025-12-30 15:42
Total contacts found: 23
============================================================

## Work Colleagues (8)
  • Alexandra Chen (TechVentures Inc) ⭐
    Relationship: senior developer
    Context: Collaborated on microservices architecture
    Sentiment: positive
    Contact: alex.chen@techventures.example

  • Michael Rodriguez (DataFlow Systems)
    Relationship: project manager
    Importance: medium

## Family (4)
  • Mom ⭐
    Relationship: mother
    Importance: high

  • Jamie
    Relationship: younger sibling
    Context: Planning family reunion

## Professional Network (6)
  • Dr. Sarah Williams ⭐
    Relationship: mentor
    Context: Career guidance and technical discussions
    Importance: high

## Recruiters (3)
  • Jennifer Walsh (TalentSearch Partners)
    Relationship: technical recruiter
    Contact: jwalsh@talentsearch.example

============================================================
## 🎄 HOLIDAY GREETING LIST (High + Medium)
============================================================
  ☐ Alexandra Chen - TechVentures Inc
  ☐ Mom
  ☐ Dr. Sarah Williams
  ☐ Michael Rodriguez - DataFlow Systems
  ☐ Jamie

💡 Review and add anyone you remember!
============================================================

Total: 23 contacts across 5 categories
```

## 💻 Code Quality

This project emphasizes professional code quality:
- **Type hints** throughout for better IDE support and type safety
- **Structured logging** with configurable verbosity
- **Comprehensive error handling** for file I/O, JSON parsing, and API errors
- **Multi-provider support** - same CLI works with Gemini or Claude
- **Tool/function calling** - structured extraction, no JSON parsing failures
- **Checkpoint/resume** - interrupted runs can continue where they left off
- **Argparse CLI** with helpful flags (`-m`, `-o`, `--limit`, `--start`, `-v`)
- **Proper exit codes** for scripting and automation
- **Privacy-first design** with robust .gitignore patterns

## 🤝 Contributing

PRs welcome! Ideas for new mining scripts:
- [ ] Meeting/appointment extractor
- [ ] Code snippet collector
- [ ] Decision log builder
- [ ] Learning journal generator

## 📜 License

MIT - Do whatever you want, just don't blame me.

---

Created by [Daniel Zivkovic](https://linkedin.com/in/magmainc) / Magma Inc.
Powered by Gemini 🚀 and Claude 🤖
