# claude-mining ⛏️

**LLM-powered** extraction of insights from your Claude conversation history.

> **⚠️ PRIVACY FIRST - READ THIS:** This repo contains **scripts only**. Your exported data should stay in a private location (Google Drive, local folder) and **NEVER** be committed to this repository. All examples in this README use obviously fake data.

> **This is NOT regex matching.** This uses Claude AI to **intelligently understand** relationships, context, and importance - just like a human assistant reading your conversations.

## 🎯 Use Cases

| Script | Purpose | Method |
|--------|---------|--------|
| `intelligent_contacts.py` | Extract & classify people for holiday greetings | **Claude API** 🧠 |
| `holiday_contacts.py` | Basic name extraction (fallback) | Regex |
| `project_summary.py` | Summarize what you worked on (coming soon) | Claude API |
| `knowledge_graph.py` | Build a graph of people & topics (coming soon) | Claude API |

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
pip install anthropic --break-system-packages
export ANTHROPIC_API_KEY="your-key-from-console.anthropic.com"
```

**⚠️ API Key Required:** The intelligent extraction uses Claude API to understand context. Get your key at [console.anthropic.com](https://console.anthropic.com/)

### 4. Run Intelligent Extraction

```bash
# Main script - LLM-powered intelligent extraction
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json

# Optional: Add context about yourself for better extraction
python scripts/intelligent_contacts.py ~/GoogleDrive/claude_export.json \
  --context "I'm a software engineer working in cloud infrastructure"
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
2. **Chunk** conversations into groups of 10-15
3. **Send** each chunk to Claude API with extraction prompt
4. **Claude understands** context, relationships, sentiment
5. **Merge** results, deduplicate, categorize
6. **Output** report + JSON for further use

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
- **Argparse CLI** with helpful flags (`-o`, `--batch-size`, `-v`)
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
Powered by Claude 🤖
