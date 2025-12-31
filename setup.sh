#!/bin/bash
# setup.sh - Quick setup for claude-mining
# ==========================================

echo "🔧 Setting up claude-mining..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install anthropic --break-system-packages 2>/dev/null || pip install anthropic

# Check for API key
echo ""
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY not set"
    echo "   For AI-powered extraction, run:"
    echo "   export ANTHROPIC_API_KEY='your-key-from-console.anthropic.com'"
else
    echo "✅ ANTHROPIC_API_KEY is set"
fi

# Remind about data export
echo ""
echo "📁 Next steps:"
echo "   1. Export your Claude data:"
echo "      → claude.ai → Settings → Privacy → Export data"
echo ""
echo "   2. Save export to a PRIVATE location (not this repo!):"
echo "      → Google Drive, iCloud, or ~/Documents"
echo ""
echo "   3. Run a script:"
echo "      python scripts/holiday_contacts.py ~/path/to/export.json"
echo ""
echo "🎄 Ready to mine your conversations!"
