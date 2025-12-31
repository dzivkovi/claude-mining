#!/usr/bin/env python3
"""
Holiday Contacts Extractor for Claude Export Data
==================================================

DEPRECATED (December 2025)
--------------------------
This regex-based fallback script is no longer recommended.
Use `intelligent_contacts.py` instead, which supports:
- Multi-provider LLM extraction (Gemini/Claude)
- Tool/function calling for zero data loss
- Gemini 3 Flash: ~$5 for 740 conversations (60x cheaper than Claude)

See README.md and ROADMAP.md for current project status.

---

Original Description:
Extracts names of people mentioned in your Claude conversations
to help build a holiday greetings list.

Author: Daniel Zivkovic / Magma Inc.
Created: December 2025

Usage:
    python extract_holiday_contacts.py path/to/claude_export.json

Requirements:
    pip install anthropic --break-system-packages

    Set your API key:
    export ANTHROPIC_API_KEY="your-key-here"
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Optional: Use Claude API for smarter extraction
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Note: Install 'anthropic' package for AI-powered extraction")
    print("      pip install anthropic --break-system-packages\n")


def load_claude_export(file_path: str) -> dict:
    """Load and parse Claude export JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_names_regex(text: str) -> set:
    """
    Extract potential names using regex patterns.
    Catches: "Hi Daniel", "Dear Mr. Smith", "@John", etc.
    """
    names = set()
    
    # Common greeting patterns
    patterns = [
        r'\b(?:Hi|Hello|Hey|Dear|Morning|Thanks?|Thank you)\s+([A-Z][a-z]+)',
        r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'@([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'\bto\s+([A-Z][a-z]+)\b',
        r'\bfrom\s+([A-Z][a-z]+)\b',
        r"([A-Z][a-z]+)'s\s+(?:email|message|reply|response)",
        r'\b([A-Z][a-z]+)\s+(?:said|mentioned|told|asked|replied|confirmed)',
        r'(?:my|our)\s+(?:wife|husband|daughter|son|friend|colleague|boss|manager)\s+([A-Z][a-z]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        names.update(matches)
    
    # Filter out common false positives
    false_positives = {
        'The', 'This', 'That', 'Here', 'There', 'What', 'When', 'Where',
        'How', 'Why', 'Can', 'Could', 'Would', 'Should', 'Will', 'May',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'January', 'February', 'March', 'April', 'June', 'July', 'August',
        'September', 'October', 'November', 'December', 'Claude', 'Anthropic',
        'Google', 'Microsoft', 'Amazon', 'Apple', 'GitHub', 'LinkedIn',
        'Thanks', 'Thank', 'Please', 'Sorry', 'Great', 'Good', 'Best',
    }
    
    return {n for n in names if n not in false_positives and len(n) > 2}


def extract_emails_and_names(text: str) -> dict:
    """Extract email addresses and associated names."""
    email_pattern = r'([A-Za-z][A-Za-z0-9._%+-]*)\s*[<(]?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})[>)]?'
    
    results = {}
    for match in re.finditer(email_pattern, text):
        name_part = match.group(1).strip()
        email = match.group(2).lower()
        
        # Clean up name
        if name_part and not name_part.startswith(('http', 'www', 'mailto')):
            # Convert email prefix to name if no name found
            if not name_part or name_part == email.split('@')[0]:
                name_part = email.split('@')[0].replace('.', ' ').replace('_', ' ').title()
            results[email] = name_part
    
    return results


def extract_with_claude_api(conversations: list, api_key: str = None) -> dict:
    """
    Use Claude API to intelligently extract and categorize contacts.
    Returns structured data about people mentioned.
    """
    if not HAS_ANTHROPIC:
        return {}
    
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Warning: No ANTHROPIC_API_KEY found. Skipping AI extraction.")
        return {}
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Combine conversation samples (limit to avoid huge API calls)
    sample_text = ""
    for conv in conversations[:50]:  # Process first 50 conversations
        if 'messages' in conv:
            for msg in conv.get('messages', [])[:10]:  # First 10 messages each
                content = msg.get('content', '')
                if isinstance(content, str):
                    sample_text += content + "\n\n"
        elif 'content' in conv:
            sample_text += str(conv['content'])[:5000] + "\n\n"
        
        if len(sample_text) > 100000:  # Cap at ~100K chars
            break
    
    if not sample_text.strip():
        return {}
    
    prompt = f"""Analyze these conversation excerpts and extract ALL people mentioned.
For each person, provide:
- Full name (as best you can determine)
- Relationship/role (colleague, family, recruiter, client, etc.)
- Company/organization if mentioned
- Any email if visible

Return as JSON array:
[{{"name": "...", "role": "...", "company": "...", "email": "..."}}]

Only include real people (not AI assistants, companies, or generic references).
Be thorough - include everyone mentioned even briefly.

Conversations:
{sample_text[:80000]}

Return ONLY the JSON array, no other text."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result_text = response.content[0].text.strip()
        
        # Parse JSON from response
        if result_text.startswith('['):
            return json.loads(result_text)
        else:
            # Try to find JSON in response
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
    
    except Exception as e:
        print(f"API extraction error: {e}")
    
    return []


def process_export(export_data: dict) -> dict:
    """Process the entire export and extract contacts."""
    all_names = set()
    all_emails = {}
    conversations_by_project = defaultdict(list)
    
    # Handle different export structures
    conversations = []
    
    if isinstance(export_data, list):
        conversations = export_data
    elif 'conversations' in export_data:
        conversations = export_data['conversations']
    elif 'chats' in export_data:
        conversations = export_data['chats']
    else:
        # Treat entire export as conversations
        conversations = [export_data]
    
    print(f"Found {len(conversations)} conversations to process...")
    
    for conv in conversations:
        # Get project info if available
        project = conv.get('project', {}).get('name', 'No Project')
        conversations_by_project[project].append(conv)
        
        # Extract from conversation title
        title = conv.get('title', '') or conv.get('name', '')
        all_names.update(extract_names_regex(title))
        
        # Extract from messages
        messages = conv.get('messages', []) or conv.get('chat_messages', [])
        for msg in messages:
            content = msg.get('content', '') or msg.get('text', '')
            if isinstance(content, str):
                all_names.update(extract_names_regex(content))
                all_emails.update(extract_emails_and_names(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        all_names.update(extract_names_regex(block['text']))
                        all_emails.update(extract_emails_and_names(block['text']))
    
    return {
        'names': all_names,
        'emails': all_emails,
        'projects': dict(conversations_by_project),
        'conversations': conversations
    }


def generate_report(extracted: dict, use_api: bool = False) -> str:
    """Generate a formatted report of extracted contacts."""
    
    report = []
    report.append("=" * 60)
    report.append("🎄 HOLIDAY CONTACTS EXTRACTION REPORT 🎄")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 60)
    report.append("")
    
    # Names found via regex
    report.append("## Names Found (Regex Extraction)")
    report.append("-" * 40)
    for name in sorted(extracted['names']):
        report.append(f"  • {name}")
    report.append(f"\nTotal: {len(extracted['names'])} names")
    report.append("")
    
    # Emails found
    if extracted['emails']:
        report.append("## Email Addresses Found")
        report.append("-" * 40)
        for email, name in sorted(extracted['emails'].items()):
            report.append(f"  • {name}: {email}")
        report.append(f"\nTotal: {len(extracted['emails'])} emails")
        report.append("")
    
    # Projects summary
    report.append("## Projects Analyzed")
    report.append("-" * 40)
    for project, convs in extracted['projects'].items():
        report.append(f"  • {project}: {len(convs)} conversations")
    report.append("")
    
    # AI extraction (if enabled)
    if use_api and HAS_ANTHROPIC:
        report.append("## AI-Powered Extraction (Claude API)")
        report.append("-" * 40)
        
        api_results = extract_with_claude_api(extracted['conversations'])
        
        if api_results:
            # Group by role
            by_role = defaultdict(list)
            for person in api_results:
                role = person.get('role', 'Unknown')
                by_role[role].append(person)
            
            for role, people in sorted(by_role.items()):
                report.append(f"\n### {role.title()}")
                for p in people:
                    name = p.get('name', 'Unknown')
                    company = p.get('company', '')
                    email = p.get('email', '')
                    
                    line = f"  • {name}"
                    if company:
                        line += f" ({company})"
                    if email:
                        line += f" - {email}"
                    report.append(line)
        else:
            report.append("  (No additional contacts found via API)")
    
    report.append("")
    report.append("=" * 60)
    report.append("💡 Tip: Review this list and add anyone you remember!")
    report.append("=" * 60)
    
    return "\n".join(report)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo input file specified!")
        print("Usage: python extract_holiday_contacts.py path/to/claude_export.json")
        print("\nTo export your Claude data:")
        print("  1. Go to claude.ai")
        print("  2. Click your profile → Settings → Privacy")
        print("  3. Click 'Export data'")
        print("  4. Download the JSON file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    use_api = '--api' in sys.argv or '-a' in sys.argv
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading export from: {input_file}")
    export_data = load_claude_export(input_file)
    
    print("Extracting contacts...")
    extracted = process_export(export_data)
    
    print("Generating report...")
    report = generate_report(extracted, use_api=use_api)
    
    # Output report
    print("\n" + report)
    
    # Save to file
    output_file = Path(input_file).stem + "_contacts.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ Report saved to: {output_file}")
    
    # Also save as JSON for further processing
    json_output = Path(input_file).stem + "_contacts.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump({
            'names': list(extracted['names']),
            'emails': extracted['emails'],
            'projects': list(extracted['projects'].keys()),
            'generated': datetime.now().isoformat()
        }, f, indent=2)
    print(f"✅ JSON data saved to: {json_output}")


if __name__ == '__main__':
    main()
