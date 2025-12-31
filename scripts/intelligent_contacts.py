#!/usr/bin/env python3
"""
Intelligent Contact Extractor for Claude Conversation Exports

Uses Claude AI to intelligently extract and categorize contacts from
conversation exports, identifying relationships, sentiment, and context.

Author: Daniel Zivkovic / Magma Inc.
License: MIT
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 10
MAX_TOKENS = 8192

CATEGORIES = [
    "Family",
    "Work Colleagues",
    "Clients",
    "Recruiters",
    "Professional Network",
    "Financial & Real Estate",
    "Friends & Neighbors",
    "Other",
]

EXTRACTION_PROMPT = """You are an expert at extracting contact information from conversation text.

Analyze the following conversation excerpts and extract ALL people mentioned. For each person, provide:

1. **name**: Full name if available, or partial name/identifier
2. **relationship**: How this person relates to the conversation author (e.g., "manager", "friend", "recruiter", "doctor", "landlord")
3. **organization**: Company, institution, or organization they're associated with (if mentioned)
4. **context**: Brief description of how/why they were mentioned (1-2 sentences)
5. **importance**: Rate as "high", "medium", or "low" based on frequency of mention and significance
6. **sentiment**: The apparent relationship sentiment - "positive", "neutral", "complicated", or "negative"
7. **contact_info**: Any email, phone, LinkedIn, or other contact details mentioned (or null if none)

Return a JSON array of contact objects. Only include real people (not fictional characters, AI assistants, or generic references like "the doctor" without a name).

Example output format:
```json
[
  {
    "name": "Sarah Chen",
    "relationship": "manager",
    "organization": "Acme Corp",
    "context": "Discussed project deadlines and performance review",
    "importance": "high",
    "sentiment": "positive",
    "contact_info": "sarah.chen@acme.com"
  }
]
```

If no contacts are found, return an empty array: []

CONVERSATIONS TO ANALYZE:
"""


def load_claude_export(file_path: str) -> list[dict[str, Any]]:
    """Load and parse Claude export JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {file_path}")

    logger.info(f"📂 Loading export from: {file_path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different export formats
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict):
        # Try common keys for conversation arrays
        for key in ["conversations", "chats", "messages", "data"]:
            if key in data and isinstance(data[key], list):
                conversations = data[key]
                break
        else:
            # Treat the whole dict as a single conversation
            conversations = [data]
    else:
        raise ValueError(f"Unexpected export format: {type(data)}")

    logger.info(f"📊 Loaded {len(conversations)} conversation(s)")
    return conversations


def extract_conversation_text(conversations: list[dict[str, Any]]) -> list[str]:
    """Extract text content from conversations."""
    texts = []

    for conv in conversations:
        parts = []

        # Extract title/name if present
        if "name" in conv:
            parts.append(f"Conversation: {conv['name']}")
        elif "title" in conv:
            parts.append(f"Conversation: {conv['title']}")

        # Extract messages
        messages = conv.get("chat_messages", conv.get("messages", []))
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    # Handle different message formats
                    content = msg.get("content", msg.get("text", ""))
                    if isinstance(content, list):
                        # Handle content blocks
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                parts.append(block["text"])
                            elif isinstance(block, str):
                                parts.append(block)
                    elif isinstance(content, str) and content.strip():
                        parts.append(content)
                elif isinstance(msg, str):
                    parts.append(msg)

        if parts:
            texts.append("\n\n".join(parts))

    return texts


def chunk_conversations(texts: list[str], batch_size: int = BATCH_SIZE) -> list[str]:
    """Chunk conversation texts into batches for API processing."""
    chunks = []
    current_chunk = []
    current_count = 0

    for text in texts:
        current_chunk.append(text)
        current_count += 1

        if current_count >= batch_size:
            chunks.append("\n\n---\n\n".join(current_chunk))
            current_chunk = []
            current_count = 0

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n---\n\n".join(current_chunk))

    logger.info(f"🔄 Created {len(chunks)} batch(es) from {len(texts)} conversation(s)")
    return chunks


def extract_contacts_from_batch(
    client: anthropic.Anthropic, batch_text: str, batch_num: int, total_batches: int
) -> list[dict[str, Any]]:
    """Send a batch to Claude API and extract contacts."""
    logger.info(f"Processing batch {batch_num}/{total_batches}...")

    prompt = EXTRACTION_PROMPT + batch_text

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        # Parse JSON from response
        # Handle markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        contacts = json.loads(response_text)

        if not isinstance(contacts, list):
            contacts = [contacts]

        logger.info(f"  Found {len(contacts)} contact(s) in batch {batch_num}")
        return contacts

    except json.JSONDecodeError as e:
        logger.warning(f"  Failed to parse JSON from batch {batch_num}: {e}")
        return []
    except anthropic.APIError as e:
        logger.error(f"  API error in batch {batch_num}: {e}")
        return []


def normalize_name(name: str) -> str:
    """Normalize a name for deduplication."""
    return " ".join(name.lower().split())


def merge_contacts(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge two contact records, preferring more complete information."""
    merged = existing.copy()

    # Prefer longer/more complete values
    for key in ["name", "organization", "context", "contact_info"]:
        new_val = new.get(key)
        existing_val = existing.get(key)
        if new_val and (not existing_val or len(str(new_val)) > len(str(existing_val))):
            merged[key] = new_val

    # Keep higher importance
    importance_rank = {"high": 3, "medium": 2, "low": 1}
    new_imp = importance_rank.get(new.get("importance", "low"), 1)
    existing_imp = importance_rank.get(existing.get("importance", "low"), 1)
    if new_imp > existing_imp:
        merged["importance"] = new["importance"]

    # Merge relationships if different
    if new.get("relationship") and new["relationship"] != existing.get("relationship"):
        if existing.get("relationship"):
            merged["relationship"] = f"{existing['relationship']}, {new['relationship']}"
        else:
            merged["relationship"] = new["relationship"]

    return merged


def deduplicate_contacts(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate contacts by name, merging information."""
    seen: dict[str, dict[str, Any]] = {}

    for contact in contacts:
        name = contact.get("name", "").strip()
        if not name:
            continue

        normalized = normalize_name(name)

        if normalized in seen:
            seen[normalized] = merge_contacts(seen[normalized], contact)
        else:
            seen[normalized] = contact.copy()

    logger.info(
        f"🔗 Deduplicated {len(contacts)} contacts to {len(seen)} unique contact(s)"
    )
    return list(seen.values())


def categorize_contact(contact: dict[str, Any]) -> str:
    """Assign a category to a contact based on relationship and context."""
    relationship = (contact.get("relationship") or "").lower()
    organization = (contact.get("organization") or "").lower()
    context = (contact.get("context") or "").lower()

    combined = f"{relationship} {organization} {context}"

    # Family indicators
    if any(
        word in combined
        for word in [
            "mother",
            "father",
            "mom",
            "dad",
            "parent",
            "brother",
            "sister",
            "sibling",
            "son",
            "daughter",
            "child",
            "wife",
            "husband",
            "spouse",
            "partner",
            "uncle",
            "aunt",
            "cousin",
            "grandma",
            "grandpa",
            "grandmother",
            "grandfather",
            "family",
            "in-law",
        ]
    ):
        return "Family"

    # Recruiter indicators
    if any(
        word in combined
        for word in [
            "recruiter",
            "recruiting",
            "talent",
            "hiring",
            "headhunter",
            "staffing",
        ]
    ):
        return "Recruiters"

    # Client indicators
    if any(word in combined for word in ["client", "customer", "account"]):
        return "Clients"

    # Financial & Real Estate
    if any(
        word in combined
        for word in [
            "bank",
            "mortgage",
            "loan",
            "realtor",
            "real estate",
            "agent",
            "broker",
            "insurance",
            "financial",
            "advisor",
            "accountant",
            "tax",
            "landlord",
            "property",
        ]
    ):
        return "Financial & Real Estate"

    # Work Colleagues
    if any(
        word in combined
        for word in [
            "colleague",
            "coworker",
            "co-worker",
            "manager",
            "boss",
            "team",
            "employee",
            "ceo",
            "cto",
            "director",
            "vp",
            "engineer",
            "developer",
            "designer",
            "analyst",
            "intern",
            "supervisor",
            "lead",
            "head of",
        ]
    ):
        return "Work Colleagues"

    # Friends & Neighbors
    if any(
        word in combined
        for word in ["friend", "neighbor", "neighbour", "buddy", "pal", "roommate"]
    ):
        return "Friends & Neighbors"

    # Professional Network (catch-all for business contacts)
    if any(
        word in combined
        for word in [
            "professional",
            "network",
            "linkedin",
            "conference",
            "meetup",
            "mentor",
            "advisor",
            "consultant",
            "investor",
            "founder",
            "entrepreneur",
        ]
    ):
        return "Professional Network"

    return "Other"


def generate_report(
    contacts: list[dict[str, Any]], output_path: str
) -> None:
    """Generate a formatted text report."""
    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for contact in contacts:
        category = categorize_contact(contact)
        contact["category"] = category
        by_category[category].append(contact)

    # Sort categories by our predefined order
    sorted_categories = [c for c in CATEGORIES if c in by_category]
    sorted_categories.extend([c for c in by_category if c not in CATEGORIES])

    lines = [
        "=" * 70,
        "🎄 INTELLIGENT CONTACT EXTRACTION REPORT 🎄",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Contacts: {len(contacts)}",
        "=" * 70,
        "",
    ]

    # Summary by category
    lines.append("SUMMARY BY CATEGORY")
    lines.append("-" * 40)
    for category in sorted_categories:
        count = len(by_category[category])
        lines.append(f"  {category}: {count}")
    lines.append("")

    # Detailed listings
    for category in sorted_categories:
        category_contacts = by_category[category]
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"{category.upper()} ({len(category_contacts)})")
        lines.append("=" * 70)

        # Sort by importance
        importance_order = {"high": 0, "medium": 1, "low": 2}
        category_contacts.sort(
            key=lambda x: importance_order.get(x.get("importance", "low"), 3)
        )

        for contact in category_contacts:
            lines.append("")
            lines.append(f"  Name: {contact.get('name', 'Unknown')}")
            if contact.get("relationship"):
                lines.append(f"  Relationship: {contact['relationship']}")
            if contact.get("organization"):
                lines.append(f"  Organization: {contact['organization']}")
            if contact.get("context"):
                lines.append(f"  Context: {contact['context']}")
            lines.append(
                f"  Importance: {contact.get('importance', 'unknown').upper()}"
            )
            sentiment = contact.get("sentiment", "unknown")
            sentiment_emoji = {
                "positive": "(+)",
                "neutral": "(o)",
                "complicated": "(?)",
                "negative": "(-)",
            }.get(sentiment, "")
            lines.append(f"  Sentiment: {sentiment} {sentiment_emoji}")
            if contact.get("contact_info"):
                lines.append(f"  Contact: {contact['contact_info']}")
            lines.append("  " + "-" * 40)

    # Holiday greeting checklist
    high_contacts = [c for c in contacts if c.get("importance") == "high"]
    medium_contacts = [c for c in contacts if c.get("importance") == "medium"]
    holiday_list = high_contacts + medium_contacts

    if holiday_list:
        lines.append("")
        lines.append("=" * 70)
        lines.append("## 🎄 HOLIDAY GREETING LIST (High + Medium)")
        lines.append("=" * 70)
        for person in holiday_list:
            name = person.get("name", "Unknown")
            org = person.get("organization", "")
            lines.append(f"  ☐ {name}" + (f" - {org}" if org else ""))
        lines.append("")
        lines.append("💡 Review and add anyone you remember!")
        lines.append("=" * 70)

    # Write report
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"✅ Report written to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract contacts from Claude conversation exports using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intelligent_contacts.py conversations.json
  python intelligent_contacts.py export.json -o my_contacts
  python intelligent_contacts.py data/claude_export.json --batch-size 5

Author: Daniel Zivkovic / Magma Inc.
        """,
    )
    parser.add_argument(
        "input_file",
        help="Path to Claude export JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="extracted_contacts",
        help="Output file base name (default: extracted_contacts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of conversations per API batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return 1

    try:
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)

        # Load and process export
        conversations = load_claude_export(args.input_file)
        texts = extract_conversation_text(conversations)

        if not texts:
            logger.warning("No conversation text found in export")
            return 1

        # Process in batches
        chunks = chunk_conversations(texts, args.batch_size)
        all_contacts: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks, 1):
            contacts = extract_contacts_from_batch(client, chunk, i, len(chunks))
            all_contacts.extend(contacts)

        if not all_contacts:
            logger.warning("No contacts extracted from conversations")
            return 0

        # Deduplicate and categorize
        unique_contacts = deduplicate_contacts(all_contacts)
        for contact in unique_contacts:
            contact["category"] = categorize_contact(contact)

        # Generate outputs
        output_base = args.output.rstrip(".json").rstrip(".txt")
        txt_path = f"{output_base}.txt"
        json_path = f"{output_base}.json"

        generate_report(unique_contacts, txt_path)

        # Write JSON output
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unique_contacts, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ JSON data written to: {json_path}")

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"✨ EXTRACTION COMPLETE: {len(unique_contacts)} unique contacts")
        logger.info(f"  📄 Report: {txt_path}")
        logger.info(f"  📊 Data:   {json_path}")
        logger.info("=" * 50)

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return 1
    except anthropic.AuthenticationError:
        logger.error("Invalid ANTHROPIC_API_KEY - please check your API key")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
