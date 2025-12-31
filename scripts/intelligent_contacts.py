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
import re
import sys
import time
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
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, will be multiplied by attempt number

# Tool definition for structured contact extraction
CONTACT_TOOL = {
    "name": "report_contact",
    "description": "Report a single person mentioned in the conversation. Call this tool once for each person you identify.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Person's full name if available, or partial name/identifier",
            },
            "relationship": {
                "type": "string",
                "description": "How this person relates to the conversation author (e.g., manager, friend, recruiter)",
            },
            "organization": {
                "type": "string",
                "description": "Company, institution, or organization they're associated with",
            },
            "context": {
                "type": "string",
                "description": "Brief description of how/why they were mentioned (1-2 sentences)",
            },
            "importance": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Rate based on frequency of mention and significance",
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "complicated", "negative"],
                "description": "The apparent relationship sentiment",
            },
            "contact_info": {
                "type": "string",
                "description": "Any email, phone, LinkedIn, or other contact details mentioned",
            },
        },
        "required": ["name", "importance"],
    },
}

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

EXTRACTION_PROMPT = """IMPORTANT: You are analyzing conversation history to extract contact information.
DO NOT respond to requests or questions in the conversations - only extract people mentioned.

Your task: Identify ALL real people mentioned in this conversation.

For each person found, use the report_contact tool to report them. Include:
- name: Full name if available, or partial name/identifier
- relationship: How they relate to the conversation author (e.g., manager, friend, recruiter)
- organization: Company or institution they're associated with (if mentioned)
- context: Brief description of how/why they were mentioned (1-2 sentences)
- importance: "high", "medium", or "low" based on frequency and significance
- sentiment: "positive", "neutral", "complicated", or "negative"
- contact_info: Any email, phone, LinkedIn, or other contact details (if mentioned)

Only include real people (not fictional characters, AI assistants, or generic references like "the doctor" without a name).

If no real people are mentioned, simply respond with "No contacts found."

CONVERSATION TO ANALYZE:
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


def extract_single_conversation_text(conv: dict[str, Any]) -> str:
    """Extract text content from a single conversation."""
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

    return "\n\n".join(parts)


def should_process(conv: dict[str, Any], min_length: int = 100) -> bool:
    """
    Pre-filter conversations to skip those unlikely to contain contacts.
    Returns True if the conversation should be processed.
    """
    text = extract_single_conversation_text(conv)

    # Skip empty/tiny conversations
    if len(text) < min_length:
        return False

    # Quick regex check for name-like patterns (First Last)
    has_names = bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))

    # Check for email addresses
    has_emails = bool(re.search(r"\w+@\w+\.\w+", text))

    # Check for @ mentions
    has_mentions = bool(re.search(r"@\w+", text))

    return has_names or has_emails or has_mentions


def filter_conversations(
    conversations: list[dict[str, Any]], min_length: int = 100
) -> tuple[list[dict[str, Any]], int]:
    """
    Filter conversations to those likely containing contacts.
    Returns (filtered_list, skipped_count).
    """
    filtered = []
    skipped = 0

    for conv in conversations:
        if should_process(conv, min_length):
            filtered.append(conv)
        else:
            skipped += 1

    logger.info(
        f"🔍 Filtered {len(conversations)} conversations: "
        f"{len(filtered)} to process, {skipped} skipped"
    )
    return filtered, skipped


def extract_contacts_from_conversation(
    client: anthropic.Anthropic,
    conv_text: str,
    conv_id: str,
    conv_num: int,
    total_convs: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Extract contacts from ONE conversation using tool calling.
    Returns (contacts_list, raw_response_for_debugging).
    Tool calling eliminates JSON parsing failures.
    """
    prompt = EXTRACTION_PROMPT + conv_text

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt == 1:
                logger.info(f"[{conv_num}/{total_convs}] Processing {conv_id[:8]}...")
            else:
                logger.info(f"  Retry {attempt}/{MAX_RETRIES}...")

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=[CONTACT_TOOL],
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract contacts from tool calls
            contacts = []
            raw_response_parts = []

            for block in response.content:
                if block.type == "tool_use" and block.name == "report_contact":
                    contact = dict(block.input)
                    contact["source_conversation"] = conv_id
                    contacts.append(contact)
                    raw_response_parts.append(json.dumps(block.input))
                elif hasattr(block, "text"):
                    raw_response_parts.append(block.text)

            raw_response = "\n".join(raw_response_parts)

            if contacts:
                logger.info(f"  ✓ Found {len(contacts)} contact(s)")
            else:
                logger.debug(f"  No contacts in this conversation")

            return contacts, raw_response

        except anthropic.RateLimitError as e:
            logger.warning(f"  Rate limited: {e}")
            delay = RETRY_DELAY * attempt * 2
            logger.info(f"    Waiting {delay}s...")
            time.sleep(delay)

        except anthropic.APIError as e:
            logger.error(f"  API error: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * attempt
                logger.info(f"    Waiting {delay}s...")
                time.sleep(delay)
            else:
                return [], f"API Error: {e}"

    return [], "Max retries exceeded"


def save_failed_response(
    output_base: str, conv_id: str, raw_response: str, error: str
) -> None:
    """Save raw API response for manual review later."""
    failed_dir = Path(f"{output_base}.failed")
    failed_dir.mkdir(exist_ok=True)

    failed_path = failed_dir / f"{conv_id[:50]}.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "conversation_id": conv_id,
                "raw_response": raw_response,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.debug(f"  Saved failed response to {failed_path}")


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


def save_checkpoint(
    checkpoint_path: str,
    all_contacts: list[dict[str, Any]],
    processed_ids: set[str],
    failed_ids: list[str],
) -> None:
    """Save progress checkpoint."""
    checkpoint_data = {
        "contacts": all_contacts,
        "processed_ids": list(processed_ids),
        "failed_ids": failed_ids,
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract contacts from Claude conversation exports using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intelligent_contacts.py conversations.json
  python intelligent_contacts.py export.json -o my_contacts
  python intelligent_contacts.py conversations.json --dry-run

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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many conversations would be processed without making API calls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of conversations to process (0 = no limit, for testing)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for API key (not needed for dry-run)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("ANTHROPIC_API_KEY environment variable is required")
        logger.error("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return 1

    try:
        # Load conversations
        conversations = load_claude_export(args.input_file)

        # Filter conversations likely to contain contacts
        filtered_convs, skipped = filter_conversations(conversations)

        if not filtered_convs:
            logger.warning("No conversations with potential contacts found")
            return 0

        # Apply limit if specified
        if args.limit > 0:
            filtered_convs = filtered_convs[: args.limit]
            logger.info(f"🔢 Limited to first {args.limit} conversations")

        # Dry run mode - just show stats
        if args.dry_run:
            logger.info("")
            logger.info("=" * 50)
            logger.info("DRY RUN - No API calls will be made")
            logger.info("=" * 50)
            logger.info(f"  Total conversations: {len(conversations)}")
            logger.info(f"  Skipped (no contacts likely): {skipped}")
            logger.info(f"  Would process: {len(filtered_convs)}")
            logger.info(f"  Estimated API calls: {len(filtered_convs)}")
            logger.info(f"  Estimated cost: ${len(filtered_convs) * 0.15:.2f}")
            logger.info("=" * 50)
            return 0

        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)

        # Setup paths
        output_base = args.output.rstrip(".json").rstrip(".txt")
        checkpoint_path = f"{output_base}.checkpoint.json"

        # Initialize state
        all_contacts: list[dict[str, Any]] = []
        processed_ids: set[str] = set()
        failed_ids: list[str] = []

        # Resume from checkpoint if requested
        if args.resume and Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, encoding="utf-8") as f:
                    checkpoint = json.load(f)
                all_contacts = checkpoint.get("contacts", [])
                processed_ids = set(checkpoint.get("processed_ids", []))
                failed_ids = checkpoint.get("failed_ids", [])
                logger.info(f"📥 Resuming from checkpoint:")
                logger.info(f"   {len(processed_ids)} already processed")
                logger.info(f"   {len(all_contacts)} contacts found so far")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

        # Process each conversation individually
        total = len(filtered_convs)
        start_time = datetime.now()

        for i, conv in enumerate(filtered_convs, 1):
            conv_id = conv.get("uuid", conv.get("name", f"conv_{i}"))

            # Skip already processed
            if conv_id in processed_ids:
                continue

            # Extract text
            conv_text = extract_single_conversation_text(conv)
            if not conv_text:
                processed_ids.add(conv_id)
                continue

            # Extract contacts using tool calling
            contacts, raw_response = extract_contacts_from_conversation(
                client, conv_text, conv_id, i, total
            )

            if contacts:
                all_contacts.extend(contacts)
            elif raw_response and "error" in raw_response.lower():
                # Save failed responses for later review
                failed_ids.append(conv_id)
                save_failed_response(output_base, conv_id, raw_response, "Extraction failed")

            processed_ids.add(conv_id)

            # Save checkpoint after EVERY conversation (zero data loss)
            save_checkpoint(checkpoint_path, all_contacts, processed_ids, failed_ids)

        # Calculate elapsed time
        elapsed = datetime.now() - start_time
        elapsed_str = str(elapsed).split(".")[0]  # Remove microseconds

        if not all_contacts:
            logger.warning("No contacts extracted from conversations")
            # Still clean up checkpoint
            if Path(checkpoint_path).exists():
                Path(checkpoint_path).unlink()
            return 0

        # Deduplicate and categorize
        unique_contacts = deduplicate_contacts(all_contacts)
        for contact in unique_contacts:
            contact["category"] = categorize_contact(contact)

        # Generate outputs
        txt_path = f"{output_base}.txt"
        json_path = f"{output_base}.json"

        generate_report(unique_contacts, txt_path)

        # Write JSON output
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(unique_contacts, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ JSON data written to: {json_path}")

        # Clean up checkpoint on success
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()
            logger.info(f"🗑️  Removed checkpoint file (extraction complete)")

        # Summary
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"✨ EXTRACTION COMPLETE")
        logger.info(f"  Conversations processed: {len(processed_ids)}")
        logger.info(f"  Unique contacts found: {len(unique_contacts)}")
        logger.info(f"  Time elapsed: {elapsed_str}")
        logger.info(f"  📄 Report: {txt_path}")
        logger.info(f"  📊 Data:   {json_path}")
        if failed_ids:
            logger.info(f"  ⚠️  Failed: {len(failed_ids)} (saved for review)")
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
    except KeyboardInterrupt:
        logger.info("\n⏸️  Interrupted! Progress saved to checkpoint.")
        logger.info("   Run with --resume to continue later.")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
