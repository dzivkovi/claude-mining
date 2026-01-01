#!/usr/bin/env python3
"""
Contact Deduplication & Entity Resolution Script

Post-processes extracted contacts to:
- Remove celebrity references (Gordon Ramsay, etc.)
- Remove self-references (author entries)
- Merge duplicate contacts (fuzzy name matching + LinkedIn URL matching)
- Clean relationship fields (dedupe, limit to top 5)
- Validate categories (fix miscategorized contacts)

Author: Daniel Zivkovic / Magma Inc.
License: MIT
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional RapidFuzz for fuzzy matching
try:
    from rapidfuzz.distance import JaroWinkler

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    JaroWinkler = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

CELEBRITY_PATTERNS = [
    "Gordon Ramsay",
    "Dave Ramsey",
    "Anthony Robbins",
    "Tony Robbins",
    "Gary Vaynerchuk",
    "Gary Vee",
    "Jeff Bezos",
    "Elon Musk",
    "Steve Jobs",
    "Bill Gates",
    "Mark Zuckerberg",
    "Warren Buffett",
    "Oprah Winfrey",
    "Barack Obama",
    "Donald Trump",
]

SELF_INDICATORS = [
    "self",
    "author",
    "conversation author",
    "myself",
    "self/author",
    "self (author)",
    "the author",
]

# Keywords that indicate a reference rather than actual contact
REFERENCE_KEYWORDS = [
    "mentioned",
    "reference",
    "analogy",
    "metaphor",
    "example",
    "quoted",
    "cited",
    "public figure",
    "celebrity",
]

# Category validation rules
CATEGORY_RULES = {
    "Family": {
        "required": [
            "wife",
            "husband",
            "mother",
            "father",
            "daughter",
            "son",
            "sister",
            "brother",
            "spouse",
            "parent",
            "mom",
            "dad",
            "child",
            "uncle",
            "aunt",
            "cousin",
            "grandma",
            "grandpa",
            "grandmother",
            "grandfather",
            "in-law",
        ],
        "fallback": "Professional Network",
    },
}

# Relationship priority for sorting (important roles first)
RELATIONSHIP_PRIORITY = [
    "mentor",
    "manager",
    "colleague",
    "client",
    "friend",
    "collaborator",
    "partner",
    "advisor",
    "consultant",
]


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    return " ".join(name.lower().split())


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ============================================================================
# Detection Functions
# ============================================================================


def is_celebrity(contact: dict) -> bool:
    """Detect celebrity references (not real contacts)."""
    name = contact.get("name", "")
    context = (contact.get("context", "") or "").lower()
    relationship = (contact.get("relationship", "") or "").lower()
    combined = f"{context} {relationship}"

    # Check against known celebrities
    for celeb in CELEBRITY_PATTERNS:
        if celeb.lower() in name.lower():
            # If relationship/context indicates it's a reference, flag it
            if any(kw in combined for kw in REFERENCE_KEYWORDS):
                return True
            # Also flag if relationship contains "none" or similar
            if "none" in relationship and "celebrity" not in relationship:
                return True

    return False


def is_self(contact: dict, user_name: str = "Daniel", user_last_name: str = "Zivkovic") -> bool:
    """
    Detect entries that refer to the user themselves.

    CONSERVATIVE approach: Only removes entries where we're highly confident.
    Ambiguous cases (e.g., "Daniel Stone" with "self" relationship) are left
    for the fuzzy matching phase where humans can review.

    Matches (auto-remove):
    - Exact first name alone: "Daniel"
    - First name + last name variants: "Daniel Zivkovic", "Daniel Z.", "Daniel Zikovic"
    - Name with parenthetical username: "Daniel (danie)", "Daniel (dzivkovi)"
    - Generic self-references: "author", "self", "myself"

    Does NOT match (kept for human review):
    - Different surnames: "Daniel Stone", "Daniel Blanchard", "Daniel Vaughan"
    - Family references: "Daniel's wife", "Daniel's daughter"
    - Unrelated people: "Philemon Daniel"
    """
    name = contact.get("name", "").strip()
    name_lower = name.lower()
    relationship = (contact.get("relationship", "") or "").lower()

    # Skip family references (possessive)
    if "'s " in name_lower:
        return False

    # Check if relationship strongly indicates self
    has_self_indicator = any(ind in relationship for ind in SELF_INDICATORS)

    if not has_self_indicator:
        return False

    # Now check if name matches user identity precisely
    user_first = user_name.lower()
    user_last = user_last_name.lower()

    # Exact first name only (no last name = likely self if has self-indicator)
    if name_lower == user_first:
        return True

    # First name with initial: "Daniel Z.", "Daniel Z"
    if name_lower.startswith(user_first + " ") and len(name_lower) <= len(user_first) + 3:
        remainder = name_lower[len(user_first) + 1:]
        if remainder in [user_last[0], user_last[0] + "."]:
            return True

    # Full name or close variants (with typo tolerance)
    # Must start with exact first name followed by space
    if name_lower.startswith(user_first + " "):
        after_first = name_lower[len(user_first) + 1:].strip()
        # Remove punctuation for cleaner matching
        after_clean = re.sub(r'[^\w\s]', '', after_first).strip().split()[0] if after_first else ""

        if after_clean and len(after_clean) >= 4:
            # Check similarity to user's last name using Jaro-Winkler
            if RAPIDFUZZ_AVAILABLE:
                similarity = JaroWinkler.similarity(after_clean, user_last)
                if similarity >= 0.80:  # 80% threshold for last name typos
                    return True
            else:
                # Fallback: character overlap
                overlap = sum(1 for c in user_last if c in after_clean)
                if overlap >= len(user_last) * 0.7:
                    return True

    # Name with parenthetical identifier suggesting author/username
    # e.g., "Daniel (danie)", "Daniel (author)", "Daniel (dzivkovi)", "Daniel (c_dzivkovic)"
    if name_lower.startswith(user_first + " (") and name_lower.endswith(")"):
        inner = name_lower[len(user_first) + 2:-1]
        # Check if inner part contains username patterns or "author"
        if "author" in inner:
            return True
        # Check if inner contains user's name/username patterns
        if user_first[:3] in inner or user_last[:4] in inner:
            return True
        # Check for underscore-prefixed usernames like "c_dzivkovic", "dzivkovi"
        if "_" in inner or inner.startswith(user_first[0]):
            # Additional check: does it look like a username?
            if re.match(r'^[a-z0-9_]+$', inner):
                return True

    # Generic self-references
    if name_lower in ["author", "self", "myself", "me"]:
        return True

    return False


# ============================================================================
# Candidate Finding Functions
# ============================================================================


def find_linkedin_candidates(contacts: list) -> list:
    """Find candidates sharing LinkedIn URLs (exact match = definite duplicate)."""
    linkedin_pattern = re.compile(r"linkedin\.com/in/([a-zA-Z0-9_-]+)", re.IGNORECASE)
    linkedin_index = defaultdict(list)

    for i, c in enumerate(contacts):
        contact_info = c.get("contact_info", "") or ""
        for handle in linkedin_pattern.findall(contact_info):
            linkedin_index[handle.lower()].append(i)

    candidates = []
    for handle, indices in linkedin_index.items():
        if len(indices) > 1:
            for a_idx, idx_a in enumerate(indices):
                for idx_b in indices[a_idx + 1 :]:
                    candidates.append(
                        {
                            "index_a": idx_a,
                            "index_b": idx_b,
                            "name_a": contacts[idx_a].get("name"),
                            "name_b": contacts[idx_b].get("name"),
                            "score": 1.0,
                            "reason": f"linkedin:{handle}",
                        }
                    )

    return candidates


def is_sequential_label(name: str) -> bool:
    """Check if name is a sequential label like Speaker_01, Participant 3, etc."""
    patterns = [
        r"^speaker[_\s]?\d+$",
        r"^participant[_\s]?\d+$",
        r"^person[_\s]?\d+$",
        r"^user[_\s]?\d+$",
        r"^guest[_\s]?\d+$",
    ]
    name_lower = name.lower().strip()
    return any(re.match(p, name_lower) for p in patterns)


def find_fuzzy_candidates(contacts: list, threshold: float = 0.88) -> list:
    """Find candidate duplicate pairs using Jaro-Winkler similarity."""
    if not RAPIDFUZZ_AVAILABLE:
        logger.warning("RapidFuzz not installed. Skipping fuzzy matching.")
        logger.warning("Install with: pip install rapidfuzz")
        return []

    candidates = []
    n = len(contacts)

    # Progress logging for large datasets
    if n > 500:
        logger.info(f"Computing fuzzy matches for {n} contacts...")

    for i in range(n):
        name_a = normalize_name(contacts[i].get("name", ""))
        if len(name_a) < 2:
            continue

        # Skip sequential labels (Speaker_01, etc.)
        if is_sequential_label(contacts[i].get("name", "")):
            continue

        for j in range(i + 1, n):
            name_b = normalize_name(contacts[j].get("name", ""))
            if len(name_b) < 2:
                continue

            # Skip sequential labels
            if is_sequential_label(contacts[j].get("name", "")):
                continue

            # Quick length check to skip obviously different names
            len_diff = abs(len(name_a) - len(name_b))
            if len_diff > max(len(name_a), len(name_b)) * 0.5:
                continue

            score = JaroWinkler.similarity(name_a, name_b)

            if score >= threshold:
                candidates.append(
                    {
                        "index_a": i,
                        "index_b": j,
                        "name_a": contacts[i].get("name"),
                        "name_b": contacts[j].get("name"),
                        "score": score,
                        "reason": "fuzzy_name",
                    }
                )

    return candidates


# ============================================================================
# Merge & Cleanup Functions
# ============================================================================


def merge_contacts(contact_a: dict, contact_b: dict) -> dict:
    """Merge two contacts, preferring more complete information."""
    merged = contact_a.copy()

    # Prefer longer/more complete values for these fields
    for key in ["name", "organization", "context", "contact_info"]:
        val_a = contact_a.get(key) or ""
        val_b = contact_b.get(key) or ""
        if len(val_b) > len(val_a):
            merged[key] = val_b

    # Keep higher importance
    importance_rank = {"high": 3, "medium": 2, "low": 1}
    imp_a = importance_rank.get(contact_a.get("importance", "low"), 1)
    imp_b = importance_rank.get(contact_b.get("importance", "low"), 1)
    if imp_b > imp_a:
        merged["importance"] = contact_b["importance"]

    # Merge relationships (will be cleaned later)
    rel_a = contact_a.get("relationship", "") or ""
    rel_b = contact_b.get("relationship", "") or ""
    if rel_b and rel_b != rel_a:
        if rel_a:
            merged["relationship"] = f"{rel_a}, {rel_b}"
        else:
            merged["relationship"] = rel_b

    return merged


def clean_relationship_field(raw: str) -> str:
    """Deduplicate and limit relationship field to top 5 values."""
    if not raw:
        return ""

    # Split by comma
    parts = [p.strip() for p in raw.split(",")]

    # Normalize and deduplicate (case-insensitive)
    seen = set()
    unique = []
    for part in parts:
        if not part:
            continue
        normalized = part.lower().strip()
        # Skip very short or generic entries
        if len(normalized) < 2:
            continue
        if normalized not in seen:
            seen.add(normalized)
            unique.append(part.strip())

    # Priority sort (important roles first)
    def sort_key(r):
        r_lower = r.lower()
        for i, p in enumerate(RELATIONSHIP_PRIORITY):
            if p in r_lower:
                return (i, r)
        return (len(RELATIONSHIP_PRIORITY), r)

    unique.sort(key=sort_key)

    # Limit to 5
    return ", ".join(unique[:5])


def validate_category(contact: dict) -> str:
    """Fix miscategorized contacts."""
    category = contact.get("category", "Other")
    relationship = (contact.get("relationship", "") or "").lower()
    context = (contact.get("context", "") or "").lower()
    combined = f"{relationship} {context}"

    if category in CATEGORY_RULES:
        rules = CATEGORY_RULES[category]
        if not any(kw in combined for kw in rules["required"]):
            return rules["fallback"]

    return category


# ============================================================================
# Human Review Interface
# ============================================================================


def present_for_review(candidate: dict, contacts: list) -> str:
    """CLI interface for human review of merge candidates."""
    a = contacts[candidate["index_a"]]
    b = contacts[candidate["index_b"]]

    print("\n" + "=" * 70)
    print(f"POTENTIAL DUPLICATE (Score: {candidate['score']:.2f})")
    print("=" * 70)

    print(f"\n[A] {a.get('name')}")
    print(f"    Category: {a.get('category', 'N/A')}")
    print(f"    Org: {a.get('organization', 'N/A')}")
    print(f"    Rel: {truncate(a.get('relationship', ''), 70)}")
    print(f"    Contact: {truncate(a.get('contact_info', ''), 70)}")

    print(f"\n[B] {b.get('name')}")
    print(f"    Category: {b.get('category', 'N/A')}")
    print(f"    Org: {b.get('organization', 'N/A')}")
    print(f"    Rel: {truncate(b.get('relationship', ''), 70)}")
    print(f"    Contact: {truncate(b.get('contact_info', ''), 70)}")

    print(f"\nReason: {candidate['reason']}")
    print("\n[M]erge  [S]kip  [A]uto-merge remaining  [Q]uit")

    while True:
        decision = input("Decision: ").strip().lower()
        if decision in ["m", "s", "a", "q", "merge", "skip", "auto", "quit"]:
            return decision[0]
        print("Invalid choice. Enter M, S, A, or Q.")


# ============================================================================
# Main Processing
# ============================================================================


def process_contacts(
    contacts: list,
    remove_celebrities: bool = True,
    remove_self: bool = True,
    auto_merge_threshold: float = 0.0,
    fuzzy_threshold: float = 0.88,
    dry_run: bool = False,
    clean_only: bool = False,
    user_name: str = "Daniel",
    user_last_name: str = "Zivkovic",
) -> tuple[list, list]:
    """
    Main processing pipeline.

    Returns (processed_contacts, audit_log).
    """
    audit_log = []
    result = contacts.copy()

    # Phase 1: Remove celebrities
    if remove_celebrities:
        before = len(result)
        removed = []
        kept = []
        for c in result:
            if is_celebrity(c):
                removed.append(c)
                audit_log.append(
                    {
                        "action": "remove_celebrity",
                        "name": c.get("name"),
                        "reason": "Celebrity reference detected",
                    }
                )
            else:
                kept.append(c)
        result = kept
        logger.info(f"🎭 Removed {before - len(result)} celebrity references")

    # Phase 2: Remove self-references
    if remove_self:
        before = len(result)
        kept = []
        for c in result:
            if is_self(c, user_name, user_last_name):
                audit_log.append(
                    {
                        "action": "remove_self",
                        "name": c.get("name"),
                        "reason": "Self-reference detected",
                    }
                )
            else:
                kept.append(c)
        result = kept
        logger.info(f"👤 Removed {before - len(result)} self-references")

    # Phase 3: Find duplicate candidates (if not clean-only mode)
    if not clean_only:
        # LinkedIn matches (highest confidence)
        linkedin_candidates = find_linkedin_candidates(result)
        logger.info(f"🔗 Found {len(linkedin_candidates)} LinkedIn URL matches")

        # Fuzzy name matches
        fuzzy_candidates = find_fuzzy_candidates(result, fuzzy_threshold)
        logger.info(f"🔍 Found {len(fuzzy_candidates)} fuzzy name matches")

        # Combine and sort by score (highest first)
        all_candidates = linkedin_candidates + fuzzy_candidates
        all_candidates.sort(key=lambda x: -x["score"])

        # Deduplicate candidate pairs (same pair might appear from multiple methods)
        seen_pairs = set()
        unique_candidates = []
        for c in all_candidates:
            pair = tuple(sorted([c["index_a"], c["index_b"]]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_candidates.append(c)

        if dry_run:
            print("\n" + "=" * 70)
            print("DRY RUN - Candidates Found (no changes made)")
            print("=" * 70)
            for i, c in enumerate(unique_candidates[:50], 1):
                print(
                    f"{i}. [{c['score']:.2f}] {c['name_a']} <-> {c['name_b']} ({c['reason']})"
                )
            if len(unique_candidates) > 50:
                print(f"... and {len(unique_candidates) - 50} more")
            return result, audit_log

        # Phase 4: Process candidates (merge or skip)
        auto_merge_all = False
        merged_indices = set()  # Track which indices have been merged

        for candidate in unique_candidates:
            idx_a = candidate["index_a"]
            idx_b = candidate["index_b"]

            # Skip if either contact was already merged
            if idx_a in merged_indices or idx_b in merged_indices:
                continue

            # Auto-merge if above threshold or user requested
            if auto_merge_all or candidate["score"] >= auto_merge_threshold > 0:
                # Merge B into A
                result[idx_a] = merge_contacts(result[idx_a], result[idx_b])
                merged_indices.add(idx_b)
                audit_log.append(
                    {
                        "action": "auto_merge",
                        "name_a": candidate["name_a"],
                        "name_b": candidate["name_b"],
                        "score": candidate["score"],
                        "reason": candidate["reason"],
                    }
                )
                logger.debug(
                    f"  Auto-merged: {candidate['name_a']} + {candidate['name_b']}"
                )
            else:
                # Human review
                decision = present_for_review(candidate, result)

                if decision == "m":
                    result[idx_a] = merge_contacts(result[idx_a], result[idx_b])
                    merged_indices.add(idx_b)
                    audit_log.append(
                        {
                            "action": "manual_merge",
                            "name_a": candidate["name_a"],
                            "name_b": candidate["name_b"],
                            "score": candidate["score"],
                        }
                    )
                elif decision == "s":
                    audit_log.append(
                        {
                            "action": "skip",
                            "name_a": candidate["name_a"],
                            "name_b": candidate["name_b"],
                        }
                    )
                elif decision == "a":
                    # Auto-merge this and all remaining
                    auto_merge_all = True
                    result[idx_a] = merge_contacts(result[idx_a], result[idx_b])
                    merged_indices.add(idx_b)
                    audit_log.append(
                        {
                            "action": "auto_merge_triggered",
                            "name_a": candidate["name_a"],
                            "name_b": candidate["name_b"],
                        }
                    )
                elif decision == "q":
                    logger.info("User quit review. Processing remaining contacts...")
                    break

        # Remove merged contacts
        result = [c for i, c in enumerate(result) if i not in merged_indices]
        logger.info(f"✂️ Merged {len(merged_indices)} duplicate entries")

    # Phase 5: Clean relationship fields (all contacts)
    for contact in result:
        original = contact.get("relationship", "")
        cleaned = clean_relationship_field(original)
        if cleaned != original:
            contact["relationship"] = cleaned

    logger.info("🧹 Cleaned relationship fields")

    # Phase 6: Validate categories
    category_fixes = 0
    for contact in result:
        original = contact.get("category", "Other")
        validated = validate_category(contact)
        if validated != original:
            contact["category"] = validated
            category_fixes += 1
            audit_log.append(
                {
                    "action": "fix_category",
                    "name": contact.get("name"),
                    "from": original,
                    "to": validated,
                }
            )

    if category_fixes:
        logger.info(f"📁 Fixed {category_fixes} category assignments")

    return result, audit_log


def write_report(contacts: list, output_path: Path) -> None:
    """Write human-readable report."""
    # Group by category
    by_category = defaultdict(list)
    for c in contacts:
        by_category[c.get("category", "Other")].append(c)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DEDUPLICATED CONTACTS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total contacts: {len(contacts)}\n")
        f.write("=" * 70 + "\n\n")

        # Sort categories
        category_order = [
            "Family",
            "Work Colleagues",
            "Clients",
            "Professional Network",
            "Recruiters",
            "Financial & Real Estate",
            "Friends & Neighbors",
            "Other",
        ]

        for category in category_order:
            if category not in by_category:
                continue

            contacts_in_cat = by_category[category]
            f.write(f"\n## {category} ({len(contacts_in_cat)})\n")
            f.write("-" * 40 + "\n")

            # Sort by importance
            importance_order = {"high": 0, "medium": 1, "low": 2}
            contacts_in_cat.sort(
                key=lambda x: importance_order.get(x.get("importance", "low"), 2)
            )

            for c in contacts_in_cat:
                importance = c.get("importance", "low")
                star = " ⭐" if importance == "high" else ""
                f.write(f"\n  • {c.get('name', 'Unknown')}{star}\n")

                if c.get("organization"):
                    f.write(f"    Org: {c['organization']}\n")
                if c.get("relationship"):
                    f.write(f"    Rel: {c['relationship']}\n")
                if c.get("contact_info"):
                    f.write(f"    Contact: {truncate(c['contact_info'], 60)}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total: {len(contacts)} contacts\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and clean extracted contacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python deduplicate_contacts.py contacts.json

  # Auto-merge high-confidence matches
  python deduplicate_contacts.py contacts.json --auto-merge 0.95

  # Remove celebrities and self-references automatically
  python deduplicate_contacts.py contacts.json --remove-celebrities --remove-self

  # Dry run (preview candidates only)
  python deduplicate_contacts.py contacts.json --dry-run

  # Clean relationships only (no merge prompts)
  python deduplicate_contacts.py contacts.json --clean-only
        """,
    )

    parser.add_argument("input", help="Path to contacts JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: input-deduped.json)",
    )
    parser.add_argument(
        "--auto-merge",
        type=float,
        default=0.0,
        metavar="THRESHOLD",
        help="Auto-merge candidates with score >= threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.88,
        help="Jaro-Winkler threshold for fuzzy matching (default: 0.88)",
    )
    parser.add_argument(
        "--remove-celebrities",
        action="store_true",
        help="Remove celebrity references automatically",
    )
    parser.add_argument(
        "--remove-self",
        action="store_true",
        help="Remove self-references automatically",
    )
    parser.add_argument(
        "--user-name",
        default="Daniel",
        help="User's first name for self-detection (default: Daniel)",
    )
    parser.add_argument(
        "--user-last-name",
        default="Zivkovic",
        help="User's last name for self-detection (default: Zivkovic)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview candidates without making changes",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean relationship fields (no merge prompts)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine output path
    input_path = Path(args.input)
    if args.output:
        output_base = Path(args.output).with_suffix("")
    else:
        output_base = input_path.with_suffix("").with_name(
            f"{input_path.stem}-deduped"
        )

    # Load contacts
    logger.info(f"📂 Loading contacts from {input_path}")
    try:
        with open(input_path, encoding="utf-8") as f:
            contacts = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load contacts: {e}")
        sys.exit(1)

    logger.info(f"📊 Loaded {len(contacts)} contacts")

    # Process
    processed, audit_log = process_contacts(
        contacts,
        remove_celebrities=args.remove_celebrities,
        remove_self=args.remove_self,
        auto_merge_threshold=args.auto_merge,
        fuzzy_threshold=args.fuzzy_threshold,
        dry_run=args.dry_run,
        clean_only=args.clean_only,
        user_name=args.user_name,
        user_last_name=args.user_last_name,
    )

    if args.dry_run:
        logger.info("Dry run complete. No files written.")
        return

    # Write outputs
    output_json = output_base.with_suffix(".json")
    output_txt = output_base.with_suffix(".txt")
    output_audit = output_base.with_name(f"{output_base.name}.audit.json")

    # JSON output
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Wrote {len(processed)} contacts to {output_json}")

    # Text report
    write_report(processed, output_txt)
    logger.info(f"📝 Wrote report to {output_txt}")

    # Audit log
    with open(output_audit, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "input": str(input_path),
                "contacts_before": len(contacts),
                "contacts_after": len(processed),
                "actions": audit_log,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"📋 Wrote audit log to {output_audit}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Before: {len(contacts)} contacts")
    print(f"  After:  {len(processed)} contacts")
    print(f"  Reduced by: {len(contacts) - len(processed)} ({(1 - len(processed)/len(contacts))*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
