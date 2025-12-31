"""
common.py - Shared utilities for claude-mining scripts
======================================================
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


def load_claude_export(file_path: str) -> dict[str, Any]:
    """
    Load and parse Claude export JSON file.
    Handles different export formats automatically.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize different export structures
    if isinstance(data, list):
        return {'conversations': data}
    elif 'conversations' not in data and 'chats' in data:
        data['conversations'] = data.pop('chats')
    elif 'conversations' not in data:
        # Single conversation or unknown format
        data = {'conversations': [data]}
    
    return data


def get_conversations(export_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract conversations list from export data."""
    return export_data.get('conversations', [])


def get_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract messages from a conversation."""
    return (
        conversation.get('messages', []) or 
        conversation.get('chat_messages', []) or
        []
    )


def get_message_content(message: dict[str, Any]) -> str:
    """Extract text content from a message."""
    content = message.get('content', '') or message.get('text', '')
    
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle content blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                texts.append(block['text'])
            elif isinstance(block, str):
                texts.append(block)
        return '\n'.join(texts)
    
    return str(content)


def get_project_name(conversation: dict[str, Any]) -> str:
    """Get project name from conversation, or 'No Project'."""
    project = conversation.get('project', {})
    if isinstance(project, dict):
        return project.get('name', 'No Project')
    return 'No Project'


def get_conversation_title(conversation: dict[str, Any]) -> str:
    """Get conversation title."""
    return (
        conversation.get('title', '') or 
        conversation.get('name', '') or
        'Untitled'
    )


def save_report(content: str, base_name: str, output_dir: str = '.') -> str:
    """Save report to file, return path."""
    output_path = Path(output_dir) / f"{base_name}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return str(output_path)


def save_json(data: Any, base_name: str, output_dir: str = '.') -> str:
    """Save data as JSON, return path."""
    output_path = Path(output_dir) / f"{base_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    return str(output_path)


def get_anthropic_client():
    """
    Get Anthropic API client if available.
    Returns None if not installed or no API key.
    """
    try:
        import anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        pass
    return None


def call_claude(
    client,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096
) -> Optional[str]:
    """
    Call Claude API with error handling.
    Returns response text or None on error.
    """
    if not client:
        return None
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"API error: {e}")
        return None


def format_timestamp(iso_string: str) -> str:
    """Convert ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return iso_string


# Common regex patterns for extraction
PATTERNS = {
    'greeting_name': r'\b(?:Hi|Hello|Hey|Dear|Morning|Thanks?|Thank you)\s+([A-Z][a-z]+)',
    'title_name': r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    'at_mention': r'@([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
    'url': r'https?://[^\s<>"\']+',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
}

# Common false positives to filter out
FALSE_POSITIVE_NAMES = {
    'The', 'This', 'That', 'Here', 'There', 'What', 'When', 'Where',
    'How', 'Why', 'Can', 'Could', 'Would', 'Should', 'Will', 'May',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'January', 'February', 'March', 'April', 'June', 'July', 'August',
    'September', 'October', 'November', 'December', 'Claude', 'Anthropic',
    'Google', 'Microsoft', 'Amazon', 'Apple', 'GitHub', 'LinkedIn',
    'Thanks', 'Thank', 'Please', 'Sorry', 'Great', 'Good', 'Best',
    'Hello', 'Morning', 'Evening', 'Night', 'Today', 'Tomorrow',
}
