# Claude Export Data Format

This document describes the structure of Claude's data export.

## How to Export

1. Go to [claude.ai](https://claude.ai)
2. Click your profile initials → **Settings**
3. Navigate to **Privacy**
4. Click **Export data**
5. Wait for email with download link
6. Download the JSON file

## Export Structure

```json
{
  "account": {
    "email": "your@email.com",
    "created_at": "2024-01-15T..."
  },
  "conversations": [
    {
      "uuid": "abc123...",
      "title": "Conversation Title",
      "created_at": "2024-12-01T...",
      "updated_at": "2024-12-04T...",
      "project": {
        "uuid": "project-uuid",
        "name": "Project Name"
      },
      "messages": [
        {
          "uuid": "msg-uuid",
          "role": "user",
          "content": "User message text...",
          "created_at": "2024-12-01T..."
        },
        {
          "uuid": "msg-uuid-2",
          "role": "assistant", 
          "content": "Claude's response...",
          "created_at": "2024-12-01T..."
        }
      ]
    }
  ]
}
```

## Key Fields

### Conversation Object

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Unique identifier |
| `title` | string | Auto-generated or user-set title |
| `created_at` | ISO datetime | When conversation started |
| `updated_at` | ISO datetime | Last activity |
| `project` | object/null | Project if in a project |
| `messages` | array | All messages in order |

### Message Object

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Message identifier |
| `role` | string | "user" or "assistant" |
| `content` | string/array | Message content |
| `created_at` | ISO datetime | When sent |

### Content Formats

Content can be:

1. **Simple string:**
   ```json
   "content": "Hello, how are you?"
   ```

2. **Content blocks array:**
   ```json
   "content": [
     {"type": "text", "text": "Here's the code:"},
     {"type": "code", "language": "python", "code": "print('hello')"}
   ]
   ```

## Projects

Conversations inside projects have a `project` field:

```json
"project": {
  "uuid": "proj-123",
  "name": "My Project"
}
```

Conversations outside projects have `project: null`.

## Privacy Considerations

⚠️ **Your export contains:**
- All message content (including sensitive info you shared)
- Email addresses mentioned in conversations
- Code snippets
- File contents you uploaded
- Your account email

**Never commit export files to public repositories.**

## Typical Export Sizes

| Conversations | Approximate Size |
|---------------|------------------|
| 50 | 1-5 MB |
| 500 | 10-50 MB |
| 5000 | 100-500 MB |

Heavy code users may have larger exports.
