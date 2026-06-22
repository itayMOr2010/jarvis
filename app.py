"""
JARVIS Central Server
---------------------
Runs on PythonAnywhere (free tier).
Handles tasks, calendar events, memory, and NLP command parsing.
All responses are in English; input can be Hebrew, English, or mixed.
"""

from flask import Flask, request, jsonify
import json
import os
import datetime
import re

app = Flask(__name__)

# ── Data files (PythonAnywhere keeps these forever on the free tier) ──────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

TASKS_FILE    = os.path.join(DATA_DIR, "tasks.json")
EVENTS_FILE   = os.path.join(DATA_DIR, "events.json")
MEMORY_FILE   = os.path.join(DATA_DIR, "memory.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def next_id(items):
    return max((i.get("id", 0) for i in items), default=0) + 1

def today_str():
    return datetime.date.today().isoformat()

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ── NLP parser: understands Hebrew + English + mixed ─────────────────────────
# We call the Anthropic API (claude-sonnet-4-6) so JARVIS can parse any phrase.

import urllib.request

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PARSE = """You are a command parser for a personal assistant called JARVIS.
The user may write in Hebrew, English, or a mix of both (Heblish).
Parse the user's message and return ONLY a JSON object (no markdown, no explanation).

JSON schema:
{
  "intent": "<one of: add_task | complete_task | delete_task | list_tasks |
                        add_event | list_events | delete_event |
                        remember | recall | status | unknown>",
  "task_name":   "<string or null>",
  "task_id":     "<integer or null>",
  "event_title": "<string or null>",
  "event_date":  "<YYYY-MM-DD or null>",
  "event_time":  "<HH:MM 24h or null>",
  "memory_key":  "<string or null>",
  "memory_value":"<string or null>",
  "filter":      "<today | this_week | all | null>"
}

Examples:
- "add a task buy groceries"          → intent: add_task, task_name: "buy groceries"
- "תוסיף משימה לקנות חלב"             → intent: add_task, task_name: "buy milk"  (translate to English)
- "add for the agenda today פגישה עם הבוס at 9 am" → intent: add_event, event_title: "Meeting with boss", event_date: <today>, event_time: "09:00"
- "mark task 3 as done"               → intent: complete_task, task_id: 3
- "what's on the agenda this week"    → intent: list_events, filter: "this_week"
- "remember that my camp starts July 5" → intent: remember, memory_key: "camp start", memory_value: "July 5"
- "what do you know about my camp"    → intent: recall, memory_key: "camp"

Always translate Hebrew content values (task names, event titles) into English.
Today is """ + today_str() + "."

def parse_command(text: str) -> dict:
    """Send user text to Claude for intent parsing."""
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 300,
        "system": SYSTEM_PARSE,
        "messages": [{"role": "user", "content": text}]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    raw = body["content"][0]["text"].strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


SYSTEM_RESPOND = """You are JARVIS, a personal AI assistant.
Personality: sharp, polite, mildly witty British butler — like Tony Stark's JARVIS.
Always respond in English, concisely (1-3 sentences max).
Address the user as "sir".
Never use emojis. Never be sycophantic.
Today is """ + today_str() + "."

def jarvis_reply(situation: str) -> str:
    """Generate a JARVIS-style English reply for the given situation."""
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 150,
        "system": SYSTEM_RESPOND,
        "messages": [{"role": "user", "content": situation}]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    return body["content"][0]["text"].strip()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "JARVIS online", "time": now_str()})


@app.route("/command", methods=["POST"])
def command():
    """Main endpoint. Accepts { "text": "..." } and returns { "reply": "..." }."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"reply": "I didn't catch that, sir."}), 400

    try:
        parsed = parse_command(text)
    except Exception as e:
        return jsonify({"reply": f"Parsing error, sir: {e}"}), 500

    intent = parsed.get("intent", "unknown")
    reply  = ""

    # ── TASKS ─────────────────────────────────────────────
    if intent == "add_task":
        tasks = load(TASKS_FILE)
        name = parsed.get("task_name") or "unnamed task"
        task = {"id": next_id(tasks), "name": name, "done": False, "created": today_str()}
        tasks.append(task)
        save(TASKS_FILE, tasks)
        situation = f'Task added successfully: "{name}". Give a brief confirmation.'
        reply = jarvis_reply(situation)

    elif intent == "list_tasks":
        tasks = load(TASKS_FILE)
        pending = [t for t in tasks if not t["done"]]
        if not pending:
            reply = jarvis_reply("The user has no pending tasks. Say so briefly.")
        else:
            names = ", ".join(f'[{t["id"]}] {t["name"]}' for t in pending)
            reply = jarvis_reply(f"List these pending tasks to the user: {names}")

    elif intent == "complete_task":
        tasks = load(TASKS_FILE)
        tid = parsed.get("task_id")
        matched = next((t for t in tasks if t["id"] == tid), None)
        if matched:
            matched["done"] = True
            matched["completed"] = today_str()
            save(TASKS_FILE, tasks)
            reply = jarvis_reply(f'Task "{matched["name"]}" marked as complete. Brief confirmation.')
        else:
            reply = jarvis_reply(f"Task ID {tid} not found. Say so briefly.")

    elif intent == "delete_task":
        tasks = load(TASKS_FILE)
        tid = parsed.get("task_id")
        new_tasks = [t for t in tasks if t["id"] != tid]
        if len(new_tasks) < len(tasks):
            save(TASKS_FILE, new_tasks)
            reply = jarvis_reply(f"Task {tid} deleted. Brief confirmation.")
        else:
            reply = jarvis_reply(f"Task ID {tid} not found. Say so.")

    # ── EVENTS / CALENDAR ─────────────────────────────────
    elif intent == "add_event":
        events = load(EVENTS_FILE)
        title = parsed.get("event_title") or "Untitled event"
        date  = parsed.get("event_date")  or today_str()
        time  = parsed.get("event_time")  or ""
        event = {"id": next_id(events), "title": title, "date": date, "time": time}
        events.append(event)
        save(EVENTS_FILE, events)
        when = f"{date} at {time}" if time else date
        reply = jarvis_reply(f'Event added: "{title}" on {when}. Brief witty confirmation.')

    elif intent == "list_events":
        events = load(EVENTS_FILE)
        f = parsed.get("filter") or "all"
        today = today_str()

        if f == "today":
            filtered = [e for e in events if e.get("date") == today]
        elif f == "this_week":
            week_end = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
            filtered = [e for e in events if today <= e.get("date","") <= week_end]
        else:
            filtered = events

        filtered.sort(key=lambda e: (e.get("date",""), e.get("time","")))

        if not filtered:
            reply = jarvis_reply("No events found for the requested period. Say so briefly.")
        else:
            lines = ", ".join(
                f'[{e["id"]}] {e["title"]} on {e["date"]}' + (f' at {e["time"]}' if e.get("time") else "")
                for e in filtered
            )
            reply = jarvis_reply(f"Read out these upcoming events: {lines}")

    elif intent == "delete_event":
        events = load(EVENTS_FILE)
        eid = parsed.get("task_id") or parsed.get("event_id")
        new_events = [e for e in events if e["id"] != eid]
        if len(new_events) < len(events):
            save(EVENTS_FILE, new_events)
            reply = jarvis_reply(f"Event {eid} removed from the agenda.")
        else:
            reply = jarvis_reply(f"Event {eid} not found.")

    # ── MEMORY ────────────────────────────────────────────
    elif intent == "remember":
        memory = load(MEMORY_FILE)
        key   = parsed.get("memory_key","").lower().strip()
        value = parsed.get("memory_value","")
        entry = next((m for m in memory if m["key"] == key), None)
        if entry:
            entry["value"] = value
            entry["updated"] = today_str()
        else:
            memory.append({"key": key, "value": value, "created": today_str()})
        save(MEMORY_FILE, memory)
        reply = jarvis_reply(f'I will remember: {key} = {value}. Brief confirmation.')

    elif intent == "recall":
        memory = load(MEMORY_FILE)
        key    = (parsed.get("memory_key") or "").lower().strip()
        matches = [m for m in memory if key in m["key"]]
        if matches:
            facts = "; ".join(f'{m["key"]}: {m["value"]}' for m in matches)
            reply = jarvis_reply(f"Share this memory with the user: {facts}")
        else:
            reply = jarvis_reply(f"No memory found for '{key}'. Say so briefly.")

    # ── STATUS ────────────────────────────────────────────
    elif intent == "status":
        tasks  = load(TASKS_FILE)
        events = load(EVENTS_FILE)
        today  = today_str()
        pending       = sum(1 for t in tasks  if not t["done"])
        today_events  = sum(1 for e in events if e.get("date") == today)
        reply = jarvis_reply(
            f"Give the user a morning briefing: {pending} pending tasks, "
            f"{today_events} events today ({today}). Be concise and witty."
        )

    else:
        # Unknown intent — let Claude figure out a helpful response
        reply = jarvis_reply(f'The user said: "{text}". Respond helpfully as JARVIS.')

    return jsonify({"reply": reply, "intent": intent, "parsed": parsed})


# ── Direct data endpoints (for debugging / mobile app) ────────────────────────

@app.route("/tasks", methods=["GET"])
def get_tasks():
    return jsonify(load(TASKS_FILE))

@app.route("/events", methods=["GET"])
def get_events():
    return jsonify(load(EVENTS_FILE))

@app.route("/memory", methods=["GET"])
def get_memory():
    return jsonify(load(MEMORY_FILE))


if __name__ == "__main__":
    app.run(debug=True)
