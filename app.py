import os, re, uuid, hashlib
from datetime import date, timedelta, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
INDIA_TZ = ZoneInfo("Asia/Kolkata")

app = FastAPI(title="Agentic RAG Schedule Assistant", version="4.1")


def now_india():
    return datetime.now(INDIA_TZ)


def today_india():
    return now_india().date()


client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("schedule", metadata={"hnsw:space": "cosine"})


def local_embedding(text: str, size: int = 96):
    values = [0.0] * size
    words = re.findall(r"[a-z0-9]+", text.lower()) or [text.lower()]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            idx = int.from_bytes(digest[i:i + 2], "big") % size
            values[idx] += 1.0
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def samples():
    s = today_india()
    return [
        {"id":"evt-001","title":"Project Team Meeting","event_type":"meeting","date":(s+timedelta(1)).isoformat(),"start_time":"10:00","end_time":"11:00","location":"ECE Seminar Hall","notes":"Discuss project progress and assign tasks."},
        {"id":"evt-002","title":"Python AI Workshop","event_type":"workshop","date":(s+timedelta(2)).isoformat(),"start_time":"14:00","end_time":"16:00","location":"Computer Lab 2","notes":"Hands-on workshop on AI agents and RAG."},
        {"id":"evt-003","title":"DBMS Assignment","event_type":"task","date":(s+timedelta(3)).isoformat(),"start_time":"18:00","end_time":"19:00","location":"Home","notes":"Complete normalization and SQL exercises."},
        {"id":"evt-004","title":"Faculty Appointment","event_type":"appointment","date":(s+timedelta(4)).isoformat(),"start_time":"11:30","end_time":"12:00","location":"Faculty Room","notes":"Discuss project guidance."},
        {"id":"evt-005","title":"IEEE Seminar","event_type":"seminar","date":(s+timedelta(6)).isoformat(),"start_time":"10:00","end_time":"12:00","location":"Auditorium","notes":"Seminar on electric vehicles and smart energy systems."},
        {"id":"evt-006","title":"Machine Learning Lab","event_type":"lab","date":(s+timedelta(8)).isoformat(),"start_time":"09:00","end_time":"11:00","location":"ML Lab","notes":"Complete classification experiment."},
        {"id":"evt-007","title":"Project Review","event_type":"meeting","date":(s+timedelta(10)).isoformat(),"start_time":"15:00","end_time":"16:30","location":"Project Lab","notes":"Internal project review and demonstration."},
        {"id":"evt-008","title":"Sports Practice","event_type":"activity","date":(s+timedelta(12)).isoformat(),"start_time":"17:00","end_time":"18:30","location":"College Ground","notes":"Team practice."},
        {"id":"evt-009","title":"Data Structures Test","event_type":"exam","date":(s+timedelta(15)).isoformat(),"start_time":"09:30","end_time":"11:00","location":"Classroom 204","notes":"Trees, graphs and sorting."},
        {"id":"evt-010","title":"Career Guidance Workshop","event_type":"workshop","date":(s+timedelta(20)).isoformat(),"start_time":"13:00","end_time":"15:00","location":"Auditorium","notes":"Resume, interview and placement preparation."},
    ]


def event_text(e):
    return (f"Title: {e['title']}. Type: {e['event_type']}. Date: {e['date']}. "
            f"Time: {e['start_time']} to {e['end_time']}. Location: {e.get('location','')}. "
            f"Notes: {e.get('notes','')}.")


def all_events():
    return collection.get(include=["metadatas"]).get("metadatas") or []


if collection.count() == 0:
    initial = samples()
    collection.add(ids=[e["id"] for e in initial], documents=[event_text(e) for e in initial], embeddings=[local_embedding(event_text(e)) for e in initial], metadatas=initial)


def format_events(events):
    if not events:
        return "No events are scheduled for that date."
    events = sorted(events, key=lambda e: (e.get("date", ""), e.get("start_time", "")))
    return "\n".join(f"{e['date']} | {e['start_time']}-{e['end_time']} | {e['title']} | {e['event_type']} | {e.get('location','')}" for e in events)


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        if not API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not configured in Render.")
        _llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, google_api_key=API_KEY)
    return _llm


@tool
def get_schedule(query: str) -> str:
    """Retrieve schedule information for dates, events and availability."""
    q = query.lower().strip()
    events = all_events()
    today = today_india()

    # Exact date questions must NEVER fall through to semantic RAG.
    if any(p in q for p in ["today's date", "todays date", "what is today's date", "what's today's date", "current date", "today date"]):
        return f"Today is {today.strftime('%A, %B %d, %Y')}."

    if "tomorrow" in q:
        target = today + timedelta(days=1)
        return format_events([e for e in events if e.get("date") == target.isoformat()])

    if re.search(r"\btoday\b", q):
        return format_events([e for e in events if e.get("date") == today.isoformat()])

    if "this week" in q:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return format_events([e for e in events if start.isoformat() <= e.get("date", "") <= end.isoformat()])

    if "next week" in q:
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        return format_events([e for e in events if start.isoformat() <= e.get("date", "") <= end.isoformat()])

    iso = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q)
    if iso:
        return format_events([e for e in events if e.get("date") == iso.group(0)])

    try:
        result = collection.query(query_embeddings=[local_embedding(query)], n_results=min(8, max(1, collection.count())), include=["metadatas"])
        return format_events(result.get("metadatas", [[]])[0])
    except Exception:
        return format_events(events[:8])


@tool
def update_schedule(action: str, event_id: Optional[str] = None, title: Optional[str] = None, event_type: Optional[str] = None, date_value: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, location: Optional[str] = None, notes: Optional[str] = None, search_query: Optional[str] = None) -> str:
    """Add, update or remove a schedule event."""
    action = action.lower().strip()
    if action not in {"add", "update", "remove"}:
        return "Invalid action. Use add, update, or remove."
    if action == "add":
        if not title or not date_value or not start_time:
            return "Please provide the event name, date and start time."
        eid = event_id or "evt-" + uuid.uuid4().hex[:8]
        event = {"id":eid,"title":title,"event_type":event_type or "meeting","date":date_value,"start_time":start_time,"end_time":end_time or start_time,"location":location or "","notes":notes or ""}
        doc = event_text(event)
        collection.upsert(ids=[eid], documents=[doc], embeddings=[local_embedding(doc)], metadatas=[event])
        return f"Added: {event['title']} on {event['date']} at {event['start_time']}."
    if not event_id and search_query:
        wanted = search_query.lower().strip()
        exact = [e for e in all_events() if wanted in e.get("title", "").lower()]
        if exact:
            event_id = exact[0]["id"]
        else:
            result = collection.query(query_embeddings=[local_embedding(search_query)], n_results=1, include=["metadatas"])
            found = result.get("metadatas", [[]])[0]
            if found:
                event_id = found[0]["id"]
    if not event_id:
        return "I could not identify that event."
    found = collection.get(ids=[event_id], include=["metadatas"]).get("metadatas") or []
    if not found:
        return "Event not found."
    if action == "remove":
        collection.delete(ids=[event_id])
        return f"Removed: {found[0]['title']} on {found[0]['date']}."
    event = dict(found[0])
    updates = {"title":title,"event_type":event_type,"date":date_value,"start_time":start_time,"end_time":end_time,"location":location,"notes":notes}
    event.update({k:v for k,v in updates.items() if v is not None})
    doc = event_text(event)
    collection.upsert(ids=[event_id], documents=[doc], embeddings=[local_embedding(doc)], metadatas=[event])
    return f"Updated: {event['title']} on {event['date']} at {event['start_time']}."


def parse_time(value):
    value = value.strip().upper().replace(".", "")
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?", value)
    if not m: return None
    hour = int(m.group(1)); minute = int(m.group(2) or 0); ap = m.group(3)
    if ap == "PM" and hour < 12: hour += 12
    if ap == "AM" and hour == 12: hour = 0
    if hour > 23 or minute > 59: return None
    return f"{hour:02d}:{minute:02d}"


def parse_date(value):
    q = value.lower(); today = today_india()
    if "tomorrow" in q: return today + timedelta(days=1)
    if "today" in q: return today
    m = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if m: return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d", "%b %d"):
        try:
            d = datetime.strptime(value.strip(), fmt).date()
            return d.replace(year=today.year) if d.year == 1900 else d
        except ValueError: pass
    return None


def fallback(msg):
    q = msg.strip(); low = q.lower()
    if any(x in low for x in ("today's date", "todays date", "what is today's date", "what's today's date", "current date", "today date")):
        return f"Today is {today_india().strftime('%A, %B %d, %Y')}."
    if any(x in low for x in ("remove ", "delete ", "cancel ")):
        m = re.search(r"(?:remove|delete|cancel)\s+(?:my\s+)?(.+)", q, re.I)
        if m: return update_schedule.invoke({"action":"remove", "search_query":m.group(1).strip().rstrip(".")})
    if any(x in low for x in ("add ", "schedule ", "plan ")):
        tm_match = re.search(r"\bat\s+(.+?)(?:\s+until\s+|\s+on\s+|$)", q, re.I)
        tm = parse_time(tm_match.group(1)) if tm_match else None
        if not tm:
            tm_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\b", q, re.I)
            tm = parse_time(tm_match.group(1)) if tm_match else None
        d = parse_date(q)
        if tm and d:
            title = re.sub(r"\b(add|schedule|plan)\b", "", q, count=1, flags=re.I)
            title = re.sub(r"\bon\b.+", "", title, flags=re.I).strip(" ,.")
            title = re.sub(r"\bat\b.+", "", title, flags=re.I).strip(" ,.")
            title = re.sub(r"^(a|an|the)\s+", "", title, flags=re.I).strip() or "Scheduled Event"
            return update_schedule.invoke({"action":"add","title":title.title(),"event_type":"personal","date_value":d.isoformat(),"start_time":tm,"end_time":tm})
    return get_schedule.invoke(q)


def agent(message):
    if not API_KEY:
        return fallback(message) + "\n\nNote: Gemini is not configured, so the local schedule engine handled this request."
    tools = [get_schedule, update_schedule]; tool_map = {t.name:t for t in tools}
    try:
        llm = get_llm().bind_tools(tools)
        system = ("You are an Agentic RAG Schedule Assistant. Use get_schedule for questions and availability. "
                  "Use update_schedule for adding, moving, editing or removing events. Never invent existing events. "
                  "Answer direct date questions directly. Understand natural language dates such as tomorrow, Friday, "
                  "next Monday and times such as evening. Current date in India is " + today_india().isoformat() + ".")
        messages = [("system",system),("human",message)]
        for _ in range(6):
            response = llm.invoke(messages); messages.append(response)
            calls = getattr(response,"tool_calls",[]) or []
            if not calls: return response.content if isinstance(response.content,str) else str(response.content)
            for call in calls:
                result = tool_map[call["name"]].invoke(call.get("args",{}))
                messages.append(ToolMessage(content=str(result),tool_call_id=call["id"]))
        return "The request was completed, but the assistant reached its tool limit."
    except Exception:
        return fallback(message)


class Request(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>Schedule Assistant</title><style>*{box-sizing:border-box}body{margin:0;font-family:Inter,Arial,sans-serif;background:linear-gradient(135deg,#eef2ff,#faf9ff);color:#17203a}.app{min-height:100vh;display:flex;gap:18px;padding:22px;max-width:1450px;margin:auto}.side{width:230px;background:#fff;border-radius:24px;padding:22px 14px;display:flex;flex-direction:column;box-shadow:0 15px 50px #5548e51c}.brand{font-size:23px;font-weight:800;padding:8px 12px 24px}.brand span{display:block;color:#6b4cff}.nav{padding:13px 14px;border-radius:13px;margin:4px 0;font-weight:650;color:#59647d;cursor:pointer}.nav:hover,.active{background:#f1edff;color:#684cff}.ai{margin-top:auto;background:#f7f3ff;border-radius:17px;padding:16px;color:#68728a}.ai b{color:#684cff}.main{flex:1;background:#fff;border-radius:28px;padding:28px;box-shadow:0 15px 55px #5548e51c;min-width:0}.title{font-size:32px;font-weight:850;letter-spacing:-1px}.sub{color:#7a849a;margin-top:5px}.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0 16px}.chip{border:0;border-radius:16px;padding:16px;background:#f5f6ff;text-align:left;font-weight:700;color:#303a53;cursor:pointer}.chat{height:450px;overflow:auto;border:1px dashed #dfe2ee;border-radius:21px;padding:20px;background:#fcfbff}.empty{text-align:center;padding:120px 20px;color:#6f7890}.robot{font-size:55px}.msg{max-width:78%;padding:14px 18px;border-radius:18px;margin:12px 0;line-height:1.5;white-space:pre-wrap}.user{margin-left:auto;background:linear-gradient(135deg,#674cff,#7958ff);color:white}.bot{background:white;border:1px solid #e0e3ed}.input{display:flex;gap:10px;margin-top:17px;border:2px solid #7456ff;border-radius:20px;padding:8px 10px}.input textarea{flex:1;border:0;outline:0;resize:none;height:62px;padding:12px;font:16px Arial}.send{width:56px;border:0;border-radius:15px;background:#6b4cff;color:white;font-size:23px;cursor:pointer}.hint{text-align:center;color:#8991a5;font-size:13px;margin-top:9px}.modal{display:none;position:fixed;inset:0;background:#10182d88;backdrop-filter:blur(5px);align-items:center;justify-content:center;padding:20px;z-index:10}.modal.show{display:flex}.form{width:min(570px,100%);background:#fff;border-radius:24px;padding:26px;box-shadow:0 30px 90px #0003}.form h2{margin:0}.form p{color:#788197}.field{margin-top:13px}.field label{display:block;font-size:13px;font-weight:700;margin-bottom:6px;color:#4d5870}.field input,.field select,.field textarea{width:100%;padding:12px;border:1px solid #dfe2eb;border-radius:11px;font:inherit}.field textarea{height:75px;resize:vertical}.grid{display:grid;grid-template-columns:1fr 1fr;gap:13px}.actions{display:flex;justify-content:flex-end;gap:10px;margin-top:20px}.cancel,.save{border:0;border-radius:11px;padding:12px 18px;font-weight:700;cursor:pointer}.cancel{background:#eef0f5}.save{background:#684cff;color:white}@media(max-width:850px){.side{display:none}.app{padding:10px}.main{padding:18px}.chips{grid-template-columns:1fr 1fr}.title{font-size:25px}}@media(max-width:520px){.chips{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.chat{height:420px}}</style></head><body><div class='app'><aside class='side'><div class='brand'>📅 Schedule<span>Assistant</span></div><div class='nav active'>💬 Chat Assistant</div><div class='nav' onclick='showSchedule()'>🗓️ My Schedule</div><div class='nav' onclick='openModal()'>➕ Add Event</div><div class='nav' onclick='showSchedule()'>📋 All Events</div><div class='nav'>⚙️ Settings</div><div class='ai'>✨ <b>AI Powered</b><br><small>Your personal schedule assistant</small></div></aside><main class='main'><div class='title'>Agentic RAG Schedule Assistant</div><div class='sub'>Plan, add, update and manage your schedule naturally.</div><div class='chips'><button class='chip' onclick="ask('What do I have scheduled tomorrow?')">🕐 What do I have tomorrow?</button><button class='chip' onclick="ask('What is my schedule this week?')">🗓️ This week's schedule</button><button class='chip' onclick='openModal()'>➕ Add a new event</button><button class='chip' onclick="ask('Find a free time tomorrow afternoon')">🔎 Find free time</button></div><div id='chat' class='chat'><div id='empty' class='empty'><div class='robot'>🤖</div><h2>Hello! 👋</h2><p>Ask about your schedule or create a plan using natural language.</p></div></div><div class='input'><textarea id='msg' placeholder='Try: What is today\'s date?'></textarea><button class='send' onclick='send()'>➤</button></div><div class='hint'>Questions • Appointments • Movies • Meetings • Tasks • Changes</div></main></div><div id='modal' class='modal'><div class='form'><h2>➕ Add to your schedule</h2><p>Nothing will be added until you press <b>Add to Schedule</b>.</p><div class='field'><label>Purpose / Event name *</label><input id='purpose' placeholder='Doctor appointment, Movie, Study session...'></div><div class='grid'><div class='field'><label>Date *</label><input id='eventDate' type='date'></div><div class='field'><label>Event type</label><select id='eventType'><option>appointment</option><option>meeting</option><option>task</option><option>activity</option><option>workshop</option><option>personal</option></select></div><div class='field'><label>Start time *</label><input id='start' type='time'></div><div class='field'><label>End time</label><input id='end' type='time'></div></div><div class='field'><label>Location</label><input id='location' placeholder='Hospital, Cinema, Home...'></div><div class='field'><label>Notes</label><textarea id='notes' placeholder='Optional details'></textarea></div><div class='actions'><button class='cancel' onclick='closeModal()'>Cancel</button><button class='save' onclick='saveEvent()'>Add to Schedule</button></div></div></div><script>const chat=document.getElementById('chat'),modal=document.getElementById('modal');function openModal(){modal.classList.add('show');document.getElementById('eventDate').value=new Date(Date.now()+86400000).toISOString().slice(0,10);document.getElementById('purpose').focus()}function closeModal(){modal.classList.remove('show')}modal.addEventListener('click',e=>{if(e.target===modal)closeModal()});function addMsg(t,c){document.getElementById('empty')?.remove();const d=document.createElement('div');d.className='msg '+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}async function ask(t){addMsg(t,'user');const b=addMsg('Thinking…','bot');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});const d=await r.json();b.textContent=d.answer||d.detail||'No response.'}catch(e){b.textContent='Unable to reach the assistant.'}}function send(){const x=document.getElementById('msg');const v=x.value.trim();if(!v)return;x.value='';ask(v)}document.getElementById('msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});function saveEvent(){const p=document.getElementById('purpose').value.trim(),d=document.getElementById('eventDate').value,s=document.getElementById('start').value,e=document.getElementById('end').value,t=document.getElementById('eventType').value,l=document.getElementById('location').value.trim(),n=document.getElementById('notes').value.trim();if(!p||!d||!s){alert('Purpose, date and start time are required.');return}let q=`Add a ${t} called ${p} on ${d} at ${s}`;if(e)q+=` until ${e}`;if(l)q+=` at ${l}`;if(n)q+=`. Notes: ${n}`;closeModal();ask(q)}function showSchedule(){ask('Show my schedule for the next 30 days.')}</script></body></html>""")


@app.post("/chat")
def chat(request: Request):
    try:
        answer = agent(request.message.strip())
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Schedule assistant error: {exc}")


@app.get("/health")
def health():
    return {"status":"ok","events":collection.count(),"today":today_india().isoformat(),"timezone":"Asia/Kolkata","gemini_configured":bool(API_KEY)}
