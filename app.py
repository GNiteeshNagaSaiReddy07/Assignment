import os, re, uuid, hashlib
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TZ = ZoneInfo("Asia/Kolkata")
DB_PATH = os.getenv("CHROMA_DIR", "./chroma_db")

app = FastAPI(title="Agentic RAG Schedule Assistant", version="6.0")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("schedule", metadata={"hnsw:space": "cosine"})


def today():
    return datetime.now(TZ).date()


def embedding(text, size=96):
    values = [0.0] * size
    words = re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(0, len(digest), 2):
            values[int.from_bytes(digest[i:i + 2], "big") % size] += 1.0
    norm = sum(x * x for x in values) ** 0.5 or 1.0
    return [x / norm for x in values]


def seed_events():
    d = today()
    return [
        {"id":"evt-001","title":"Project Team Meeting","event_type":"meeting","date":(d+timedelta(1)).isoformat(),"start_time":"10:00","end_time":"11:00","location":"ECE Seminar Hall","notes":"Project progress and tasks."},
        {"id":"evt-002","title":"Python AI Workshop","event_type":"workshop","date":(d+timedelta(2)).isoformat(),"start_time":"14:00","end_time":"16:00","location":"Computer Lab 2","notes":"AI agents and RAG."},
        {"id":"evt-003","title":"DBMS Assignment","event_type":"task","date":(d+timedelta(3)).isoformat(),"start_time":"18:00","end_time":"19:00","location":"Home","notes":"Normalization and SQL exercises."},
        {"id":"evt-004","title":"Faculty Appointment","event_type":"appointment","date":(d+timedelta(4)).isoformat(),"start_time":"11:30","end_time":"12:00","location":"Faculty Room","notes":"Project guidance."},
        {"id":"evt-005","title":"IEEE Seminar","event_type":"seminar","date":(d+timedelta(6)).isoformat(),"start_time":"10:00","end_time":"12:00","location":"Auditorium","notes":"EV and smart energy systems."},
        {"id":"evt-006","title":"Machine Learning Lab","event_type":"lab","date":(d+timedelta(8)).isoformat(),"start_time":"09:00","end_time":"11:00","location":"ML Lab","notes":"Classification experiment."},
        {"id":"evt-007","title":"Project Review","event_type":"meeting","date":(d+timedelta(10)).isoformat(),"start_time":"15:00","end_time":"16:30","location":"Project Lab","notes":"Project demonstration."},
        {"id":"evt-008","title":"Sports Practice","event_type":"activity","date":(d+timedelta(12)).isoformat(),"start_time":"17:00","end_time":"18:30","location":"College Ground","notes":"Team practice."},
        {"id":"evt-009","title":"Data Structures Test","event_type":"exam","date":(d+timedelta(15)).isoformat(),"start_time":"09:30","end_time":"11:00","location":"Classroom 204","notes":"Trees, graphs and sorting."},
        {"id":"evt-010","title":"Career Guidance Workshop","event_type":"workshop","date":(d+timedelta(20)).isoformat(),"start_time":"13:00","end_time":"15:00","location":"Auditorium","notes":"Resume and placement preparation."},
    ]


def event_text(e):
    return f"{e['title']} | {e['event_type']} | {e['date']} | {e['start_time']}-{e['end_time']} | {e.get('location','')} | {e.get('notes','')}"


def all_events():
    return collection.get(include=["metadatas"]).get("metadatas") or []


if collection.count() == 0:
    initial = seed_events()
    collection.add(
        ids=[e["id"] for e in initial],
        documents=[event_text(e) for e in initial],
        embeddings=[embedding(event_text(e)) for e in initial],
        metadatas=initial,
    )


def format_events(items):
    if not items:
        return "No events are scheduled for that date."
    items = sorted(items, key=lambda x: (x.get("date", ""), x.get("start_time", "")))
    return "\n".join(
        f"{e['date']} | {e['start_time']}-{e['end_time']} | {e['title']} | {e['event_type']} | {e.get('location','')}"
        for e in items
    )


@tool
def get_schedule(query: str) -> str:
    """Retrieve schedule information using date filters and ChromaDB RAG."""
    q = query.lower().strip()
    items = all_events()
    d = today()
    if any(x in q for x in ["today's date", "todays date", "what is today's date", "what's today's date", "current date", "today date"]):
        return f"Today is {d.strftime('%A, %B %d, %Y')}."
    if "tomorrow" in q:
        target = d + timedelta(days=1)
        return format_events([e for e in items if e["date"] == target.isoformat()])
    if re.search(r"\btoday\b", q):
        return format_events([e for e in items if e["date"] == d.isoformat()])
    if "this week" in q:
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        return format_events([e for e in items if start.isoformat() <= e["date"] <= end.isoformat()])
    if "next week" in q:
        start = d - timedelta(days=d.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        return format_events([e for e in items if start.isoformat() <= e["date"] <= end.isoformat()])
    match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", q)
    if match:
        return format_events([e for e in items if e["date"] == match.group(0)])
    try:
        result = collection.query(query_embeddings=[embedding(query)], n_results=min(8, collection.count()), include=["metadatas"])
        return format_events(result.get("metadatas", [[]])[0])
    except Exception:
        return format_events(items[:8])


@tool
def update_schedule(action: str, event_id: Optional[str] = None, title: Optional[str] = None, event_type: Optional[str] = None, date_value: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, location: Optional[str] = None, notes: Optional[str] = None, search_query: Optional[str] = None) -> str:
    """Add, update or remove schedule events."""
    action = action.lower().strip()
    if action not in {"add", "update", "remove"}:
        return "Invalid action."
    if action == "add":
        if not title or not date_value or not start_time:
            return "Please provide the event name, date and start time."
        eid = event_id or "evt-" + uuid.uuid4().hex[:8]
        event = {"id":eid,"title":title,"event_type":event_type or "personal","date":date_value,"start_time":start_time,"end_time":end_time or start_time,"location":location or "","notes":notes or ""}
        doc = event_text(event)
        collection.upsert(ids=[eid], documents=[doc], embeddings=[embedding(doc)], metadatas=[event])
        return f"Added: {event['title']} on {event['date']} at {event['start_time']}."
    if not event_id and search_query:
        wanted = search_query.lower().strip()
        exact = [e for e in all_events() if wanted in e.get("title", "").lower()]
        if exact:
            event_id = exact[0]["id"]
        else:
            result = collection.query(query_embeddings=[embedding(search_query)], n_results=1, include=["metadatas"])
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
    values = {"title":title,"event_type":event_type,"date":date_value,"start_time":start_time,"end_time":end_time,"location":location,"notes":notes}
    event.update({k:v for k,v in values.items() if v is not None})
    doc = event_text(event)
    collection.upsert(ids=[event_id], documents=[doc], embeddings=[embedding(doc)], metadatas=[event])
    return f"Updated: {event['title']} on {event['date']} at {event['start_time']}."


llm = None


def get_llm():
    global llm
    if llm is None:
        if not API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not configured.")
        llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, google_api_key=API_KEY)
    return llm


def parse_time(text):
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?\b", text.upper())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ap = match.group(3)
    if ap == "PM" and hour < 12:
        hour += 12
    if ap == "AM" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def parse_date(text):
    q = text.lower()
    d = today()
    if "tomorrow" in q:
        return d + timedelta(days=1)
    if "today" in q:
        return d
    match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    for fmt in ("%B %d %Y", "%b %d %Y", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(text.strip(), fmt).date()
            return parsed.replace(year=d.year) if parsed.year == 1900 else parsed
        except ValueError:
            continue
    return None


def fallback(message):
    q = message.strip()
    low = q.lower()
    if any(x in low for x in ["today's date", "todays date", "what is today's date", "what's today's date", "current date", "today date"]):
        return f"Today is {today().strftime('%A, %B %d, %Y')}."
    if any(x in low for x in ["remove ", "delete ", "cancel "]):
        match = re.search(r"(?:remove|delete|cancel)\s+(?:my\s+)?(.+)", q, re.I)
        if match:
            return update_schedule.invoke({"action":"remove","search_query":match.group(1).rstrip(".")})
    if any(x in low for x in ["add ", "schedule ", "plan ", "book "]):
        d = parse_date(q)
        tm = parse_time(q)
        if d and tm:
            title = re.sub(r"^(add|schedule|plan|book)\s+", "", q, flags=re.I)
            title = re.split(r"\s+(?:on|at|for)\s+", title, maxsplit=1, flags=re.I)[0].strip()
            title = title or "Scheduled Event"
            kind = "appointment" if any(x in low for x in ["doctor", "dentist", "appointment"]) else "personal"
            return update_schedule.invoke({"action":"add","title":title.title(),"event_type":kind,"date_value":d.isoformat(),"start_time":tm,"end_time":tm})
    return get_schedule.invoke(q)


def agent(message):
    if not API_KEY:
        return fallback(message) + "\n\nGemini is not configured; local schedule mode was used."
    try:
        tools = [get_schedule, update_schedule]
        tool_map = {x.name: x for x in tools}
        model = get_llm().bind_tools(tools)
        system = "You are an Agentic RAG Schedule Assistant. Use get_schedule for questions and update_schedule for adding, changing or removing events. Never invent existing events. Understand natural language dates and times. Current India date is " + today().isoformat() + "."
        messages = [("system", system), ("human", message)]
        for _ in range(6):
            response = model.invoke(messages)
            messages.append(response)
            calls = getattr(response, "tool_calls", []) or []
            if not calls:
                return response.content if isinstance(response.content, str) else str(response.content)
            for call in calls:
                result = tool_map[call["name"]].invoke(call.get("args", {}))
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    except Exception:
        return fallback(message)


class Request(BaseModel):
    message: str


HTML = '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule Assistant</title><style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:linear-gradient(135deg,#f0efff,#fbfbff);color:#17204a}.app{min-height:100vh;display:grid;grid-template-columns:235px 1fr}.side{background:linear-gradient(180deg,#17105b,#251879);color:#fff;padding:22px 14px;display:flex;flex-direction:column}.logo{font-size:21px;font-weight:800;padding:8px 14px 25px}.nav{padding:12px 14px;margin:3px 0;border-radius:12px;color:#ddd9ff;cursor:pointer}.nav:hover,.nav.on{background:linear-gradient(90deg,#6347f5,#5b43ee);color:#fff}.profile{margin-top:auto;border-top:1px solid #ffffff22;padding:18px 8px}.main{padding:28px;min-width:0}.top,.hero,.head{display:flex;justify-content:space-between;align-items:center}.top h1{margin:0}.date{background:#fff;border:1px solid #e8e8f2;padding:11px 16px;border-radius:13px;font-weight:700}.hero{margin:20px 0}.hero h2{margin:0 0 4px;font-size:28px}.muted{color:#737b98}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}.card{background:#fff;border:1px solid #e8e8f2;border-radius:18px;box-shadow:0 12px 35px #4e45a30b}.stat{padding:17px;display:flex;gap:12px;align-items:center}.stat strong{font-size:25px;display:block}.stat small{color:#737b98}.ico{width:45px;height:45px;border-radius:14px;background:#f0edff;display:grid;place-items:center;font-size:21px}.grid{display:grid;grid-template-columns:1.35fr .9fr 320px;gap:15px}.panel{padding:17px}.head h3{margin:0;font-size:16px}.event{display:grid;grid-template-columns:88px 1fr auto;gap:10px;padding:13px 6px;border-bottom:1px solid #f0f0f5}.time{color:#6347f5;font-weight:800;font-size:12px}.event b{font-size:14px}.event small{display:block;color:#737b98;margin-top:4px}.tag{font-size:10px;background:#f0edff;color:#6347f5;padding:5px 7px;border-radius:8px;font-weight:800;height:max-content}.calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:4px;text-align:center;margin-top:12px}.calendar div{padding:7px;font-size:12px}.dow{color:#8b91a8;font-weight:700}.sel{background:#6347f5;color:#fff;border-radius:9px;font-weight:800}.upitem{padding:11px 3px;border-bottom:1px solid #f0f0f5}.upitem b{font-size:13px}.upitem small{display:block;color:#737b98;margin-top:3px}.assistant{padding:0;overflow:hidden}.ahead{background:linear-gradient(135deg,#5d43ef,#7651ff);color:#fff;padding:18px}.ahead h3{margin:0 0 3px}.messages{height:355px;overflow:auto;padding:13px;background:#fcfcff}.bubble{max-width:90%;padding:10px 12px;border-radius:14px;margin:9px 0;font-size:13px;white-space:pre-wrap;line-height:1.45}.bot{background:#fff;border:1px solid #e8e8f2}.user{margin-left:auto;background:#6a4cf6;color:#fff}.quick{padding:10px;display:grid;gap:7px}.quick button,.action{background:#fff;border:1px solid #d7d0ff;color:#5140c8;border-radius:10px;padding:9px;cursor:pointer}.chat{display:flex;gap:7px;padding:9px;border-top:1px solid #eee}.chat textarea{flex:1;border:1px solid #e5e5ef;border-radius:10px;padding:10px;resize:none;height:43px}.send{width:44px;border:0;border-radius:10px;background:#6347f5;color:#fff;font-size:19px}.actions{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:16px}.action{text-align:left;padding:15px;font-weight:750}.modal{display:none;position:fixed;inset:0;background:#10182d99;align-items:center;justify-content:center;padding:15px;z-index:10}.modal.show{display:flex}.form{background:#fff;border-radius:20px;padding:22px;width:min(550px,100%)}.form h2{margin:0}.field{margin-top:11px}.field label{display:block;font-size:12px;font-weight:800;margin-bottom:5px}.field input,.field select,.field textarea{width:100%;padding:10px;border:1px solid #e5e5ef;border-radius:9px}.field textarea{height:65px}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.formbuttons{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}.formbuttons button{border:0;border-radius:9px;padding:10px 15px;font-weight:800}.save{background:#6347f5;color:#fff}.cancel{background:#eeeef5}@media(max-width:1200px){.grid{grid-template-columns:1.2fr .8fr}.assistant{grid-column:1/-1}.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:800px){.app{grid-template-columns:1fr}.side{display:none}.main{padding:14px}.grid{grid-template-columns:1fr}.actions{grid-template-columns:1fr 1fr}.hero{display:block}}@media(max-width:500px){.stats{grid-template-columns:1fr}.actions{grid-template-columns:1fr}.two{grid-template-columns:1fr}}
</style></head><body><div class="app"><aside class="side"><div class="logo">📅 Schedule<br>Assistant</div><div class="nav on">🏠 Dashboard</div><div class="nav" onclick="ask('Show my schedule for the next 30 days')">🗓️ My Schedule</div><div class="nav" onclick="openModal()">➕ Add Event</div><div class="nav" onclick="ask('Find a free time tomorrow')">✓ Check Availability</div><div class="nav" onclick="ask('What are my reminders?')">🔔 Reminders</div><div class="nav" onclick="ask('Show my schedule for the next 30 days')">📊 Analytics</div><div class="nav">⚙️ Settings</div><div class="profile">👤 <b>Schedule User</b><br><small>Personal planner</small></div></aside><main class="main"><div class="top"><h1>Dashboard</h1><div class="date">📅 <span id="dateLabel"></span></div></div><div class="hero"><div><h2>Good evening! 👋</h2><div class="muted">Here’s your schedule overview.</div></div></div><div class="stats"><div class="card stat"><div class="ico">📅</div><div><strong id="total">0</strong><small>Total Events<br>Next 30 days</small></div></div><div class="card stat"><div class="ico">☀️</div><div><strong id="todayCount">0</strong><small>Today<br>Scheduled</small></div></div><div class="card stat"><div class="ico">⏰</div><div><strong id="weekCount">0</strong><small>Next 7 days<br>Upcoming</small></div></div><div class="card stat"><div class="ico">✨</div><div><strong>AI</strong><small>Assistant<br>Online</small></div></div></div><div class="grid"><section class="card panel"><div class="head"><h3>📋 Today's Schedule</h3><button class="quick" style="border:0;background:none" onclick="ask('What do I have today?')">Ask AI →</button></div><div id="todayList"></div></section><section class="card panel"><div class="head"><h3>🗓️ Calendar</h3><span id="month"></span></div><div id="calendar" class="calendar"></div><div style="margin-top:18px"><div class="head"><h3>Upcoming Events</h3></div><div id="upcoming"></div></div></section><section class="card assistant"><div class="ahead"><h3>✨ AI Schedule Assistant</h3><small>● Online</small></div><div id="messages" class="messages"><div class="bubble bot">Hello! I can manage your schedule. Ask me about today, tomorrow, availability, or create a new plan.</div></div><div class="quick"><button onclick="ask('What is today\'s date?')">📅 Today's date</button><button onclick="ask('What do I have today?')">🗓️ What do I have today?</button><button onclick="openModal()">➕ Add a new event</button><button onclick="ask('Find a free time tomorrow afternoon')">✓ Check availability</button><button onclick="ask('What is my schedule this week?')">📆 This week</button></div><div class="chat"><textarea id="message" placeholder="Type your message..."></textarea><button class="send" onclick="send()">➤</button></div></section></div><div class="actions"><button class="action" onclick="openModal()">📅<br>Add Event</button><button class="action" onclick="ask('Find a free time tomorrow')">🟢<br>Check Availability</button><button class="action" onclick="ask('What is my schedule this week?')">🗓️<br>View This Week</button><button class="action" onclick="ask('What are my reminders?')">🔔<br>Reminders</button><button class="action" onclick="ask('Show my schedule for the next 30 days')">📊<br>All Events</button></div></main></div><div id="modal" class="modal"><div class="form"><h2>➕ Add to your schedule</h2><p class="muted">Nothing is added until you confirm.</p><div class="field"><label>Purpose / Event name *</label><input id="purpose" placeholder="Doctor appointment, Movie, Study session..."></div><div class="two"><div class="field"><label>Date *</label><input id="eventDate" type="date"></div><div class="field"><label>Type</label><select id="eventType"><option>appointment</option><option>meeting</option><option>task</option><option>activity</option><option>personal</option><option>workshop</option></select></div><div class="field"><label>Start time *</label><input id="start" type="time"></div><div class="field"><label>End time</label><input id="end" type="time"></div></div><div class="field"><label>Location</label><input id="location" placeholder="Hospital, Cinema, Home..."></div><div class="field"><label>Notes</label><textarea id="notes" placeholder="Optional details"></textarea></div><div class="formbuttons"><button class="cancel" onclick="closeModal()">Cancel</button><button class="save" onclick="saveEvent()">Add to Schedule</button></div></div></div><script>
const modal=document.getElementById('modal'),messages=document.getElementById('messages');
function openModal(){modal.classList.add('show');document.getElementById('eventDate').value=new Date(Date.now()+86400000).toISOString().slice(0,10);document.getElementById('purpose').focus()}
function closeModal(){modal.classList.remove('show')}
modal.onclick=e=>{if(e.target===modal)closeModal()}
function bubble(text,cls){const x=document.createElement('div');x.className='bubble '+cls;x.textContent=text;messages.appendChild(x);messages.scrollTop=messages.scrollHeight;return x}
async function ask(text){bubble(text,'user');const b=bubble('Thinking…','bot');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});const data=await r.json();b.textContent=data.answer||data.detail||'No response.';load()}catch(e){b.textContent='Unable to reach the assistant.'}}
function send(){const x=document.getElementById('message');const v=x.value.trim();if(v){x.value='';ask(v)}}
document.getElementById('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}})
function saveEvent(){const p=document.getElementById('purpose').value.trim(),d=document.getElementById('eventDate').value,s=document.getElementById('start').value,e=document.getElementById('end').value,t=document.getElementById('eventType').value,l=document.getElementById('location').value.trim(),n=document.getElementById('notes').value.trim();if(!p||!d||!s){alert('Purpose, date and start time are required.');return}let q='Schedule a '+t+' called '+p+' on '+d+' at '+s+(e?' until '+e:'')+(l?' at location '+l:'')+(n?'. Notes: '+n:'');closeModal();ask(q)}
function fmtTime(s){if(!s)return '';let a=s.split(':').map(Number),h=a[0],m=a[1],ap=h>=12?'PM':'AM';h=h%12||12;return h+':'+String(m).padStart(2,'0')+' '+ap}
async function load(){try{const r=await fetch('/api/events');const d=await r.json();document.getElementById('total').textContent=d.total;document.getElementById('todayCount').textContent=d.today.length;document.getElementById('weekCount').textContent=d.next7.length;document.getElementById('todayList').innerHTML=d.today.length?d.today.map(e=>'<div class="event"><div class="time">'+fmtTime(e.start_time)+'<br>– '+fmtTime(e.end_time)+'</div><div><b>'+e.title+'</b><small>⌖ '+(e.location||'No location')+'</small></div><span class="tag">'+e.event_type+'</span></div>').join(''):'<div style="padding:35px;text-align:center;color:#858ca3">No events today 🎉</div>';document.getElementById('upcoming').innerHTML=d.upcoming.slice(0,4).map(e=>'<div class="upitem"><b>'+e.title+'</b><small>'+e.date+' · '+fmtTime(e.start_time)+' · '+(e.location||'No location')+'</small></div>').join('')||'<small>No upcoming events.</small>'}catch(e){}}
function calendar(){const n=new Date(),y=n.getFullYear(),m=n.getMonth(),first=new Date(y,m,1).getDay(),days=new Date(y,m+1,0).getDate();document.getElementById('dateLabel').textContent=n.toLocaleDateString('en-IN',{weekday:'long',month:'long',day:'numeric',year:'numeric'});document.getElementById('month').textContent=n.toLocaleString('en',{month:'long',year:'numeric'});let h=['Su','Mo','Tu','We','Th','Fr','Sa'].map(x=>'<div class="dow">'+x+'</div>').join('');for(let i=0;i<first;i++)h+='<div></div>';for(let d=1;d<=days;d++)h+='<div class="'+(d===n.getDate()?'sel':'')+'">'+d+'</div>';document.getElementById('calendar').innerHTML=h}
calendar();load();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(HTML)


@app.post("/chat")
def chat(request: Request):
    try:
        return {"answer": agent(request.message.strip())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/events")
def api_events():
    d = today()
    items = sorted(all_events(), key=lambda e: (e["date"], e["start_time"]))
    return {
        "total": len(items),
        "today": [e for e in items if e["date"] == d.isoformat()],
        "next7": [e for e in items if d <= date.fromisoformat(e["date"]) <= d + timedelta(days=7)],
        "upcoming": [e for e in items if date.fromisoformat(e["date"]) >= d],
    }


@app.get("/health")
def health():
    return {"status":"ok","events":collection.count(),"today":today().isoformat(),"timezone":"Asia/Kolkata","gemini_configured":bool(API_KEY)}
