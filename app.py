import os, re, uuid
from datetime import date, timedelta, datetime
from typing import Optional
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

API_KEY = os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise RuntimeError('GOOGLE_API_KEY is not set.')

app = FastAPI(title='Agentic RAG Schedule Assistant', version='2.0')
client = chromadb.PersistentClient(path=os.getenv('CHROMA_DIR', './chroma_db'))
emb = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001', google_api_key=API_KEY)
col = client.get_or_create_collection('schedule', metadata={'hnsw:space': 'cosine'})
llm = ChatGoogleGenerativeAI(
    model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
    temperature=0,
    google_api_key=API_KEY,
)


def samples():
    s = date.today()
    return [
        {'id':'evt-001','title':'Project Team Meeting','event_type':'meeting','date':(s+timedelta(1)).isoformat(),'start_time':'10:00','end_time':'11:00','location':'ECE Seminar Hall','notes':'Discuss project progress and assign tasks.'},
        {'id':'evt-002','title':'Python AI Workshop','event_type':'workshop','date':(s+timedelta(2)).isoformat(),'start_time':'14:00','end_time':'16:00','location':'Computer Lab 2','notes':'Hands-on workshop on AI agents and RAG.'},
        {'id':'evt-003','title':'DBMS Assignment','event_type':'task','date':(s+timedelta(3)).isoformat(),'start_time':'18:00','end_time':'19:00','location':'Home','notes':'Complete normalization and SQL exercises.'},
        {'id':'evt-004','title':'Faculty Appointment','event_type':'appointment','date':(s+timedelta(4)).isoformat(),'start_time':'11:30','end_time':'12:00','location':'Faculty Room','notes':'Discuss project guidance.'},
        {'id':'evt-005','title':'IEEE Seminar','event_type':'seminar','date':(s+timedelta(6)).isoformat(),'start_time':'10:00','end_time':'12:00','location':'Auditorium','notes':'Seminar on electric vehicles and smart energy systems.'},
        {'id':'evt-006','title':'Machine Learning Lab','event_type':'lab','date':(s+timedelta(8)).isoformat(),'start_time':'09:00','end_time':'11:00','location':'ML Lab','notes':'Complete classification experiment.'},
        {'id':'evt-007','title':'Project Review','event_type':'meeting','date':(s+timedelta(10)).isoformat(),'start_time':'15:00','end_time':'16:30','location':'Project Lab','notes':'Internal project review and demonstration.'},
        {'id':'evt-008','title':'Sports Practice','event_type':'activity','date':(s+timedelta(12)).isoformat(),'start_time':'17:00','end_time':'18:30','location':'College Ground','notes':'Team practice.'},
        {'id':'evt-009','title':'Data Structures Test','event_type':'exam','date':(s+timedelta(15)).isoformat(),'start_time':'09:30','end_time':'11:00','location':'Classroom 204','notes':'Trees, graphs and sorting.'},
        {'id':'evt-010','title':'Career Guidance Workshop','event_type':'workshop','date':(s+timedelta(20)).isoformat(),'start_time':'13:00','end_time':'15:00','location':'Auditorium','notes':'Resume, interview and placement preparation.'},
    ]


def text(e):
    return f"Title: {e['title']}. Type: {e['event_type']}. Date: {e['date']}. Time: {e['start_time']} to {e['end_time']}. Location: {e.get('location','')}. Notes: {e.get('notes','')}."


if col.count() == 0:
    ev = samples()
    docs = [text(e) for e in ev]
    vec = emb.embed_documents(docs)
    col.add(ids=[e['id'] for e in ev], documents=docs, embeddings=vec, metadatas=ev)


def format_events(events):
    if not events:
        return 'No matching schedule entries were found.'
    return '\n'.join(
        f"{m['id']} | {m['date']} | {m['start_time']}-{m['end_time']} | {m['title']} | {m['event_type']} | {m.get('location','')} | {m.get('notes','')}"
        for m in events
    )


def all_events():
    return col.get(include=['metadatas']).get('metadatas') or []


@tool
def get_schedule(query: str) -> str:
    """Retrieve schedule information using date-aware filtering and semantic RAG."""
    q = query.lower().strip()
    events = all_events()
    today = date.today()
    selected = []

    if 'tomorrow' in q:
        target = today + timedelta(days=1)
        selected = [e for e in events if e.get('date') == target.isoformat()]
    elif 'today' in q:
        selected = [e for e in events if e.get('date') == today.isoformat()]
    elif 'this week' in q or 'week schedule' in q:
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        selected = [e for e in events if start.isoformat() <= e.get('date','') <= end.isoformat()]
    elif 'next week' in q:
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        end = start + timedelta(days=6)
        selected = [e for e in events if start.isoformat() <= e.get('date','') <= end.isoformat()]

    iso = re.search(r'\b20\d{2}-\d{2}-\d{2}\b', q)
    if iso:
        selected = [e for e in events if e.get('date') == iso.group(0)]

    if selected:
        selected.sort(key=lambda e: (e.get('date',''), e.get('start_time','')))
        return format_events(selected)

    try:
        r = col.query(query_embeddings=[emb.embed_query(query)], n_results=min(8, max(1, col.count())), include=['metadatas'])
        return format_events(r.get('metadatas', [[]])[0])
    except Exception:
        return format_events(events[:8])


@tool
def update_schedule(action: str, event_id: Optional[str]=None, title: Optional[str]=None,
                    event_type: Optional[str]=None, date_value: Optional[str]=None,
                    start_time: Optional[str]=None, end_time: Optional[str]=None,
                    location: Optional[str]=None, notes: Optional[str]=None,
                    search_query: Optional[str]=None) -> str:
    """Add, update, or remove a schedule entry. Use YYYY-MM-DD and HH:MM."""
    action = action.lower().strip()
    if action not in ('add', 'update', 'remove'):
        return 'Invalid action. Use add, update, or remove.'
    if action == 'add':
        if not title or not date_value or not start_time:
            return 'title, date_value and start_time are required.'
        eid = event_id or 'evt-' + uuid.uuid4().hex[:8]
        e = {'id':eid,'title':title,'event_type':event_type or 'meeting','date':date_value,
             'start_time':start_time,'end_time':end_time or start_time,'location':location or '', 'notes':notes or ''}
        t = text(e)
        col.upsert(ids=[eid], documents=[t], embeddings=[emb.embed_documents([t])[0]], metadatas=[e])
        return 'Added: ' + t

    if not event_id and search_query:
        q = search_query.lower()
        events = all_events()
        exact = [e for e in events if q in e.get('title','').lower()]
        if exact:
            event_id = exact[0]['id']
        else:
            r = col.query(query_embeddings=[emb.embed_query(search_query)], n_results=1, include=['metadatas'])
            ms = r.get('metadatas', [[]])[0]
            if ms:
                event_id = ms[0]['id']

    if not event_id:
        return 'Could not identify the event.'
    r = col.get(ids=[event_id], include=['metadatas'])
    ms = r.get('metadatas') or []
    if not ms:
        return 'Event not found.'
    if action == 'remove':
        col.delete(ids=[event_id])
        return f"Removed: {ms[0]['title']} on {ms[0]['date']}"

    e = dict(ms[0])
    e.update({k:v for k,v in {'title':title,'event_type':event_type,'date':date_value,
                              'start_time':start_time,'end_time':end_time,'location':location,'notes':notes}.items() if v is not None})
    t = text(e)
    col.upsert(ids=[event_id], documents=[t], embeddings=[emb.embed_documents([t])[0]], metadatas=[e])
    return 'Updated: ' + t


tools = [get_schedule, update_schedule]
tmap = {t.name: t for t in tools}
model = llm.bind_tools(tools)
SYSTEM = '''You are an Agentic RAG Schedule Assistant. Use get_schedule for existing schedule/date/time/availability questions. Use update_schedule for add, move, edit, or remove requests. For ambiguous move/remove, retrieve first. Never invent existing events. Convert dates to YYYY-MM-DD and times to HH:MM. Current date is {today}. Keep answers concise and helpful.'''


def parse_time(value):
    value = value.strip().upper().replace('.', '')
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?', value)
    if not m:
        return None
    h = int(m.group(1)); minute = int(m.group(2) or 0); ap = m.group(3)
    if ap == 'PM' and h < 12: h += 12
    if ap == 'AM' and h == 12: h = 0
    if h > 23 or minute > 59: return None
    return f'{h:02d}:{minute:02d}'


def fallback(msg):
    q = msg.strip()
    low = q.lower()
    # Common remove request fallback
    m = re.search(r'(?:remove|delete|cancel)\s+(?:my\s+)?(.+)', low)
    if m:
        title = m.group(1).strip().rstrip('.')
        return update_schedule.invoke({'action':'remove','search_query':title})

    # Common add request fallback: Add a meeting on August 20 at 3 PM
    m = re.search(r'add\s+(?:a\s+)?(.+?)\s+on\s+([A-Za-z]+\s+\d{1,2})(?:,\s*(20\d{2}))?\s+at\s+(.+)', q, re.I)
    if m:
        title = m.group(1).strip()
        month_day = m.group(2)
        year = int(m.group(3) or date.today().year)
        try:
            d = datetime.strptime(f'{month_day} {year}', '%B %d %Y').date()
        except ValueError:
            try:
                d = datetime.strptime(f'{month_day} {year}', '%b %d %Y').date()
            except ValueError:
                d = None
        tm = parse_time(m.group(4))
        if d and tm:
            return update_schedule.invoke({'action':'add','title':title,'event_type':'meeting','date_value':d.isoformat(),'start_time':tm,'end_time':tm})

    # Common move request fallback
    m = re.search(r'move\s+(?:my\s+)?(.+?)\s+to\s+(.+)', q, re.I)
    if m:
        title = m.group(1).strip().rstrip('.')
        tm = parse_time(m.group(2))
        if tm:
            events = all_events()
            match = next((e for e in events if title.lower() in e.get('title','').lower()), None)
            if match:
                return update_schedule.invoke({'action':'update','event_id':match['id'],'start_time':tm,'end_time':tm})

    return get_schedule.invoke(q)


def agent(msg):
    messages = [('system', SYSTEM.format(today=date.today().isoformat())), ('human', msg)]
    try:
        for _ in range(6):
            res = model.invoke(messages)
            messages.append(res)
            calls = getattr(res, 'tool_calls', []) or []
            if not calls:
                return res.content if isinstance(res.content, str) else str(res.content)
            for c in calls:
                result = tmap[c['name']].invoke(c.get('args', {}))
                messages.append(ToolMessage(content=str(result), tool_call_id=c['id']))
        return 'I completed the request but reached the tool-call limit.'
    except Exception:
        # Keep the deployed app useful even if Gemini temporarily rejects a request.
        return fallback(msg)


class Req(BaseModel):
    message: str


@app.get('/', response_class=HTMLResponse)
def home():
    return '''<!doctype html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule AI</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:linear-gradient(135deg,#eef2ff,#f8f7ff 55%,#f0ebff);color:#17203a}.app{min-height:100vh;display:flex;max-width:1500px;margin:auto;padding:24px;gap:20px}.side{width:245px;background:rgba(255,255,255,.82);border:1px solid #fff;border-radius:24px;padding:24px 16px;box-shadow:0 15px 50px #4f46e51c;display:flex;flex-direction:column}.brand{font-size:23px;font-weight:800;padding:10px 12px 28px}.brand span{display:block;color:#6d4cff}.nav{padding:13px 14px;border-radius:14px;margin:4px 0;color:#59627a;font-weight:600}.nav.active{background:#f0ebff;color:#684cff}.ai{margin-top:auto;padding:18px;border-radius:18px;background:linear-gradient(145deg,#f5f0ff,#fff);border:1px solid #eee7ff}.ai b{color:#684cff}.main{flex:1;min-width:0;background:rgba(255,255,255,.88);border-radius:28px;padding:30px;box-shadow:0 18px 60px #4f46e51a;border:1px solid #fff}.top{display:flex;justify-content:space-between;align-items:center}.title{font-size:34px;font-weight:850;letter-spacing:-1px}.sub{color:#707a92;margin-top:7px}.theme{width:44px;height:44px;border:1px solid #e9e7f2;background:#fff;border-radius:50%;font-size:20px}.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 18px}.chip{border:0;border-radius:17px;padding:17px;text-align:left;background:#f4f6ff;color:#29324a;font-weight:700;cursor:pointer;transition:.2s}.chip:hover{transform:translateY(-2px);box-shadow:0 8px 20px #4f46e51a}.chat{height:480px;overflow-y:auto;border:1px dashed #dfe2ef;border-radius:22px;padding:24px;background:linear-gradient(#fff,#fbfaff);scroll-behavior:smooth}.empty{text-align:center;padding:120px 20px;color:#68728b}.bot{font-size:58px}.msg{max-width:78%;padding:14px 18px;border-radius:18px;margin:12px 0;line-height:1.5;white-space:pre-wrap}.user{margin-left:auto;background:linear-gradient(135deg,#694cff,#7c5cff);color:#fff;border-bottom-right-radius:5px}.assistant{background:#f5f6fb;border:1px solid #e8eaf2;color:#25304a;border-bottom-left-radius:5px}.composer{display:flex;gap:12px;margin-top:18px;border:2px solid #8a70ff;background:#fff;border-radius:20px;padding:10px 12px;box-shadow:0 10px 30px #6d4cff18}.composer textarea{flex:1;border:0;outline:0;resize:none;font:inherit;font-size:16px;padding:12px;background:transparent;min-height:70px}.send{width:58px;height:58px;border:0;border-radius:17px;background:linear-gradient(135deg,#6748ff,#805cff);color:white;font-size:24px;cursor:pointer;align-self:flex-end}.hint{text-align:center;color:#8991a6;font-size:13px;margin-top:10px}@media(max-width:900px){.app{padding:12px}.side{display:none}.main{padding:20px;border-radius:20px}.chips{grid-template-columns:repeat(2,1fr)}.title{font-size:27px}.chat{height:55vh}}@media(max-width:520px){.chips{grid-template-columns:1fr}.chip{padding:13px}.msg{max-width:90%}}
</style></head>
<body><div class="app"><aside class="side"><div class="brand">📅 Schedule<span>Assistant</span></div><div class="nav active">💬 Chat Assistant</div><div class="nav">🗓️ My Schedule</div><div class="nav">➕ Add Event</div><div class="nav">📋 All Events</div><div class="nav">⚙️ Settings</div><div class="ai">✨ <b>AI Powered</b><br><small>Your personal schedule assistant</small></div></aside>
<main class="main"><div class="top"><div><div class="title">Agentic RAG Schedule Assistant</div><div class="sub">Ask anything about your schedule or manage events naturally.</div></div><button class="theme" onclick="document.body.classList.toggle('dark')">☀️</button></div>
<div class="chips"><button class="chip" onclick="quick('What do I have scheduled tomorrow?')">🕐 What do I have tomorrow?</button><button class="chip" onclick="quick('What is my schedule this week?')">🗓️ My schedule this week</button><button class="chip" onclick="quick('Add a meeting on August 20 at 3 PM')">➕ Add a meeting</button><button class="chip" onclick="quick('Remove my IEEE Seminar')">🗑️ Remove an event</button></div>
<div id="chat" class="chat"><div id="empty" class="empty"><div class="bot">🤖</div><h2>Hello! 👋</h2><p>How can I help you with your schedule today?</p></div></div>
<div class="composer"><textarea id="m" placeholder="Type your request here..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();ask()}"></textarea><button class="send" onclick="ask()">➤</button></div><div class="hint">Press Enter to send • Shift + Enter for a new line</div></main></div>
<script>
function quick(t){document.getElementById('m').value=t;ask()}
function add(text,kind){let c=document.getElementById('chat');let e=document.getElementById('empty');if(e)e.remove();let d=document.createElement('div');d.className='msg '+kind;d.textContent=text;c.appendChild(d);c.scrollTop=c.scrollHeight;return d}
async function ask(){let box=document.getElementById('m'),m=box.value.trim();if(!m)return;add(m,'user');box.value='';let wait=add('Thinking…','assistant');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});let d=await r.json();wait.textContent=d.answer||d.detail||'I could not process that request.'}catch(e){wait.textContent='Connection problem. Please try again.'}document.getElementById('chat').scrollTop=document.getElementById('chat').scrollHeight}
</script></body></html>'''


@app.post('/chat')
def chat(r: Req):
    try:
        if not r.message.strip():
            raise HTTPException(400, 'Message cannot be empty.')
        return {'answer': agent(r.message)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f'Assistant error: {str(e)}')


@app.get('/health')
def health():
    return {'status':'ok','events':col.count(),'today':date.today().isoformat()}
