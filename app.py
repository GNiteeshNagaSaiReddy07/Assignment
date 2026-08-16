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

API_KEY=os.getenv('GOOGLE_API_KEY','').strip()
MODEL=os.getenv('GEMINI_MODEL','gemini-2.5-flash')
TZ=ZoneInfo('Asia/Kolkata')
CHROMA_DIR=os.getenv('CHROMA_DIR','./chroma_db')
app=FastAPI(title='Agentic RAG Schedule Assistant',version='5.0')

def today(): return datetime.now(TZ).date()
def emb(text,size=96):
    v=[0.0]*size
    for w in re.findall(r'[a-z0-9]+',text.lower()) or ['x']:
        d=hashlib.sha256(w.encode()).digest()
        for i in range(0,len(d),2): v[int.from_bytes(d[i:i+2],'big')%size]+=1
    n=sum(x*x for x in v)**.5 or 1
    return [x/n for x in v]

client=chromadb.PersistentClient(path=CHROMA_DIR)
col=client.get_or_create_collection('schedule',metadata={'hnsw:space':'cosine'})

def seed():
    s=today()
    return [
      {'id':'evt-001','title':'Project Team Meeting','event_type':'meeting','date':(s+timedelta(1)).isoformat(),'start_time':'10:00','end_time':'11:00','location':'ECE Seminar Hall','notes':'Project progress and tasks.'},
      {'id':'evt-002','title':'Python AI Workshop','event_type':'workshop','date':(s+timedelta(2)).isoformat(),'start_time':'14:00','end_time':'16:00','location':'Computer Lab 2','notes':'Hands-on AI agents and RAG.'},
      {'id':'evt-003','title':'DBMS Assignment','event_type':'task','date':(s+timedelta(3)).isoformat(),'start_time':'18:00','end_time':'19:00','location':'Home','notes':'Normalization and SQL exercises.'},
      {'id':'evt-004','title':'Faculty Appointment','event_type':'appointment','date':(s+timedelta(4)).isoformat(),'start_time':'11:30','end_time':'12:00','location':'Faculty Room','notes':'Project guidance.'},
      {'id':'evt-005','title':'IEEE Seminar','event_type':'seminar','date':(s+timedelta(6)).isoformat(),'start_time':'10:00','end_time':'12:00','location':'Auditorium','notes':'EV and smart energy systems.'},
      {'id':'evt-006','title':'Machine Learning Lab','event_type':'lab','date':(s+timedelta(8)).isoformat(),'start_time':'09:00','end_time':'11:00','location':'ML Lab','notes':'Classification experiment.'},
      {'id':'evt-007','title':'Project Review','event_type':'meeting','date':(s+timedelta(10)).isoformat(),'start_time':'15:00','end_time':'16:30','location':'Project Lab','notes':'Project demonstration.'},
      {'id':'evt-008','title':'Sports Practice','event_type':'activity','date':(s+timedelta(12)).isoformat(),'start_time':'17:00','end_time':'18:30','location':'College Ground','notes':'Team practice.'},
      {'id':'evt-009','title':'Data Structures Test','event_type':'exam','date':(s+timedelta(15)).isoformat(),'start_time':'09:30','end_time':'11:00','location':'Classroom 204','notes':'Trees, graphs and sorting.'},
      {'id':'evt-010','title':'Career Guidance Workshop','event_type':'workshop','date':(s+timedelta(20)).isoformat(),'start_time':'13:00','end_time':'15:00','location':'Auditorium','notes':'Resume and placement preparation.'}]

def txt(e): return f"{e['title']} | {e['event_type']} | {e['date']} | {e['start_time']}-{e['end_time']} | {e.get('location','')} | {e.get('notes','')}"
def events(): return col.get(include=['metadatas']).get('metadatas') or []
if col.count()==0:
    x=seed(); col.add(ids=[e['id'] for e in x],documents=[txt(e) for e in x],embeddings=[emb(txt(e)) for e in x],metadatas=x)

def fmt(es):
    es=sorted(es,key=lambda e:(e.get('date',''),e.get('start_time','')))
    if not es:return 'No events are scheduled for that date.'
    return '\n'.join(f"{e['date']} | {e['start_time']}-{e['end_time']} | {e['title']} | {e['event_type']} | {e.get('location','')}" for e in es)

@tool
def get_schedule(query:str)->str:
    q=query.lower().strip(); all_e=events(); t=today()
    if any(x in q for x in ["today's date","todays date","what is today's date","what's today's date","current date"]):
        return f"Today is {t.strftime('%A, %B %d, %Y')}."
    if 'tomorrow' in q: return fmt([e for e in all_e if e['date']==(t+timedelta(1)).isoformat()])
    if re.search(r'\btoday\b',q): return fmt([e for e in all_e if e['date']==t.isoformat()])
    if 'this week' in q:
        a=t-timedelta(t.weekday()); b=a+timedelta(6); return fmt([e for e in all_e if a.isoformat()<=e['date']<=b.isoformat()])
    if 'next week' in q:
        a=t-timedelta(t.weekday())+timedelta(7); b=a+timedelta(6); return fmt([e for e in all_e if a.isoformat()<=e['date']<=b.isoformat()])
    m=re.search(r'20\d{2}-\d{2}-\d{2}',q)
    if m:return fmt([e for e in all_e if e['date']==m.group(0)])
    try:
        r=col.query(query_embeddings=[emb(query)],n_results=min(8,max(1,col.count())),include=['metadatas']); return fmt(r.get('metadatas',[[]])[0])
    except Exception:return fmt(all_e[:8])

@tool
def update_schedule(action:str,event_id:Optional[str]=None,title:Optional[str]=None,event_type:Optional[str]=None,date_value:Optional[str]=None,start_time:Optional[str]=None,end_time:Optional[str]=None,location:Optional[str]=None,notes:Optional[str]=None,search_query:Optional[str]=None)->str:
    action=action.lower().strip()
    if action=='add':
        if not title or not date_value or not start_time:return 'Please provide the event name, date and start time.'
        eid=event_id or 'evt-'+uuid.uuid4().hex[:8]; e={'id':eid,'title':title,'event_type':event_type or 'personal','date':date_value,'start_time':start_time,'end_time':end_time or start_time,'location':location or '','notes':notes or ''}; d=txt(e)
        col.upsert(ids=[eid],documents=[d],embeddings=[emb(d)],metadatas=[e]); return f"Added: {e['title']} on {e['date']} at {e['start_time']}."
    if not event_id and search_query:
        q=search_query.lower(); exact=[e for e in events() if q in e.get('title','').lower()]
        if exact:event_id=exact[0]['id']
        else:
            r=col.query(query_embeddings=[emb(search_query)],n_results=1,include=['metadatas']); f=r.get('metadatas',[[]])[0]; event_id=f[0]['id'] if f else None
    if not event_id:return 'I could not identify that event.'
    found=col.get(ids=[event_id],include=['metadatas']).get('metadatas') or []
    if not found:return 'Event not found.'
    if action=='remove':col.delete(ids=[event_id]);return f"Removed: {found[0]['title']} on {found[0]['date']}."
    e=dict(found[0]); e.update({k:v for k,v in {'title':title,'event_type':event_type,'date':date_value,'start_time':start_time,'end_time':end_time,'location':location,'notes':notes}.items() if v is not None}); d=txt(e); col.upsert(ids=[event_id],documents=[d],embeddings=[emb(d)],metadatas=[e]); return f"Updated: {e['title']} on {e['date']} at {e['start_time']}."

llm=None
def get_llm():
    global llm
    if llm is None:
        if not API_KEY:raise RuntimeError('GOOGLE_API_KEY is not configured in Render.')
        llm=ChatGoogleGenerativeAI(model=MODEL,temperature=0,google_api_key=API_KEY)
    return llm

def ptime(s):
    m=re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?\b',s.upper())
    if not m:return None
    h=int(m.group(1));mi=int(m.group(2) or 0);ap=m.group(3)
    if ap=='PM' and h<12:h+=12
    if ap=='AM' and h==12:h=0
    return f'{h:02d}:{mi:02d}' if h<24 and mi<60 else None

def pdate(s):
    q=s.lower();t=today()
    if 'tomorrow' in q:return t+timedelta(1)
    if 'today' in q:return t
    m=re.search(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b',q)
    if m:return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    for f in ('%B %d %Y','%b %d %Y','%B %d','%b %d'):
        try:
            d=datetime.strptime(s.strip(),f).date();return d.replace(year=t.year) if d.year==1900 else d
        except ValueError:pass

def fallback(msg):
    q=msg.strip();l=q.lower()
    if any(x in l for x in ["today's date","todays date","what is today's date","what's today's date","current date"]):return f"Today is {today().strftime('%A, %B %d, %Y')}."
    if any(x in l for x in ['remove ','delete ','cancel ']):
        m=re.search(r'(?:remove|delete|cancel)\s+(?:my\s+)?(.+)',q,re.I)
        if m:return update_schedule.invoke({'action':'remove','search_query':m.group(1).rstrip('.')})
    if any(x in l for x in ['add ','schedule ','plan ','book ']):
        d=pdate(q);tm=ptime(q)
        if d and tm:
            title=re.sub(r'^(add|schedule|plan|book)\s+','',q,flags=re.I);title=re.split(r'\s+(?:on|at|for)\s+',title,maxsplit=1,flags=re.I)[0].strip() or 'Scheduled Event'
            return update_schedule.invoke({'action':'add','title':title.title(),'event_type':'appointment' if 'doctor' in l or 'dentist' in l else 'personal','date_value':d.isoformat(),'start_time':tm,'end_time':tm})
    return get_schedule.invoke(q)

def agent(message):
    if not API_KEY:return fallback(message)+'\n\nGemini is not configured; local schedule mode was used.'
    try:
        tools=[get_schedule,update_schedule];tm={x.name:x for x in tools};model=get_llm().bind_tools(tools)
        sys='You are a schedule assistant. Use tools for schedule questions and changes. Never invent events. Understand tomorrow, today, weekdays, next week, morning, afternoon and evening. Current India date is '+today().isoformat()+'. For direct date questions answer directly.'
        msgs=[('system',sys),('human',message)]
        for _ in range(6):
            r=model.invoke(msgs);msgs.append(r);calls=getattr(r,'tool_calls',[]) or []
            if not calls:return r.content if isinstance(r.content,str) else str(r.content)
            for c in calls:msgs.append(ToolMessage(content=str(tm[c['name']].invoke(c.get('args',{}))),tool_call_id=c['id']))
    except Exception:return fallback(message)

class Request(BaseModel):message:str

HTML='''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule Assistant</title><style>
:root{--p:#6347f5;--p2:#7b55ff;--ink:#17204a;--muted:#737b98;--bg:#f7f7ff;--card:#fff;--line:#e8e8f2}*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:linear-gradient(135deg,#f0efff,#fbfbff);color:var(--ink)}button{font:inherit;cursor:pointer}.app{min-height:100vh;display:grid;grid-template-columns:240px 1fr;max-width:1600px;margin:auto}.side{background:linear-gradient(180deg,#17105b,#251879);color:white;padding:24px 14px;display:flex;flex-direction:column;min-height:100vh}.logo{font-size:22px;font-weight:800;padding:6px 14px 28px;display:flex;gap:10px;align-items:center}.logo i{font-style:normal;font-size:30px}.nav{padding:13px 15px;margin:3px 0;border-radius:13px;color:#ddd9ff;display:flex;gap:12px;align-items:center}.nav:hover,.nav.on{background:linear-gradient(90deg,#6347f5,#5b43ee);color:#fff}.profile{margin-top:auto;border-top:1px solid #ffffff24;padding:18px 8px}.avatar{display:flex;align-items:center;gap:10px}.avatar b{display:block}.avatar small{color:#c5c0e8}.theme{margin-top:15px;padding:11px;border:1px solid #ffffff22;border-radius:12px;color:#ddd9ff}.main{padding:28px;min-width:0}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.top h1{margin:0;font-size:27px}.datepill{background:#fff;border:1px solid var(--line);padding:12px 18px;border-radius:14px;font-weight:700}.hero{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:22px}.hero h2{font-size:28px;margin:0 0 5px}.hero p{margin:0;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}.stat,.panel{background:var(--card);border:1px solid var(--line);border-radius:20px;box-shadow:0 12px 35px #4e45a30b}.stat{padding:18px;display:flex;align-items:center;gap:14px}.icon{width:48px;height:48px;border-radius:15px;background:#f0edff;display:grid;place-items:center;font-size:23px}.stat strong{font-size:25px;display:block}.stat span{color:var(--muted);font-size:13px}.layout{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.9fr) 320px;gap:16px}.panel{padding:18px}.panelhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}.panelhead h3{margin:0;font-size:17px}.link{border:0;background:none;color:var(--p);font-weight:700}.event{display:grid;grid-template-columns:95px 1fr auto;gap:12px;padding:13px 10px;border-bottom:1px solid #f0f0f5;align-items:center}.event:last-child{border:0}.time{color:var(--p);font-weight:800;font-size:13px}.event b{font-size:14px}.event small{display:block;color:var(--muted);margin-top:4px}.tag{background:#f0edff;color:var(--p);padding:5px 8px;border-radius:8px;font-size:11px;font-weight:800}.calendar{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;text-align:center}.calendar div{padding:7px;font-size:12px}.dow{color:#8a90a8;font-weight:700}.day{border-radius:9px}.day:hover{background:#f0edff}.sel{background:var(--p);color:white!important;font-weight:800}.up{margin-top:18px}.upitem{padding:12px 4px;border-bottom:1px solid #f0f0f5}.upitem b{font-size:13px}.upitem small{display:block;color:var(--muted);margin-top:4px}.assistant{padding:0;overflow:hidden}.ahead{background:linear-gradient(135deg,#5d43ef,#7651ff);color:#fff;padding:19px}.ahead h3{margin:0 0 4px}.online{font-size:12px;color:#cffff0}.messages{height:370px;overflow:auto;padding:15px;background:#fcfcff}.bubble{max-width:90%;padding:11px 13px;border-radius:14px;margin:10px 0;font-size:13px;line-height:1.45;white-space:pre-wrap}.bot{background:#fff;border:1px solid var(--line)}.user{margin-left:auto;background:linear-gradient(135deg,#674cff,#7958ff);color:#fff}.quick{padding:12px;display:grid;gap:8px}.quick button{background:#fff;border:1px solid #cfc8ff;color:#5140c8;border-radius:11px;padding:9px;text-align:left;font-size:12px}.chatbox{display:flex;gap:7px;padding:10px;border-top:1px solid var(--line)}.chatbox textarea{flex:1;border:1px solid var(--line);border-radius:12px;padding:10px;resize:none;height:45px;outline:none}.send{width:46px;border:0;border-radius:12px;background:var(--p);color:#fff;font-size:20px}.actions{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:18px}.action{background:#fff;border:1px solid var(--line);border-radius:16px;padding:16px;font-weight:750;color:#31395b;text-align:left}.action:hover{border-color:#c9c0ff;transform:translateY(-1px)}.modal{display:none;position:fixed;inset:0;background:#10182d99;backdrop-filter:blur(6px);align-items:center;justify-content:center;padding:18px;z-index:5}.modal.show{display:flex}.form{background:#fff;border-radius:22px;padding:24px;width:min(560px,100%);box-shadow:0 30px 80px #0004}.form h2{margin:0}.field{margin-top:12px}.field label{display:block;font-size:12px;font-weight:800;margin-bottom:5px}.field input,.field select,.field textarea{width:100%;padding:11px;border:1px solid var(--line);border-radius:10px}.field textarea{height:70px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.formactions{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}.cancel,.save{border:0;padding:11px 17px;border-radius:10px;font-weight:800}.cancel{background:#eeeef5}.save{background:var(--p);color:#fff}@media(max-width:1200px){.layout{grid-template-columns:1.2fr .8fr}.assistant{grid-column:1/-1}.messages{height:230px}.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:800px){.app{grid-template-columns:1fr}.side{display:none}.main{padding:14px}.layout{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.actions{grid-template-columns:1fr 1fr}.hero{display:block}.datepill{display:inline-block;margin-top:12px}}@media(max-width:500px){.stats{grid-template-columns:1fr}.event{grid-template-columns:75px 1fr}.tag{display:none}.actions{grid-template-columns:1fr}.grid2{grid-template-columns:1fr}}
</style></head><body><div class="app"><aside class="side"><div class="logo"><i>📅</i> Schedule<br>Assistant</div><div class="nav on">🏠 Dashboard</div><div class="nav" onclick="showSchedule()">🗓️ My Schedule</div><div class="nav" onclick="openModal()">➕ Add Event</div><div class="nav" onclick="ask('Find a free time tomorrow')">✓ Check Availability</div><div class="nav" onclick="ask('What are my reminders?')">🔔 Reminders</div><div class="nav" onclick="showSchedule()">▥ Analytics</div><div class="nav">⚙️ Settings</div><div class="profile"><div class="avatar"><span style="font-size:31px">👤</span><div><b>Schedule User</b><small>Personal planner</small></div></div><div class="theme">◐ Dark mode</div></div></aside><main class="main"><div class="top"><h1>Dashboard</h1><div class="datepill">📅 <span id="todayLabel"></span></div></div><div class="hero"><div><h2>Good evening! 👋</h2><p>Here’s your schedule overview.</p></div></div><section class="stats"><div class="stat"><div class="icon">📅</div><div><strong id="total">0</strong><span>Total Events<br>Next 30 days</span></div></div><div class="stat"><div class="icon">☀️</div><div><strong id="todayCount">0</strong><span>Today<br>Scheduled</span></div></div><div class="stat"><div class="icon">⏰</div><div><strong id="weekCount">0</strong><span>Upcoming<br>Next 7 days</span></div></div><div class="stat"><div class="icon">✨</div><div><strong id="free">--</strong><span>Assistant<br>Online</span></div></div></section><section class="layout"><div class="panel"><div class="panelhead"><h3>📋 Today's Schedule</h3><button class="link" onclick="ask('What do I have today?')">Ask AI →</button></div><div id="todayList"></div></div><div class="panel"><div class="panelhead"><h3>🗓️ Calendar</h3><span id="month"></span></div><div class="calendar" id="cal"></div><div class="up"><div class="panelhead"><h3>Upcoming Events</h3><button class="link" onclick="showSchedule()">View All</button></div><div id="upcoming"></div></div></div><div class="panel assistant"><div class="ahead"><h3>✨ AI Schedule Assistant</h3><span class="online">● Online</span></div><div class="messages" id="messages"><div class="bubble bot">Hello! I can manage your schedule. Ask me about today, find a free time, or create a plan.</div></div><div class="quick"><button onclick="ask('What is today\'s date?')">📅 Today's date</button><button onclick="ask('What do I have today?')">🗓️ What do I have today?</button><button onclick="openModal()">➕ Add a new event</button><button onclick="ask('Find a free time tomorrow afternoon')">✓ Check my availability</button><button onclick="ask('What is my schedule this week?')">📆 What do I have this week?</button></div><div class="chatbox"><textarea id="msg" placeholder="Type your message..."></textarea><button class="send" onclick="send()">➤</button></div></div></section><section class="actions"><button class="action" onclick="openModal()">📅<br>Add Event</button><button class="action" onclick="ask('Find a free time tomorrow')">🟢<br>Check Availability</button><button class="action" onclick="ask('What is my schedule this week?')">🗓️<br>View This Week</button><button class="action" onclick="ask('What are my reminders?')">🔔<br>Reminders</button><button class="action" onclick="showSchedule()">📊<br>All Events</button></section></main></div><div class="modal" id="modal"><div class="form"><h2>➕ Add to your schedule</h2><p style="color:#737b98">Nothing is added until you confirm.</p><div class="field"><label>Purpose / Event name *</label><input id="purpose" placeholder="Doctor appointment, Movie, Study session..."></div><div class="grid2"><div class="field"><label>Date *</label><input id="eventDate" type="date"></div><div class="field"><label>Type</label><select id="eventType"><option>appointment</option><option>meeting</option><option>task</option><option>activity</option><option>personal</option><option>workshop</option></select></div><div class="field"><label>Start time *</label><input id="start" type="time"></div><div class="field"><label>End time</label><input id="end" type="time"></div></div><div class="field"><label>Location</label><input id="location" placeholder="Hospital, Cinema, Home..."></div><div class="field"><label>Notes</label><textarea id="notes" placeholder="Optional details"></textarea></div><div class="formactions"><button class="cancel" onclick="closeModal()">Cancel</button><button class="save" onclick="saveEvent()">Add to Schedule</button></div></div></div><script>
const M=document.getElementById('modal'),msgs=document.getElementById('messages');function openModal(){M.classList.add('show');document.getElementById('eventDate').value=new Date(Date.now()+86400000).toISOString().slice(0,10);document.getElementById('purpose').focus()}function closeModal(){M.classList.remove('show')}M.onclick=e=>{if(e.target===M)closeModal()};function bubble(t,c){const x=document.createElement('div');x.className='bubble '+c;x.textContent=t;msgs.appendChild(x);msgs.scrollTop=msgs.scrollHeight;return x}async function ask(t){bubble(t,'user');const b=bubble('Thinking…','bot');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})});const d=await r.json();b.textContent=d.answer||d.detail||'No response.';load()}catch(e){b.textContent='Unable to reach the assistant.'}}function send(){let x=document.getElementById('msg'),v=x.value.trim();if(v){x.value='';ask(v)}}document.getElementById('msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});function saveEvent(){const p=purpose.value.trim(),d=eventDate.value,s=start.value,e=end.value,t=eventType.value,l=location.value.trim(),n=notes.value.trim();if(!p||!d||!s){alert('Purpose, date and start time are required.');return}let q=`Schedule a ${t} called ${p} on ${d} at ${s}`+(e?` until ${e}`:'')+(l?` at location ${l}`:'')+(n?`. Notes: ${n}`:'');closeModal();ask(q)}function showSchedule(){ask('Show my schedule for the next 30 days.')}function fmtTime(s){let [h,m]=s.split(':').map(Number);let a=h>=12?'PM':'AM';h=h%12||12;return `${h}:${String(m).padStart(2,'0')} ${a}`}async function load(){try{let r=await fetch('/api/events'),d=await r.json();document.getElementById('total').textContent=d.total;document.getElementById('todayCount').textContent=d.today.length;document.getElementById('weekCount').textContent=d.next7.length;document.getElementById('free').textContent='Ready';let tl=document.getElementById('todayList');tl.innerHTML=d.today.length?d.today.map(e=>`<div class="event"><div class="time">${fmtTime(e.start_time)}<br>– ${fmtTime(e.end_time)}</div><div><b>${e.title}</b><small>⌖ ${e.location||'No location'}</small></div><span class="tag">${e.event_type}</span></div>`).join(''):'<div style="padding:35px;text-align:center;color:#858ca3">No events today 🎉</div>';document.getElementById('upcoming').innerHTML=d.upcoming.slice(0,3).map(e=>`<div class="upitem"><b>${e.title}</b><small>${e.date} · ${fmtTime(e.start_time)} · ${e.location||'No location'}</small></div>`).join('')||'<small>No upcoming events.</small>'}catch(e){}}function calendar(){let n=new Date(),y=n.getFullYear(),m=n.getMonth(),first=new Date(y,m,1).getDay(),days=new Date(y,m+1,0).getDate();document.getElementById('month').textContent=n.toLocaleString('en',{month:'long',year:'numeric'});let h=['Su','Mo','Tu','We','Th','Fr','Sa'].map(x=>`<div class="dow">${x}</div>`).join('');for(let i=0;i<first;i++)h+='<div></div>';for(let d=1;d<=days;d++)h+=`<div class="day ${d===n.getDate()?'sel':''}">${d}</div>`;document.getElementById('cal').innerHTML=h;document.getElementById('todayLabel').textContent=n.toLocaleDateString('en-IN',{weekday:'long',month:'long',day:'numeric',year:'numeric'})}calendar();load();</script></body></html>'''

@app.get('/',response_class=HTMLResponse)
def home():return HTMLResponse(HTML)
@app.post('/chat')
def chat(r:Request):
    try:return {'answer':agent(r.message.strip())}
    except Exception as e:raise HTTPException(500,str(e))
@app.get('/api/events')
def api_events():
    t=today();all_e=sorted(events(),key=lambda e:(e['date'],e['start_time']));return {'total':len(all_e),'today':[e for e in all_e if e['date']==t.isoformat()],'next7':[e for e in all_e if t<=date.fromisoformat(e['date'])<=t+timedelta(7)],'upcoming':[e for e in all_e if date.fromisoformat(e['date'])>=t]}
@app.get('/health')
def health():return {'status':'ok','events':col.count(),'today':today().isoformat(),'timezone':'Asia/Kolkata','gemini_configured':bool(API_KEY)}
