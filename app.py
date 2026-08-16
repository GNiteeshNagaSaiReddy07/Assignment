import os, uuid
from datetime import date, timedelta
from typing import Optional
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

API_KEY=os.getenv('GOOGLE_API_KEY')
if not API_KEY:
    raise RuntimeError('GOOGLE_API_KEY is not set.')

app=FastAPI(title='Agentic RAG Schedule Assistant',version='1.0')
client=chromadb.PersistentClient(path=os.getenv('CHROMA_DIR','./chroma_db'))
emb=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',google_api_key=API_KEY)
col=client.get_or_create_collection('schedule',metadata={'hnsw:space':'cosine'})
llm=ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL','gemini-2.5-flash'),temperature=0,google_api_key=API_KEY)

def samples():
    s=date.today()
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
        {'id':'evt-010','title':'Career Guidance Workshop','event_type':'workshop','date':(s+timedelta(20)).isoformat(),'start_time':'13:00','end_time':'15:00','location':'Auditorium','notes':'Resume, interview and placement preparation.'}
    ]

def text(e):
    return f"Title: {e['title']}. Type: {e['event_type']}. Date: {e['date']}. Time: {e['start_time']} to {e['end_time']}. Location: {e.get('location','')}. Notes: {e.get('notes','')}."

if col.count()==0:
    ev=samples()
    docs=[text(e) for e in ev]
    vec=emb.embed_documents(docs)
    col.add(ids=[e['id'] for e in ev],documents=docs,embeddings=vec,metadatas=ev)

@tool
def get_schedule(query:str)->str:
    """Retrieve relevant existing schedule information by semantic RAG."""
    r=col.query(query_embeddings=[emb.embed_query(query)],n_results=8,include=['metadatas'])
    ms=r.get('metadatas',[[]])[0]
    if not ms:
        return 'No matching schedule entries were found.'
    return '\n'.join(f"{m['id']} | {m['date']} | {m['start_time']}-{m['end_time']} | {m['title']} | {m['event_type']} | {m.get('location','')} | {m.get('notes','')}" for m in ms)

@tool
def update_schedule(action:str,event_id:Optional[str]=None,title:Optional[str]=None,event_type:Optional[str]=None,date_value:Optional[str]=None,start_time:Optional[str]=None,end_time:Optional[str]=None,location:Optional[str]=None,notes:Optional[str]=None,search_query:Optional[str]=None)->str:
    """Add, update, or remove a schedule entry. Use YYYY-MM-DD and HH:MM."""
    action=action.lower()
    if action not in ('add','update','remove'):
        return 'Invalid action. Use add, update, or remove.'
    if action=='add':
        if not title or not date_value or not start_time:
            return 'title, date_value and start_time are required.'
        eid=event_id or 'evt-'+uuid.uuid4().hex[:8]
        e={'id':eid,'title':title,'event_type':event_type or 'meeting','date':date_value,'start_time':start_time,'end_time':end_time or start_time,'location':location or '','notes':notes or ''}
        t=text(e)
        col.upsert(ids=[eid],documents=[t],embeddings=[emb.embed_documents([t])[0]],metadatas=[e])
        return 'Added: '+t
    if not event_id and search_query:
        r=col.query(query_embeddings=[emb.embed_query(search_query)],n_results=1,include=['metadatas'])
        ms=r.get('metadatas',[[]])[0]
        if ms:
            event_id=ms[0]['id']
    if not event_id:
        return 'Could not identify the event.'
    r=col.get(ids=[event_id],include=['metadatas'])
    ms=r.get('metadatas') or []
    if not ms:
        return 'Event not found.'
    if action=='remove':
        col.delete(ids=[event_id])
        return f"Removed: {ms[0]['title']} on {ms[0]['date']}"
    e=dict(ms[0])
    e.update({k:v for k,v in {'title':title,'event_type':event_type,'date':date_value,'start_time':start_time,'end_time':end_time,'location':location,'notes':notes}.items() if v is not None})
    t=text(e)
    col.upsert(ids=[event_id],documents=[t],embeddings=[emb.embed_documents([t])[0]],metadatas=[e])
    return 'Updated: '+t

tools=[get_schedule,update_schedule]
tmap={t.name:t for t in tools}
model=llm.bind_tools(tools)
SYSTEM='''You are an Agentic RAG Schedule Assistant. Use get_schedule for existing schedule/date/time/availability questions. Use update_schedule for add, move, edit, or remove requests. For an ambiguous move/remove, retrieve first. Never invent existing events. For availability, retrieve relevant events and reason about overlap. Convert dates to YYYY-MM-DD and times to HH:MM. Current date is {today}. Keep answers concise.'''

def agent(msg):
    messages=[('system',SYSTEM.format(today=date.today().isoformat())),('human',msg)]
    for _ in range(6):
        res=model.invoke(messages)
        messages.append(res)
        calls=getattr(res,'tool_calls',[]) or []
        if not calls:
            return res.content if isinstance(res.content,str) else str(res.content)
        for c in calls:
            result=tmap[c['name']].invoke(c.get('args',{}))
            messages.append(ToolMessage(content=str(result),tool_call_id=c['id']))
    return 'Tool-call limit reached.'

class Req(BaseModel):
    message:str

@app.get('/',response_class=HTMLResponse)
def home():
    return '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic RAG Schedule Assistant</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:linear-gradient(135deg,#eef2ff,#f8f9ff 45%,#f4efff);color:#18213a;min-height:100vh}.app{display:flex;min-height:100vh;padding:22px;gap:20px}.sidebar{width:250px;background:rgba(255,255,255,.88);backdrop-filter:blur(14px);border:1px solid #fff;border-radius:24px;padding:26px 18px;box-shadow:0 18px 50px rgba(65,55,140,.10);display:flex;flex-direction:column}.brand{padding:4px 12px 24px}.brand-icon{width:58px;height:58px;border-radius:18px;background:linear-gradient(135deg,#635bff,#8b5cf6);display:grid;place-items:center;color:#fff;font-size:30px;box-shadow:0 10px 25px #635bff33}.brand h2{margin:15px 0 2px;font-size:22px}.brand h2 span{color:#6d5dfc}.brand p{margin:0;color:#7b849b;font-size:13px}.nav{display:grid;gap:8px}.nav button{border:0;background:transparent;text-align:left;padding:13px 14px;border-radius:13px;color:#42506b;font-size:15px;cursor:pointer}.nav button:hover,.nav button.active{background:#f0edff;color:#5b4df4;font-weight:600}.nav button span{display:inline-block;width:27px}.ai-card{margin-top:auto;background:linear-gradient(145deg,#f7f4ff,#fff);border:1px solid #e9e4ff;border-radius:18px;padding:16px}.ai-card .spark{font-size:22px}.ai-card strong{display:block;margin:8px 0 5px}.ai-card small{color:#7c8498;line-height:1.5}.main{flex:1;min-width:0;background:rgba(255,255,255,.9);border:1px solid #fff;border-radius:24px;box-shadow:0 18px 55px rgba(65,55,140,.11);padding:30px 34px;display:flex;flex-direction:column}.top{display:flex;justify-content:space-between;align-items:center;gap:15px}.title-wrap{display:flex;align-items:center;gap:16px}.title-icon{width:56px;height:56px;border-radius:17px;background:#f0edff;display:grid;place-items:center;font-size:30px}.title h1{margin:0;font-size:30px;letter-spacing:-.6px}.title p{margin:6px 0 0;color:#758099}.theme{width:44px;height:44px;border:1px solid #e7e9f1;background:#fff;border-radius:50%;cursor:pointer;font-size:19px}.quick{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 22px}.quick button{border:1px solid transparent;border-radius:16px;padding:15px;text-align:left;background:#f5f7ff;color:#24304a;cursor:pointer;font-size:14px;min-height:78px}.quick button:hover{transform:translateY(-2px);box-shadow:0 8px 20px #6d5dfc15}.quick .q1{background:#eff5ff}.quick .q2{background:#ecfbf7}.quick .q3{background:#fff7e9}.quick .q4{background:#fff0f2}.quick b{display:block;margin-top:8px;font-size:14px}.chat{flex:1;min-height:330px;border:1px dashed #dce0ed;border-radius:20px;background:linear-gradient(180deg,#fcfcff,#f9f9ff);padding:28px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;overflow:auto}.welcome-icon{width:86px;height:86px;border-radius:28px;background:linear-gradient(145deg,#ede9ff,#e7f0ff);display:grid;place-items:center;font-size:44px;box-shadow:0 12px 30px #6d5dfc18}.chat h2{margin:18px 0 7px;font-size:26px}.chat p{margin:0;color:#707b93}.messages{width:100%;display:none;gap:12px;flex-direction:column;text-align:left}.msg{max-width:82%;padding:13px 16px;border-radius:16px;line-height:1.5;white-space:pre-wrap}.user{align-self:flex-end;background:#6857f6;color:white;border-bottom-right-radius:5px}.bot{align-self:flex-start;background:white;border:1px solid #e5e7ef;color:#27314b;border-bottom-left-radius:5px}.composer{margin-top:20px}.input-wrap{display:flex;align-items:flex-end;gap:10px;border:2px solid #7b63ff;border-radius:18px;background:#fff;padding:10px 10px 10px 16px;box-shadow:0 8px 28px #715cf31c}.input-wrap textarea{flex:1;border:0;outline:0;resize:none;height:58px;padding:10px 4px;font:15px inherit;color:#24304a}.input-wrap textarea::placeholder{color:#9aa2b5}.send{width:48px;height:48px;border:0;border-radius:15px;background:linear-gradient(135deg,#6756f5,#845cf6);color:#fff;font-size:21px;cursor:pointer;box-shadow:0 8px 18px #6d5dfc44}.send:disabled{opacity:.6;cursor:wait}.hint{text-align:center;color:#8a91a4;font-size:12px;margin-top:9px}.status{font-size:12px;color:#7c8497;margin-top:8px;text-align:center}.typing{display:inline-flex;gap:4px}.typing i{width:6px;height:6px;border-radius:50%;background:#7b70ef;animation:b 1s infinite}.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}@keyframes b{50%{opacity:.3;transform:translateY(-2px)}}
@media(max-width:900px){.app{padding:12px}.sidebar{width:190px}.quick{grid-template-columns:repeat(2,1fr)}.title h1{font-size:24px}}@media(max-width:680px){.app{display:block;padding:8px}.sidebar{width:100%;margin-bottom:10px;padding:14px}.brand{display:flex;align-items:center;gap:12px;padding:4px 8px 14px}.brand-icon{width:45px;height:45px;font-size:22px}.brand h2{margin:0;font-size:18px}.brand p{display:none}.nav{grid-template-columns:repeat(4,1fr)}.nav button{padding:10px 5px;text-align:center;font-size:11px}.nav button span{display:block;width:auto;font-size:18px;margin-bottom:3px}.ai-card{display:none}.main{padding:20px 15px;border-radius:20px}.title-icon{width:45px;height:45px;font-size:24px}.title h1{font-size:20px}.title p{font-size:12px}.theme{width:38px;height:38px}.quick{grid-template-columns:1fr 1fr;gap:8px;margin:18px 0 12px}.quick button{min-height:70px;padding:11px;font-size:12px}.chat{min-height:300px;padding:20px}.msg{max-width:92%}}
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
  <div class="brand"><div class="brand-icon">📅</div><h2>Schedule <span>Assistant</span></h2><p>Smart planning with AI</p></div>
  <div class="nav">
    <button class="active"><span>💬</span>Chat Assistant</button>
    <button onclick="quick('What do I have scheduled this week?')"><span>🗓️</span>My Schedule</button>
    <button onclick="quick('Add a new meeting tomorrow at 10 AM')"><span>➕</span>Add Event</button>
    <button onclick="quick('Show all my upcoming events')"><span>📋</span>All Events</button>
  </div>
  <div class="ai-card"><div class="spark">✨</div><strong>AI Powered</strong><small>Your personal schedule assistant understands natural language and manages events for you.</small></div>
</aside>
<main class="main">
  <header class="top">
    <div class="title-wrap"><div class="title-icon">🤖</div><div class="title"><h1>Agentic RAG Schedule Assistant</h1><p>Ask about your schedule or manage events naturally.</p></div></div>
    <button class="theme" onclick="toggleTheme()" title="Toggle theme">☼</button>
  </header>
  <section class="quick">
    <button class="q1" onclick="quick('What do I have scheduled tomorrow?')">🕐<b>What's on tomorrow?</b></button>
    <button class="q2" onclick="quick('What is my schedule this week?')">🗓️<b>My schedule this week</b></button>
    <button class="q3" onclick="quick('Add a meeting on August 20 at 3 PM')">➕<b>Add an event</b></button>
    <button class="q4" onclick="quick('Remove my IEEE Seminar')">🗑️<b>Remove an event</b></button>
  </section>
  <section class="chat" id="chat">
    <div id="welcome"><div class="welcome-icon">🤖</div><h2>Hello! 👋</h2><p>How can I help you with your schedule today?</p></div>
    <div class="messages" id="messages"></div>
  </section>
  <div class="composer">
    <div class="input-wrap"><textarea id="m" rows="2" placeholder="Type your request here..." onkeydown="key(event)"></textarea><button class="send" id="send" onclick="ask()">➤</button></div>
    <div class="hint">You can ask, add, update, move or remove events using natural language.</div>
    <div class="status" id="status"></div>
  </div>
</main>
</div>
<script>
const input=document.getElementById('m'),messages=document.getElementById('messages'),welcome=document.getElementById('welcome'),send=document.getElementById('send'),statusEl=document.getElementById('status');
function key(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask()}}
function quick(t){input.value=t;input.focus();ask()}
function addMessage(text,who){welcome.style.display='none';messages.style.display='flex';const d=document.createElement('div');d.className='msg '+who;d.textContent=text;messages.appendChild(d);messages.scrollTop=messages.scrollHeight}
async function ask(){const m=input.value.trim();if(!m)return;addMessage(m,'user');input.value='';send.disabled=true;statusEl.innerHTML='<span class="typing"><i></i><i></i><i></i></span> AI is thinking...';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Request failed');addMessage(d.answer||'No response received.','bot')}catch(e){addMessage('Sorry, I could not process that request. Please try again.','bot');console.error(e)}finally{send.disabled=false;statusEl.textContent='';input.focus()}}
function toggleTheme(){document.body.classList.toggle('dark');if(document.body.classList.contains('dark')){document.body.style.background='#101426';document.querySelector('.main').style.background='#171c31';document.querySelector('.sidebar').style.background='#171c31';document.body.style.color='#edf0ff'}else{document.body.style.background='linear-gradient(135deg,#eef2ff,#f8f9ff 45%,#f4efff)';document.querySelector('.main').style.background='rgba(255,255,255,.9)';document.querySelector('.sidebar').style.background='rgba(255,255,255,.88)';document.body.style.color='#18213a'}}
</script>
</body>
</html>'''

@app.post('/chat')
def chat(r:Req):
    try:
        return {'answer':agent(r.message)}
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get('/health')
def health():
    return {'status':'ok','events':col.count(),'today':date.today().isoformat()}
