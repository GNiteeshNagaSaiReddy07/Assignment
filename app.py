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

API_KEY=os.getenv('GOOGLE_API_KEY')
if not API_KEY: raise RuntimeError('GOOGLE_API_KEY is not set.')
app=FastAPI(title='Agentic RAG Schedule Assistant',version='3.0')
client=chromadb.PersistentClient(path=os.getenv('CHROMA_DIR','./chroma_db'))
emb=GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001',google_api_key=API_KEY)
col=client.get_or_create_collection('schedule',metadata={'hnsw:space':'cosine'})
llm=ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL','gemini-2.5-flash'),temperature=0,google_api_key=API_KEY)

def samples():
 s=date.today(); return [
 {'id':'evt-001','title':'Project Team Meeting','event_type':'meeting','date':(s+timedelta(1)).isoformat(),'start_time':'10:00','end_time':'11:00','location':'ECE Seminar Hall','notes':'Discuss project progress and assign tasks.'},
 {'id':'evt-002','title':'Python AI Workshop','event_type':'workshop','date':(s+timedelta(2)).isoformat(),'start_time':'14:00','end_time':'16:00','location':'Computer Lab 2','notes':'Hands-on workshop on AI agents and RAG.'},
 {'id':'evt-003','title':'DBMS Assignment','event_type':'task','date':(s+timedelta(3)).isoformat(),'start_time':'18:00','end_time':'19:00','location':'Home','notes':'Complete normalization and SQL exercises.'},
 {'id':'evt-004','title':'Faculty Appointment','event_type':'appointment','date':(s+timedelta(4)).isoformat(),'start_time':'11:30','end_time':'12:00','location':'Faculty Room','notes':'Discuss project guidance.'},
 {'id':'evt-005','title':'IEEE Seminar','event_type':'seminar','date':(s+timedelta(6)).isoformat(),'start_time':'10:00','end_time':'12:00','location':'Auditorium','notes':'Seminar on electric vehicles and smart energy systems.'},
 {'id':'evt-006','title':'Machine Learning Lab','event_type':'lab','date':(s+timedelta(8)).isoformat(),'start_time':'09:00','end_time':'11:00','location':'ML Lab','notes':'Complete classification experiment.'},
 {'id':'evt-007','title':'Project Review','event_type':'meeting','date':(s+timedelta(10)).isoformat(),'start_time':'15:00','end_time':'16:30','location':'Project Lab','notes':'Internal project review and demonstration.'},
 {'id':'evt-008','title':'Sports Practice','event_type':'activity','date':(s+timedelta(12)).isoformat(),'start_time':'17:00','end_time':'18:30','location':'College Ground','notes':'Team practice.'},
 {'id':'evt-009','title':'Data Structures Test','event_type':'exam','date':(s+timedelta(15)).isoformat(),'start_time':'09:30','end_time':'11:00','location':'Classroom 204','notes':'Trees, graphs and sorting.'},
 {'id':'evt-010','title':'Career Guidance Workshop','event_type':'workshop','date':(s+timedelta(20)).isoformat(),'start_time':'13:00','end_time':'15:00','location':'Auditorium','notes':'Resume, interview and placement preparation.'}]

def text(e): return f"Title: {e['title']}. Type: {e['event_type']}. Date: {e['date']}. Time: {e['start_time']} to {e['end_time']}. Location: {e.get('location','')}. Notes: {e.get('notes','')}."
def all_events(): return col.get(include=['metadatas']).get('metadatas') or []
if col.count()==0:
 ev=samples(); docs=[text(e) for e in ev]; col.add(ids=[e['id'] for e in ev],documents=docs,embeddings=emb.embed_documents(docs),metadatas=ev)

def format_events(events):
 if not events:return 'No matching schedule entries were found.'
 return '\n'.join(f"{e['id']} | {e['date']} | {e['start_time']}-{e['end_time']} | {e['title']} | {e['event_type']} | {e.get('location','')} | {e.get('notes','')}" for e in events)

@tool
def get_schedule(query:str)->str:
 q=query.lower().strip(); events=all_events(); today=date.today(); selected=[]
 if 'tomorrow' in q:
  target=today+timedelta(1); selected=[e for e in events if e.get('date')==target.isoformat()]
 elif 'today' in q:selected=[e for e in events if e.get('date')==today.isoformat()]
 elif 'this week' in q or 'week schedule' in q:
  start=today-timedelta(today.weekday()); end=start+timedelta(6); selected=[e for e in events if start.isoformat()<=e.get('date','')<=end.isoformat()]
 elif 'next week' in q:
  start=today-timedelta(today.weekday())+timedelta(7); end=start+timedelta(6); selected=[e for e in events if start.isoformat()<=e.get('date','')<=end.isoformat()]
 iso=re.search(r'\b20\d{2}-\d{2}-\d{2}\b',q)
 if iso:selected=[e for e in events if e.get('date')==iso.group(0)]
 if selected:
  selected.sort(key=lambda e:(e.get('date',''),e.get('start_time',''))); return format_events(selected)
 try:
  r=col.query(query_embeddings=[emb.embed_query(query)],n_results=min(8,max(1,col.count())),include=['metadatas']); return format_events(r.get('metadatas',[[]])[0])
 except Exception:return format_events(events[:8])

@tool
def update_schedule(action:str,event_id:Optional[str]=None,title:Optional[str]=None,event_type:Optional[str]=None,date_value:Optional[str]=None,start_time:Optional[str]=None,end_time:Optional[str]=None,location:Optional[str]=None,notes:Optional[str]=None,search_query:Optional[str]=None)->str:
 action=action.lower().strip()
 if action not in ('add','update','remove'):return 'Invalid action. Use add, update, or remove.'
 if action=='add':
  if not title or not date_value or not start_time:return 'title, date_value and start_time are required.'
  eid=event_id or 'evt-'+uuid.uuid4().hex[:8]; e={'id':eid,'title':title,'event_type':event_type or 'meeting','date':date_value,'start_time':start_time,'end_time':end_time or start_time,'location':location or '','notes':notes or ''}; t=text(e); col.upsert(ids=[eid],documents=[t],embeddings=[emb.embed_documents([t])[0]],metadatas=[e]); return 'Added: '+t
 if not event_id and search_query:
  q=search_query.lower(); exact=[e for e in all_events() if q in e.get('title','').lower()]
  if exact:event_id=exact[0]['id']
  else:
   r=col.query(query_embeddings=[emb.embed_query(search_query)],n_results=1,include=['metadatas']); ms=r.get('metadatas',[[]])[0]
   if ms:event_id=ms[0]['id']
 if not event_id:return 'Could not identify the event.'
 ms=col.get(ids=[event_id],include=['metadatas']).get('metadatas') or []
 if not ms:return 'Event not found.'
 if action=='remove':col.delete(ids=[event_id]); return f"Removed: {ms[0]['title']} on {ms[0]['date']}"
 e=dict(ms[0]); e.update({k:v for k,v in {'title':title,'event_type':event_type,'date':date_value,'start_time':start_time,'end_time':end_time,'location':location,'notes':notes}.items() if v is not None}); t=text(e); col.upsert(ids=[event_id],documents=[t],embeddings=[emb.embed_documents([t])[0]],metadatas=[e]); return 'Updated: '+t

tools=[get_schedule,update_schedule]; tmap={t.name:t for t in tools}; model=llm.bind_tools(tools)
SYSTEM='''You are an Agentic RAG Schedule Assistant. Use get_schedule for existing schedule/date/time/availability questions. Use update_schedule for add, move, edit, or remove requests. Never invent existing events. Convert dates to YYYY-MM-DD and times to HH:MM. Current date is {today}. Keep answers concise.'''

def agent(msg):
 messages=[('system',SYSTEM.format(today=date.today().isoformat())),('human',msg)]
 try:
  for _ in range(6):
   res=model.invoke(messages); messages.append(res); calls=getattr(res,'tool_calls',[]) or []
   if not calls:return res.content if isinstance(res.content,str) else str(res.content)
   for c in calls:messages.append(ToolMessage(content=str(tmap[c['name']].invoke(c.get('args',{}))),tool_call_id=c['id']))
  return 'I completed the request but reached the tool-call limit.'
 except Exception:return get_schedule.invoke(msg)

class Req(BaseModel): message:str

@app.get('/',response_class=HTMLResponse)
def home():
 return '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule AI</title><style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,sans-serif;background:linear-gradient(135deg,#eef2ff,#faf9ff);color:#18213d}.app{min-height:100vh;display:flex;max-width:1500px;margin:auto;padding:22px;gap:18px}.side{width:235px;background:#fff;border-radius:24px;padding:22px 14px;box-shadow:0 12px 40px #5046e51c;display:flex;flex-direction:column}.brand{font-size:23px;font-weight:800;padding:8px 12px 24px}.brand span{display:block;color:#684cff}.nav{padding:13px;border-radius:13px;margin:4px 0;color:#5b657d;font-weight:650;cursor:pointer}.nav.active,.nav:hover{background:#f1edff;color:#684cff}.ai{margin-top:auto;padding:16px;border-radius:17px;background:#f7f3ff;color:#606a82}.ai b{color:#684cff}.main{flex:1;background:#fff;border-radius:27px;padding:28px;box-shadow:0 15px 50px #5046e51a;min-width:0}.top{display:flex;justify-content:space-between;align-items:center}.title{font-size:32px;font-weight:850}.sub{color:#788198;margin-top:5px}.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:25px 0 16px}.chip{border:0;border-radius:15px;padding:15px;text-align:left;background:#f4f5ff;font-weight:700;color:#303951;cursor:pointer}.chip:hover{box-shadow:0 8px 20px #5046e51c;transform:translateY(-1px)}.chat{height:455px;overflow:auto;border:1px dashed #dfe2ed;border-radius:20px;padding:20px;background:#fcfbff}.msg{max-width:78%;padding:13px 17px;border-radius:17px;margin:11px 0;line-height:1.5;white-space:pre-wrap}.user{margin-left:auto;background:linear-gradient(135deg,#674cff,#7858ff);color:white}.bot{background:#fff;border:1px solid #e2e4ee;color:#26314b}.empty{text-align:center;padding:120px 20px;color:#68738a}.robot{font-size:55px}.inputrow{display:flex;gap:10px;margin-top:17px;border:2px solid #7558ff;border-radius:20px;padding:8px 10px;background:#fff}.inputrow textarea{border:0;outline:0;resize:none;flex:1;height:64px;padding:12px;font:16px inherit}.send{width:55px;border:0;border-radius:15px;background:#6d4cff;color:#fff;font-size:23px;cursor:pointer}.hint{text-align:center;color:#8790a5;font-size:13px;margin-top:10px}.modal{display:none;position:fixed;inset:0;background:#17203d77;backdrop-filter:blur(5px);align-items:center;justify-content:center;padding:20px;z-index:20}.modal.show{display:flex}.form{background:#fff;width:min(560px,100%);border-radius:24px;padding:26px;box-shadow:0 25px 80px #1113}.form h2{margin:0 0 5px}.form p{color:#778197;margin-top:5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:13px}.field{margin-top:13px}.field label{display:block;font-size:13px;font-weight:700;margin-bottom:6px;color:#4e5870}.field input,.field select,.field textarea{width:100%;padding:12px;border:1px solid #dfe2eb;border-radius:11px;font:inherit;outline:0}.field textarea{height:75px;resize:vertical}.actions{display:flex;justify-content:flex-end;gap:10px;margin-top:20px}.cancel,.save{border:0;border-radius:11px;padding:12px 19px;font-weight:700;cursor:pointer}.cancel{background:#eef0f5;color:#4e5870}.save{background:#684cff;color:white}@media(max-width:850px){.side{display:none}.app{padding:10px}.main{padding:18px}.chips{grid-template-columns:1fr 1fr}.title{font-size:25px}}@media(max-width:520px){.chips{grid-template-columns:1fr}.grid{grid-template-columns:1fr}.chat{height:420px}}
</style></head><body><div class="app"><aside class="side"><div class="brand">📅 Schedule <span>Assistant</span></div><div class="nav active">💬 Chat Assistant</div><div class="nav" onclick="showSchedule()">🗓️ My Schedule</div><div class="nav" onclick="openModal()">➕ Add Event</div><div class="nav" onclick="showSchedule()">📋 All Events</div><div class="nav">⚙️ Settings</div><div class="ai">✨ <b>AI Powered</b><br><small>Your personal schedule assistant</small></div></aside><main class="main"><div class="top"><div><div class="title">Agentic RAG Schedule Assistant</div><div class="sub">Plan, add, update and manage your schedule naturally.</div></div></div><div class="chips"><button class="chip" onclick="ask('What do I have scheduled tomorrow?')">🕐 What do I have tomorrow?</button><button class="chip" onclick="ask('What is my schedule this week?')">🗓️ This week's schedule</button><button class="chip" onclick="openModal()">➕ Add a new event</button><button class="chip" onclick="ask('Find a free time tomorrow afternoon')">🔎 Find free time</button></div><div id="chat" class="chat"><div id="empty" class="empty"><div class="robot">🤖</div><h2>Hello! 👋</h2><p>Tell me what you want to schedule, or use <b>Add Event</b> to enter the details yourself.</p></div></div><div class="inputrow"><textarea id="msg" placeholder="Ask about your schedule or say: Plan a movie tomorrow at 7 PM..."></textarea><button class="send" onclick="send()">➤</button></div><div class="hint">Natural language works for questions, plans, appointments, tasks and changes.</div></main></div>
<div id="modal" class="modal"><div class="form"><h2>➕ Add to your schedule</h2><p>Please enter the details before anything is added.</p><div class="field"><label>Purpose / Event name *</label><input id="purpose" placeholder="e.g. Doctor appointment, Movie, Study session"></div><div class="grid"><div class="field"><label>Date *</label><input id="eventDate" type="date"></div><div class="field"><label>Event type</label><select id="eventType"><option>meeting</option><option>appointment</option><option>task</option><option>activity</option><option>workshop</option><option>personal</option></select></div><div class="field"><label>Start time *</label><input id="start" type="time"></div><div class="field"><label>End time</label><input id="end" type="time"></div></div><div class="field"><label>Location</label><input id="location" placeholder="e.g. Hospital, Cinema, Home"></div><div class="field"><label>Notes</label><textarea id="notes" placeholder="Any extra details..."></textarea></div><div class="actions"><button class="cancel" onclick="closeModal()">Cancel</button><button class="save" onclick="saveEvent()">Add to Schedule</button></div></div></div>
<script>
const chat=document.getElementById('chat'),modal=document.getElementById('modal');
function openModal(){modal.classList.add('show');document.getElementById('eventDate').value=new Date(Date.now()+86400000).toISOString().slice(0,10);document.getElementById('purpose').focus()}
function closeModal(){modal.classList.remove('show')}
modal.addEventListener('click',e=>{if(e.target===modal)closeModal()});
function addMsg(text,who){document.getElementById('empty')?.remove();let d=document.createElement('div');d.className='msg '+who;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
async function ask(text){addMsg(text,'user');let last=addMsg('Thinking…','bot');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});let d=await r.json();last.textContent=d.answer||d.detail||'No response.'}catch(e){last.textContent='Unable to reach the assistant. Please try again.'}}
function send(){let x=document.getElementById('msg'),v=x.value.trim();if(!v)return;x.value='';ask(v)}
document.getElementById('msg').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
async function saveEvent(){let purpose=document.getElementById('purpose').value.trim(),d=document.getElementById('eventDate').value,s=document.getElementById('start').value,e=document.getElementById('end').value,type=document.getElementById('eventType').value,loc=document.getElementById('location').value.trim(),notes=document.getElementById('notes').value.trim();if(!purpose||!d||!s){alert('Please enter the purpose, date and start time.');return}let prompt=`Add a ${type} called ${purpose} on ${d} at ${s}${e&&e!==s?' until '+e:''}${loc?' at '+loc:''}${notes?' . Notes: '+notes:''}`;closeModal();ask(prompt)}
function showSchedule(){ask('Show me my schedule for the next 30 days.')}
</script></body></html>'''

@app.post('/chat')
def chat(r:Req):
 try:return {'answer':agent(r.message)}
 except Exception as e:raise HTTPException(status_code=500,detail=str(e))

@app.get('/health')
def health():return {'status':'ok','events':col.count(),'today':date.today().isoformat()}
