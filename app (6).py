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
    return '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Schedule Assistant</title><style>body{font-family:Arial;max-width:850px;margin:40px auto;padding:20px;background:#f3f6fb}.card{background:white;padding:28px;border-radius:16px;box-shadow:0 5px 20px #0001}textarea{width:100%;height:110px;padding:12px;box-sizing:border-box;border-radius:10px;border:1px solid #ccc}button{margin-top:12px;padding:12px 22px;border:0;border-radius:10px;background:#2563eb;color:white;font-size:16px}#a{margin-top:20px;padding:18px;background:#eef4ff;border-radius:10px;white-space:pre-wrap}</style></head><body><div class="card"><h1>📅 Agentic RAG Schedule Assistant</h1><p>Gemini + ChromaDB + RAG + Tool Calling</p><p><b>Try:</b> What do I have scheduled tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting on August 20 at 3 PM.<br>Move my Project Team Meeting to 4 PM.<br>Remove my IEEE Seminar.</p><textarea id="m" placeholder="Type your request..."></textarea><br><button onclick="ask()">Ask Assistant</button><div id="a">Response will appear here...</div></div><script>async function ask(){let m=document.getElementById('m').value,a=document.getElementById('a');a.textContent='Thinking...';try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});let d=await r.json();a.textContent=d.answer||d.detail||'No response'}catch(e){a.textContent='Connection error'}}</script></body></html>'''

@app.post('/chat')
def chat(r:Req):
    try:
        return {'answer':agent(r.message)}
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get('/health')
def health():
    return {'status':'ok','events':col.count(),'today':date.today().isoformat()}
