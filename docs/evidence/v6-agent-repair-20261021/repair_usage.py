"""Read local OpenCode accounting; this issues no provider requests."""
from pathlib import Path
import datetime,json,sqlite3

root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
sessions={}
for p in root.rglob('events.jsonl'):
    for line in p.read_text().splitlines():
        e=json.loads(line);v=e.get('payload',{})
        if e.get('type')!='PROJECT_AGENT_STARTED' or not str(v.get('sandbox','')).startswith(str(root)+'/'):continue
        sessions[v['session']]={'role':v['role'],'run':str(p.parent),'sandbox':v['sandbox']}
queue=list(sessions)
for sid in queue:
    for child, in db.execute('select id from session where parent_id=?',(sid,)):
        if child not in sessions:
            sessions[child]={'role':'nested','parent':sid,'run':sessions[sid]['run']};queue.append(child)
rows=[]
for sid,meta in sessions.items():
    messages=[json.loads(row[0]) for row in db.execute('select data from message where session_id=?',(sid,))]
    assistants=[m for m in messages if m.get('role')=='assistant']
    tokens={}
    for m in assistants:
        for key,value in m.get('tokens',{}).items():
            values={key+'.'+child:n for child,n in value.items()} if isinstance(value,dict) else {key:value}
            for name,n in values.items():
                if isinstance(n,(int,float)):tokens[name]=tokens.get(name,0)+n
    rows.append({'session':sid,**meta,'messages':len(assistants),'reported_cost':sum(m.get('cost',0) or 0 for m in assistants),
        'tokens':tokens,'unfinished_messages':sum(not m.get('time',{}).get('completed') for m in assistants)})
summary={'mode':'observe_only','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Unique framework Agent sessions started in this experiment root and descendants, including interrupted attempts. Excludes outer Codex conversation and previous source sessions. Provider estimates, not invoice.',
    'provider_requests_for_accounting':0,'sessions_count':len(rows),'messages':sum(x['messages'] for x in rows),
    'reported_cost':sum(x['reported_cost'] for x in rows),'unfinished_messages':sum(x['unfinished_messages'] for x in rows),'sessions':rows}
(root/'usage-snapshot.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='sessions'}))
