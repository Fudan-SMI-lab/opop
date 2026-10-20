"""Export completed repair experiments, preserving original artifact bytes."""
from pathlib import Path
import hashlib,json,zipfile,datetime

root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
archive=Path('/dev/shm/agent-repair-evidence-20261021.zip')
files={}
large={}
derived={}
limit=2*1024*1024

def export(p,relative=None,force=False):
    if not p.is_file() or p.name in {'opencode.json','opencode.jsonc'} or p.suffix in {'.yaml','.yml','.pt'}:
        return
    name=relative or p.relative_to(root).as_posix()
    if name in files or str(p) in large:return
    if p.stat().st_size>limit and not force:
        large[str(p)]={'bytes':p.stat().st_size,'retention':'Original retained on box4-2; indexed without export'}
        return
    raw=p.read_bytes()
    files[name]={'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    z.writestr(name,raw)

def derive(name,value,source):
    if name in derived:return
    raw=json.dumps(value,indent=2).encode()
    z.writestr(name,raw)
    derived[name]={'source':str(source),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
        'method':'Selected unmodified fields; original source hash is in exports or large_originals.'}

def run(b):
    export(b/'instrumented-candidate.py',force=True)
    for name in ['manifest.json','events.jsonl','request.json','prepared.json','source-inputs.json',
                 'intake.json','result.json','discovery.json','intent.json','resolved-request.json',
                 'source-manifest.json','repository.json','latest-feedback.json','discovery-failure.json',
                 'source-feedback.json']:
        export(b/name)
    result=b/'result.json'
    if result.is_file() and result.stat().st_size>limit:
        value=json.loads(result.read_text())
        if isinstance(value.get('observations'),dict):
            compact={k:value[k] for k in ['plan','scope'] if k in value}
            compact['observations']={name:{k:observation[k] for k in
                ['status','reason','provenance','elapsed_pairs','evidence'] if k in observation}
                for name,observation in value['observations'].items()}
            derive((b.relative_to(root)/'summary.json').as_posix(),compact,result)
    for p in (b/'project').rglob('*.json'):export(p)
    for p in (b/'generation-snapshots').glob('*.json'):
        export(p)
        d=json.loads(p.read_text())
        for ref in [d['prompt_ref'],*[v['ref'] for v in d['files'].values()]]:
            digest=ref.removeprefix('sha256:')
            export(b/'artifacts'/digest,'snapshot-artifacts/'+digest)
        sandbox=Path(d['sandbox'])
        for name in d.get('outputs',[]):
            export(sandbox/name,(b.relative_to(root)/'generated'/sandbox.name/name).as_posix(),force=True)
    for p in (b/'collector-plans').glob('*/result.json'):export(p)
    for p in (b/'sandboxes').glob('*/candidate.py'):export(p,force=True)
    for p in (b/'sandboxes').glob('*/proposal.json'):export(p,force=True)
    for p in (b/'sandboxes').glob('*/selection.json'):export(p,force=True)
    for p in (b/'sandboxes').glob('*/opop_adapter.py'):export(p,force=True)
    for p in [*(b/'trials').rglob('*'), *(b/'version-trials').rglob('*')]:
        if p.is_file() and p.name in {'generated.json','submitted-candidate.py','parameter-search.json',
                'result.json','replay.json','selection.json'}:export(p)
        if p.is_file() and p.name=='result.json':
            d=json.loads(p.read_text())
            def endpoint(value):
                return {'status':value.get('status'),'failure':value.get('failure'),
                    'ranks':[{'rank':r['rank'],'result':{k:r.get('result',{}).get(k) for k in
                        ['valid','checks','coverage','setup','metrics','samples']}} for r in value.get('ranks',[])]}
            compact={k:d.get(k) for k in ['trial','decision','measurement','evaluation','publication','bundle','selection','proposal','candidate_agent']}
            compact['pairs']=[{arm:endpoint(value) for arm,value in pair.items()} for pair in d.get('pairs',[])]
            compact['screening']=[{'revision':o.get('revision'),'parameters':o.get('bundle',{}).get('parameters'),
                'constraints':o.get('constraints'),'result':o.get('result'),'endpoint':endpoint(o.get('run',{}))}
                for o in d.get('parameter_search',{}).get('observations',[])]
            compact['diagnostics']={k:{'status':v.get('status'),'reason':v.get('reason'),'provenance':v.get('provenance'),
                **({'elapsed_pairs':v['elapsed_pairs']} if 'elapsed_pairs' in v else {})}
                for k,v in d.get('resources',{}).get('extra_collectors',{}).get('after',{}).items()}
            compact['collection_plan']=d.get('resources',{}).get('collection_plan')
            compact['derivation']='Selected unmodified fields from original result.json; original hash retained in manifest'
            name=(p.parent.relative_to(root)/'summary.json').as_posix()
            derive(name,compact,p)
    for p in (b/'jobs').glob('*/run/*'):
        if p.is_file() and (p.name=='receipt.json' or p.name.startswith('rank-') and p.suffix in {'.json','.log'}):export(p)
    for p in (b/'jobs').rglob('*'):
        if p.is_file() and (p.suffix in {'.nsys-rep','.sqlite'} or p.suffix=='.json' and any(t in p.name for t in ['trace','timeline','discovery-rank'])):
            export(p)
    if (b/'prepared.json').exists():
        d=json.loads((b/'prepared.json').read_text())
        if d.get('controls_path'):
            controls=Path(d['controls_path'])
            export(controls,(b.relative_to(root)/'controls.json').as_posix())
            source=controls.parent/'reference-source'
            request=json.loads((b/'request.json').read_text())
            for name in ['opop_adapter.py','opop_evaluate.py',*request.get('files',[])]:
                export(source/name,(b.relative_to(root)/'source'/name).as_posix())
    for p in (b/'project/bundles').glob('*.json'):
        d=json.loads(p.read_text())
        for edit in d.get('edits',[]):
            if edit['path']==d.get('installer'):
                export(b/'artifacts'/edit['content_ref'].removeprefix('sha256:'),
                       (b.relative_to(root)/'candidate-code'/(d['id']+'.py')).as_posix(),force=True)

with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in root.iterdir():
        if p.is_file() and p.suffix in {'.json','.py','.log'} and p.name!='export-credential-check.json':export(p)
    for p in sorted(root.rglob('manifest.json')):
        if p.parent.name not in {'inputs','source','reference-source'} and (p.parent/'events.jsonl').exists():run(p.parent)
    for name in ['binding-cost-probe','module-cost-probe','feedback-plan-probe','feedback-plan-baseline-ab']:
        run(root/name)
    for sub in ['source-audit','fresh-source-audit','fresh-fixed-source-audit']:
        for p in (root/sub).rglob('*'):
            if p.is_file() and p.suffix in {'.json','.log','.py'} and p.name!='request.json':export(p)
    for p in root.glob('*/result.json'):export(p)
    for p in root.glob('*/protocol.json'):export(p)
    for p in root.glob('*/completed.json'):export(p)
    for source,row in large.items():
        h=hashlib.sha256()
        with Path(source).open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
        row['sha256']=h.hexdigest()
    z.writestr('manifest.json',json.dumps({'exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Actual Agent repair A/B, optional collection planner probes, interrupted attempts, independent validation and fresh raw-entry cohort. No provider configs or reference tensors exported.',
        'exports':files,'derived_exports':derived,'large_originals':large},indent=2))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'exported_files':len(files),'large_files':len(large)}))
