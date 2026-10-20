"""Check the public export against configured credentials without emitting values."""
from pathlib import Path
import datetime,json,re,zipfile,yaml,hashlib
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
archive=Path('/dev/shm/agent-repair-evidence-20261021.zip')
secrets=set()
def visit(value,key=''):
    if isinstance(value,dict):
        for k,v in value.items():visit(v,str(k))
    elif isinstance(value,list):
        for v in value:visit(v,key)
    elif isinstance(value,str) and len(value)>12 and re.search(r'(?i)(api[_-]?key|password|authorization|(^|_)token$|(^|_)secret$)',key):
        if not value.startswith(('/', 'http:', 'https:')):secrets.add(value.encode())
config_paths={Path('/root/.config/opencode/opencode.jsonc')}
for p in [Path('/root/opop-v6-stage3/deployment.yaml'),*root.glob('*.yaml')]:
    settings=yaml.safe_load(p.read_text());visit(settings)
    configured=settings.get('opencode',{}).get('sandbox_config_path')
    if configured:config_paths.add(Path(configured))
for p in config_paths:
    if not p.is_file():continue
    raw=p.read_text()
    # JSON string values can be read without treating JSONC comments as JSON.
    for value in re.findall(r'(?i)"(?:apiKey|api_key|key|token|password|authorization)"\s*:\s*"((?:\\.|[^"\\])*)"',raw):
        value=json.loads('"'+value+'"')
        if len(value)>12 and not value.startswith(('{env:', '/', 'http:', 'https:')):secrets.add(value.encode())
patterns=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{40,}',rb'sk-[A-Za-z0-9_-]{30,}']
matches=[];checked=0
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if name.endswith('/'):continue
        data=z.read(name);checked+=1
        if any(s in data for s in secrets) or any(re.search(p,data) for p in patterns):matches.append(name)
result={'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files_checked':checked,'configured_secret_values_checked':len(secrets),
    'matching_files':matches,'passed':not matches,'scope':'Known configured API/password/token values and common token formats; no secret values emitted. Provider configs excluded by exporter.'}
(root/'export-credential-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
if matches:raise SystemExit(1)
