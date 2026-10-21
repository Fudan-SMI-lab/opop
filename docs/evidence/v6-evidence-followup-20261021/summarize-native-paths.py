from pathlib import Path
import hashlib, json, sqlite3

root = Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
cases = [
    ('ddp-feedback-continuation', 'trial-f7b4b8b98a'),
    ('fresh/qwen-ttft', 'trial-57b13527f6'),
    ('fresh/qwen-ttft', 'trial-fc57bca7b9'),
    ('fresh/resnet-latency', 'trial-a9f72afa23'),
]
rows = []
for run, trial in cases:
    source = root / run / 'trials' / trial / 'result.json'
    result = json.loads(source.read_text())
    nsys = result['resources']['extra_collectors']['after']['nsys']
    row = {'run': run, 'trial': trial, 'source': str(source),
           'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'provenance': nsys['provenance'], 'ranks': []}
    for rank, evidence in enumerate(nsys['ranks']):
        path = Path(evidence['evidence'])
        db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
        kernels = db.execute('SELECT s.value, k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
        apis = db.execute("SELECT s.value, count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY s.value").fetchall()
        groups = {}
        for name, graph in kernels:
            if any(t in name.lower() for t in ['nccl', 'flash', 'fmha', 'bias_relu', 'stem', 'pool']):
                key = (name, bool(graph))
                groups[key] = groups.get(key, 0) + 1
        row['ranks'].append({'rank': rank, 'sqlite': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'cuda_graph_launch_apis': dict(apis), 'kernel_count': len(kernels),
            'graph_node_kernel_count': sum(bool(graph) for _, graph in kernels),
            'nccl_kernel_count': sum('nccl' in name.lower() for name, _ in kernels),
            'selected_kernel_counts': [{'name': k[0], 'graph_node': k[1], 'count': v} for k, v in sorted(groups.items())]})
        db.close()
    rows.append(row)
summary = {'scope': 'Read-only native CUDA API/kernel inventory from the exact candidate post-trial Nsight SQLite records. GraphNodeId and GraphLaunch corroborate execution; intrusive profiler durations are not performance acceptance. Selected kernel name filters are only for this developer report, not Agent inputs or framework routing.', 'runs': rows}
(root / 'native-path-summary.json').write_text(json.dumps(summary, indent=2))
for row in rows:
    print(row['run'], row['trial'])
    for rank in row['ranks']:
        print({k: v for k, v in rank.items() if k not in ['selected_kernel_counts', 'sqlite', 'sha256']})
        print('selected', [(k['name'][:110], k['graph_node'], k['count']) for k in rank['selected_kernel_counts']])
