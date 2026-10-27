"""One fixed B/C/C/B temporal diagnostic after an independent constraint failure."""
import argparse
import json
from pathlib import Path
import statistics

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_box43_next_stage_20261027 import environment, now


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'source', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    fork_run(args.source, args.output)
    search = ProjectSearch(load_config(args.config, []), args.output)
    versions = {'B': search.prepared['revision'], 'C': search.revisions.current().id}
    original = search.request.portable()
    search.request = search.request.model_copy(update={'repeats': 20})
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'source': str(args.source), 'versions': versions, 'order': ['B','C','C','B'],
        'original_request': original, 'diagnostic_request': search.request.portable(),
        'scope': 'One predeclared 80-sample diagnostic. Frozen candidate/parameters and all quality retained; only repeats 5->20. No publication, sample deletion, until-pass repetition or replacement of the failed independent result.'})
    rows = []
    for i, arm in enumerate(['B','C','C','B']):
        before = environment(search.request.devices)
        run = search.execute(search.revisions.revision(versions[arm]), f'endpoint-{i}-{arm}', 'evaluate')
        row = {'position': i, 'arm': arm, 'valid': valid_run(run), 'before': before,
               'after': environment(search.request.devices), 'run': run}
        if row['valid']:
            samples = run['ranks'][0]['result']['samples']
            assert len(samples) == 20
            row['latency_blocks_ms'] = [statistics.median(s['latency_ms'] for s in samples[j:j+5]) for j in range(0,20,5)]
        rows.append(row)
        write_json(args.output/'result.json', {'endpoints': rows})
        print(json.dumps({k:row[k] for k in ['position','arm','valid','latency_blocks_ms'] if k in row}), flush=True)
        if not row['valid']:
            break
    write_json(args.output/'completed.json', {'at': now(), 'endpoints': len(rows), 'all_valid': len(rows)==4 and all(row['valid'] for row in rows)})


if __name__ == '__main__':
    main()
