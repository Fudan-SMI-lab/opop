from pathlib import Path
import ast
import hashlib
import json
import shutil

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.inner_loop import tune
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json

w = Path('/root/autodl-tmp/dev/v6-work')
source = w / 'p6-qwen-throughput-composition-c'
output = w / 'p6-qwen-draft-project-d'
parent = fork_run(source, output)
search = ProjectSearch(load_config(w / 'configs/box42-v6.yaml', []), output)
discovery = search.current_discovery()
draft = w / 'p6-qwen-draft-diagnostic-d/candidate.py'
selection = json.loads((source / 'sandboxes/selection-29cb5b3b4476/selection.json').read_text())
tree = ast.parse(draft.read_text())
calls = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_CALLS' for t in node.targets))
replacement_calls = [cid for roles in calls.values() for cid in roles.values() if cid]
generated = output / 'diagnostic-envelope'
generated.mkdir()
shutil.copy2(draft, generated / 'candidate.py')
proposal = {
    'implementation': 'composite', 'replacement_calls': replacement_calls,
    'requires': ['trial-0b17ae710c-param-a2600a0f'], 'replaces': [],
    'rationale': 'Developer-created diagnostic envelope around untouched unsubmitted Agent code and original automatic selection. Not an Agent-submitted proposal or project acceptance.',
    'setup_work': 'Use the unchanged draft build function.',
    'quality_risks': ['Unsubmitted joint draft; intermediate markers and swallowed errors require full workload validation.'],
    'parameter_space': {},
}
write_json(generated / 'proposal.json', proposal)
bundle, _ = search.bundle_from_outputs(generated, selection, parent, discovery, 'diagnostic-draft-pointwise')
selected, observations = tune(search, bundle, generated / 'candidate.py', {}, 1)
result = {
    'purpose': 'Isolated full-project screening and automatic diagnostics of unsubmitted Agent draft; no acceptance or publication',
    'proposal_origin': 'Explicit developer diagnostic envelope, not generated Agent proposal',
    'candidate_origin': str(draft), 'candidate_sha256': hashlib.sha256(draft.read_bytes()).hexdigest(),
    'selection_origin': str(source / 'sandboxes/selection-29cb5b3b4476/selection.json'),
    'parent': parent.id, 'current_pointer_unchanged': search.revisions.current().id == parent.id,
    'bundle': bundle.model_dump(mode='json'), 'parameter_search': observations,
}
write_json(output / 'result.json', result)
print(json.dumps({
    'current_pointer_unchanged': result['current_pointer_unchanged'],
    'observations': [
        {'bundle': row['bundle']['id'], 'result': row['result'],
         'rank_results': [{'rank': rank['rank'], 'status': rank['status'], 'error': rank.get('error'),
                          'checks': rank.get('result', {}).get('checks'),
                          'coverage': rank.get('result', {}).get('coverage')} for rank in row['run'].get('ranks', [])],
         'diagnostics': {'status': row['candidate_diagnostics']['status'],
                         'trigger': row['candidate_diagnostics'].get('trigger'),
                         'events': [event for rank in row['candidate_diagnostics'].get('ranks', [])
                                    for event in rank.get('events', []) if event['event'] == 'exception']}
         if row.get('candidate_diagnostics') else None}
        for row in observations['observations']
    ],
}))
