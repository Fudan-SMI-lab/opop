"""One-time box4-2 preparation; run remotely with the existing dev-venv Python.

This records environment evidence, not framework implementation. Credentials stay
on the server. Existing checkouts, evidence, dependencies and configs are read only.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

BASE = Path('/root/autodl-tmp')
CODE = BASE / 'dev/opop-v6'
WORK = BASE / 'dev/v6-work'
OUT = WORK / 'preflight-20261020'
PY = BASE / 'dev-venv/bin/python'


def main():
    os.umask(0o077)
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ['configs', 'private-runs', 'private-runtime', 'cache', '.opencode']:
        (WORK / name).mkdir(exist_ok=True)
    provider = BASE / 'dev/work/.opencode/opencode.jsonc'
    shutil.copyfile(provider, WORK / '.opencode/opencode.jsonc')
    os.chmod(WORK / '.opencode/opencode.jsonc', 0o600)
    cfg = yaml.safe_load((BASE / 'dev/work/configs/dev_gpu0_template.yaml').read_text())
    cfg['run']['runs_dir'] = str(WORK / 'private-runs')
    cfg['opencode'].update(launch_cwd=str(WORK), port=15360,
        sandbox_config_path=str(WORK / '.opencode/opencode.jsonc'))
    cfg['opencode'].setdefault('server_env', {}).update({
        'PYTHONPATH': f'{CODE}/src:{CODE}',
        'XDG_DATA_HOME': str(WORK / 'private-runtime/data'),
        'XDG_STATE_HOME': str(WORK / 'private-runtime/state'),
        'XDG_CACHE_HOME': str(WORK / 'private-runtime/cache'),
    })
    cfg['wsl']['venv'] = str(BASE / 'dev-venv')
    cfg['wsl']['triton_cache_dir'] = str(WORK / 'cache/triton')
    config = WORK / 'configs/box42-v6.yaml'
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    os.chmod(config, 0o600)
    launcher = WORK / 'kernel-opt-v6'
    launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        f'export PYTHONPATH={CODE}/src:{CODE}\n'
        'export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
        f'cd {CODE}\n'
        f'exec {PY} -m kernel_optimizer.cli --config {config} "$@"\n')
    launcher.chmod(0o700)
    sources = {
        'probe.py': BASE / 'comm-profiling-check-20261013T234000/probe.py',
        'comm_probe.py': BASE / 'comm-profiling-nsys-check-20261013T235800/comm_probe.py',
        'metrics_probe.py': BASE / 'comm-profiling-nsys-check-20261013T235800/metrics_probe.py',
    }
    for name, source in sources.items():
        shutil.copyfile(source, OUT / name)
    metadata = {
        'base_commit': subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),
        'code': str(CODE), 'python': str(PY), 'work': str(WORK), 'evidence': str(OUT),
        'config': str(config), 'provider_config_path': str(WORK / '.opencode/opencode.jsonc'),
        'provider_config_mode': '0600',
        'provider_copy_matches': provider.read_bytes() == (WORK / '.opencode/opencode.jsonc').read_bytes(),
        'model': cfg['agents']['default_model'],
        'probe_sources': {name: {'source': str(source), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()} for name,source in sources.items()},
    }
    (OUT / 'preparation.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
