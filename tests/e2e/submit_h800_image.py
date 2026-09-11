"""Submit the local offline Open-MOPD chain and wait for the H800 RJob."""
import os
from datetime import datetime
from pathlib import Path

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import spawn_tasks

assert os.environ['STEPMIND_BACKEND'] == 'rjob'
assert os.environ['BRAINPP_ACCESS_KEY'] and os.environ['BRAINPP_SECRET_KEY']
os.chdir('/data/ycfeng')
run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
os.environ['EXP_ID'] = f'openmopd-h800-{run_id}'
work = Path('/data/ycfeng/tmp') / f'openmopd-image-{run_id}'
work.mkdir()
command = f'bash /data/ycfeng/Open-MOPD/tests/e2e/run_h800_image.sh'
cfg = ResourceConfig(
    cpu=16, gpu=1, mem_gb=132, replica=1,
    image='hub.i.basemind.com/stepmind/megatron-step:test-optimus-3.24.0-stepccl-0.0.5.post5-vllm-0.11.0.post37-20260817-2733176',
    positive_tags=['H800'], extra_requirements=[], mounts=[],
    custom_resources=[], envs={'WORK': str(work), 'NVIDIA_SMI': '/usr/local/nvidia/bin/nvidia-smi'}, command='{COMMAND}',
    task_specs={'default': {'is_critical': True}},
)
print('WORK', work, flush=True)
workers = spawn_tasks(cfg, command=command, charged_group='codesign',
                      use_image=True, code_mount_point='/data/ycfeng')
print('RJOB_NAME', workers.rjob.meta.name, flush=True)
print('FINAL_STATUS', workers.poll(), flush=True)
