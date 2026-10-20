from pathlib import Path
from kernel_optimizer.project.candidate_diagnostics import CandidateTrace
import probe
def run(context):
    trace=CandidateTrace([{"id":"unsubmitted-draft", "file":context["parameters"]["candidate"]}])
    with trace:
        result=probe.run(context)
    return {"probe":result,"candidate_diagnostics":trace.report()}
