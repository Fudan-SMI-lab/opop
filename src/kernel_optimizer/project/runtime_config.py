"""Project role configuration; actual GPU work belongs to UUID-bound workers."""


def project_agent_config(config):
    result = config.model_copy(deep=True)
    result.opencode.server_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return result
