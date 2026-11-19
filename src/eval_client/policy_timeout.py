"""Optional policy request budget for slow remote-model planning."""
import math
import os


def policy_request_timeout(deploy_cfg):
    """Environment overrides deployment YAML; None preserves the client default."""
    value = os.environ.get('ROBODOJO_POLICY_REQUEST_TIMEOUT_S')
    if value is None:
        value = deploy_cfg.get('request_timeout_s')
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('Policy request timeout must be a positive finite number of seconds')
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Policy request timeout must be a positive finite number of seconds') from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Policy request timeout must be a positive finite number of seconds')
    return seconds
