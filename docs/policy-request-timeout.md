# Policy request timeout

Remote model agents may need longer than the WebSocket client's default request
budget for one planning call. Set a positive finite number of seconds before
launching the evaluator:

```bash
export ROBODOJO_POLICY_REQUEST_TIMEOUT_S=10800
bash scripts/eval_policy.sh ...
```

This environment variable overrides `deploy_cfg.request_timeout_s`. If neither is
set, RoboDojo leaves the shared WebSocket client's default unchanged. Invalid,
zero, negative and nonfinite values fail before connecting. This configures the
response wait budget; it does not change handshake/keepalive settings, the
simulator's control frequency, task horizon or the agent's own episode deadline.

The value is forwarded when creating the synchronous WebSocket model client,
including when connection is deferred until scene initialization.
