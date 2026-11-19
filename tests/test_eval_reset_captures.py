import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_setup_scene():
    source = Path(__file__).resolve().parents[1] / "src/eval_client/eval_env.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(module)
                  if isinstance(node, ast.FunctionDef) and node.name == "setup_scene")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["setup_scene"]


def make_environment(config):
    events = []
    env = SimpleNamespace(
        sim_config=config,
        num_envs=4,
        episode_nums=4,
        unstable_nums=0,
        scene_manager=SimpleNamespace(
            apply_saved_poses=lambda **_: None,
            layout_manager=SimpleNamespace(check_layout_stability=lambda _: (True, [])),
        ),
        obs_manager=SimpleNamespace(
            render_for_capture=lambda: events.append("render_for_capture"),
            get_obs=lambda: events.append("observation"),
        ),
        _align_layout_success=lambda: None,
        render=lambda: events.append("render"),
        sim_step=lambda: events.append("physics"),
        physx_monitor_enabled=False,
    )
    return env, events


@pytest.mark.parametrize("config,start", [({}, 0), ({"eval_reset_capture_count": 4}, 180),
                                          ({"eval_reset_capture_count": 0}, 200)])
def test_reset_captures_keep_every_settling_step(config, start):
    env, events = make_environment(config)
    load_setup_scene()(env)
    expected = ["render"] * 10
    for step in range(200):
        expected.append("physics")
        if step >= start and step % 5 == 0:
            expected.extend(("render_for_capture", "observation"))
    assert events == expected


@pytest.mark.parametrize("count", [-1, 41, True, "4"])
def test_invalid_capture_count_fails_before_settling(count):
    env, events = make_environment({"eval_reset_capture_count": count})
    with pytest.raises(ValueError, match="eval_reset_capture_count"):
        load_setup_scene()(env)
    assert not events
