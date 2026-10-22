"""Reject unusable camera buffers before they become policy observations."""
import numpy as np


class InvalidCameraObservationError(RuntimeError):
    """A capture failed; this is not an unstable layout or a policy failure."""


def invalid_color_buffers(observations):
    invalid = []
    for env_idx, obs in observations.items():
        for camera, view in obs.get('vision', {}).items():
            if 'color' not in view:
                continue
            image = np.asarray(view['color'])
            if not image.size or not np.any(image):
                invalid.append((env_idx, camera))
    return invalid


def capture_valid_observation(capture, render, *, retries=16, render_passes=4):
    """Retry only rendering, without advancing physics, actions, or layout seeds.

    All-zero RGB buffers are treated as a capture failure. Callers supporting
    deliberately blank cameras can bypass this check in their observation config.
    """
    for attempt in range(retries + 1):
        observations = capture()
        invalid = invalid_color_buffers(observations)
        if not invalid:
            return observations
        if attempt < retries:
            for _ in range(render_passes):
                render()
    raise InvalidCameraObservationError(
        f'Empty or all-zero camera buffers after {retries * render_passes} '
        f'render-only refreshes: {invalid}'
    )
