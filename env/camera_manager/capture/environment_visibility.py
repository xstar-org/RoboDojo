"""Temporarily isolate one cloned environment for camera capture."""

from contextlib import contextmanager

from pxr import Sdf, Usd, UsdGeom, UsdLux

from utils.isaacsim_compat import is_sim6


def _visibility_frontier(prim):
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        yield imageable.GetVisibilityAttr()
    else:
        for child in prim.GetChildren():
            yield from _visibility_frontier(child)


class EnvironmentVisibility:
    def __init__(self, stage, num_envs):
        self._stage = stage
        self._use_session_fields = not is_sim6()
        self._frontiers = []
        self._lights = []
        for env_id in range(num_envs):
            path = f"/World/envs/env_{env_id}"
            root = stage.GetPrimAtPath(path)
            if not root.IsValid():
                raise RuntimeError(f"Camera isolation requires environment root {path}")
            self._frontiers.append(
                [attr for child in root.GetChildren() for attr in _visibility_frontier(child)]
            )
            self._lights.append(
                [prim.GetPath() for prim in Usd.PrimRange(root) if prim.HasAPI(UsdLux.LightAPI)]
            )

    @contextmanager
    def only(self, owner):
        session = self._stage.GetSessionLayer()
        if self._use_session_fields:
            # Sim 5.1 invalidates live PhysX views when the layer stack changes.
            # Restore just the fields we author while keeping the stack fixed.
            saved = []

            def preserve(path, field):
                spec = session.GetObjectAtPath(path)
                present = spec is not None and spec.HasInfo(field)
                saved.append((path, field, present, spec.GetInfo(field) if present else None))

            try:
                self._hide_neighbors(owner, session, preserve)
                yield
            finally:
                for path, field, present, value in reversed(saved):
                    spec = session.GetObjectAtPath(path)
                    if spec is None:
                        continue
                    if present:
                        spec.SetInfo(field, value)
                    else:
                        spec.ClearInfo(field)
            return

        original_layers = list(session.subLayerPaths)
        layer = Sdf.Layer.CreateAnonymous("camera-environment-visibility.usda")
        session.subLayerPaths = [layer.identifier, *original_layers]
        try:
            self._hide_neighbors(owner, layer)
            yield
        finally:
            session.subLayerPaths = original_layers

    def _hide_neighbors(self, owner, layer, preserve=None):
        with Usd.EditContext(self._stage, layer):
            for env_id, attributes in enumerate(self._frontiers):
                if env_id == owner:
                    continue
                for attribute in attributes:
                    if preserve is not None:
                        preserve(attribute.GetPath(), "default")
                    if not attribute.Set(UsdGeom.Tokens.invisible) or attribute.Get() != UsdGeom.Tokens.invisible:
                        raise RuntimeError(f"Could not hide {attribute.GetPath()} for camera isolation")
                for path in self._lights[env_id]:
                    light = self._stage.GetPrimAtPath(path)
                    if preserve is not None:
                        preserve(path, "active")
                    if not light.SetActive(False) or light.IsActive():
                        raise RuntimeError(f"Could not disable {path} for camera isolation")
