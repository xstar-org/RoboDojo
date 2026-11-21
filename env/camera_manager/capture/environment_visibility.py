"""Temporarily isolate one cloned environment for camera capture."""

from contextlib import contextmanager

from pxr import Sdf, Usd, UsdGeom, UsdLux


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
        original_layers = list(session.subLayerPaths)
        layer = Sdf.Layer.CreateAnonymous("camera-environment-visibility.usda")
        session.subLayerPaths = [layer.identifier, *original_layers]
        try:
            with Usd.EditContext(self._stage, layer):
                for env_id, attributes in enumerate(self._frontiers):
                    if env_id == owner:
                        continue
                    for attribute in attributes:
                        if not attribute.Set(UsdGeom.Tokens.invisible) or attribute.Get() != UsdGeom.Tokens.invisible:
                            raise RuntimeError(f"Could not hide {attribute.GetPath()} for camera isolation")
                    for path in self._lights[env_id]:
                        light = self._stage.GetPrimAtPath(path)
                        if not light.SetActive(False) or light.IsActive():
                            raise RuntimeError(f"Could not disable {path} for camera isolation")
            yield
        finally:
            session.subLayerPaths = original_layers
