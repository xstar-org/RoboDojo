"""Rigid-task compatibility for Isaac Lab 2.3.2 on Isaac Sim 6.0.

Importing this module does not initialize Kit. Install the import aliases before
importing Isaac Lab, then configure the experience before creating AppLauncher.
Isaac Sim 5.x keeps its existing imports, experience and capture path.
"""

import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def is_sim6():
    """Detect the installed runtime without importing app-dependent SDK code."""
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        # Standalone packages may not publish Python distribution metadata.
        spec = importlib.util.find_spec("isaacsim")
        version_file = Path(spec.origin).parent / "VERSION" if spec and spec.origin else None
        if version_file is None or not version_file.is_file():
            return False
        version = version_file.read_text(encoding="utf-8")
    return re.match(r"^6\.", version.strip()) is not None


class _TensorAliasLoader(importlib.abc.Loader):
    def __init__(self, target=None):
        self.target = target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.target is None:
            return
        real = importlib.import_module(self.target)
        for key, value in vars(real).items():
            if key not in {"__name__", "__spec__", "__loader__", "__package__"}:
                setattr(module, key, value)
        # These aliases allow old Lab's eager type annotations to load rigid
        # assets. They do not emulate removed soft-body creation/operations.
        for old, new in [("SoftBodyView", "DeformableBodyView"),
                         ("SoftBodyMaterialView", "DeformableMaterialView")]:
            if not hasattr(module, old):
                setattr(module, old, getattr(real, new))
        physx = importlib.import_module("omni.physx")
        if not hasattr(physx, "acquire_physx_interface"):
            ifaces = importlib.import_module("omni.physx.scripts.ifaces")
            physx.acquire_physx_interface = ifaces.acquire_physx_interface
        from pxr import PhysxSchema, Usd

        if not hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
            schema_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName("OmniPhysicsDeformableBodyAPI")
            if not schema_type:
                raise RuntimeError("Isaac Sim 6 deformable schema is not registered")
            # HasAPI accepts the real registered Tf.Type. Never substitute a
            # dummy schema or an always-false deformable/material type check.
            PhysxSchema.PhysxDeformableBodyAPI = schema_type


class _TensorAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "omni.physics.tensors.impl":
            return importlib.util.spec_from_loader(fullname, _TensorAliasLoader(), is_package=True)
        if fullname == "omni.physics.tensors.impl.api":
            loader = _TensorAliasLoader("omni.physics.tensors.api")
            return importlib.util.spec_from_loader(fullname, loader)
        return None


def install_sim6_compatibility():
    """Register old import names for Sim6; the caller selects the runtime."""
    if not any(isinstance(finder, _TensorAliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TensorAliasFinder())


def _lab_apps_directory():
    spec = importlib.util.find_spec("isaaclab")
    if spec and spec.origin:
        for parent in Path(spec.origin).resolve().parents:
            candidate = parent / "apps"
            if (candidate / "isaaclab.python.kit").is_file():
                return candidate
    candidate = Path(__file__).resolve().parents[1] / "third_party/IsaacLab/apps"
    if (candidate / "isaaclab.python.kit").is_file():
        return candidate
    raise RuntimeError("Cannot locate Isaac Lab 2.3.2 apps/isaaclab.python.kit")


def _experience_text(template, apps):
    """Adapt the pinned Lab experience; preserve its renderer/physics settings."""
    text = template.replace('"isaacsim.asset.browser"', '"isaacsim.gui.content_browser"')
    text, replacements = re.subn(
        r'("isaacsim\.asset\.importer\.urdf"\s*=\s*\{version\s*=\s*)"2\.4\.31"',
        r'\g<1>"3.11.0"', text,
    )
    if replacements != 1:
        raise RuntimeError("Expected Isaac Lab 2.3.2 URDF importer pin 2.4.31")
    text = text.replace("registryEnabled = true", "registryEnabled = false")
    text = text.replace('version = "5.1.0"', 'version = "6.0.0"')
    # The generated experience lives in a cache, so ${app} no longer denotes
    # Lab's apps directory. Resolve its two search folders, not user paths.
    text = text.replace('"${app}"', json.dumps(apps.as_posix()))
    text = text.replace('"${app}/../source"', json.dumps((apps.parent / "source").as_posix()))
    return text


def configure_sim6_app(args):
    """Configure Sim6's experience; the caller selects the runtime."""
    if getattr(args, "experience", ""):
        return
    apps = _lab_apps_directory()
    text = _experience_text((apps / "isaaclab.python.kit").read_text(encoding="utf-8"), apps)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "robodojo/sim6"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:20]
    # Kit treats a dash suffix as an extension version. Using isaaclab-<hash>
    # would shadow the actual isaaclab dependency and create a solver cycle.
    destination = cache / f"robodojo.sim6.{digest}.kit"
    if not destination.exists():
        # Concurrent workers must never observe a partially written experience.
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    args.experience = str(destination)
