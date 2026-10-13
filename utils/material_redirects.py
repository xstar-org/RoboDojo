"""Redirect known remote MDL assets to bundled repository copies."""

from pathlib import Path
from urllib.parse import urlsplit


_ALUMINUM_CAST_REMOTE_PATHS = {
    "/Materials/2023_1/Base/Metals/Aluminum_Cast.mdl",
    "/Materials/2023_1/Base/Metals/Aluminum_Cast/Aluminum_Cast.mdl",
}
_ALUMINUM_CAST_FILES = (
    "Aluminum_Cast.mdl",
    "Aluminum_Cast_BaseColor.png",
    "Aluminum_Cast_N.png",
    "Aluminum_Cast_ORM.png",
)
# Sdf's layer registry uses weak ownership. Keep patched asset layers alive so
# the following reference operation reuses the in-memory edit.
_PREPARED_LAYERS = {}


def _asset_path_string(value) -> str:
    """Return a string for both strings and ``Sdf.AssetPath`` values."""
    return value if isinstance(value, str) else getattr(value, "path", "")


def _is_remote_aluminum_cast_mdl(value) -> bool:
    path = _asset_path_string(value)
    if not path:
        return False
    parsed = urlsplit(path)
    return parsed.scheme in {"http", "https"} and parsed.path in _ALUMINUM_CAST_REMOTE_PATHS


def _local_aluminum_cast_mdl(repo_root: str | Path):
    """Return the bundled MDL asset path, or ``None`` when incomplete."""
    material_dir = Path(repo_root) / "Assets" / "Material" / "material_0031"
    missing = [name for name in _ALUMINUM_CAST_FILES if not (material_dir / name).is_file()]
    if missing:
        print(
            "[WARN] Cannot redirect remote Aluminum_Cast material; "
            f"bundled files are missing from {material_dir}: {', '.join(missing)}"
        )
        return None

    from pxr import Sdf

    return Sdf.AssetPath((material_dir / "Aluminum_Cast.mdl").resolve().as_posix())


def _remote_asset_specs(layers):
    remote_specs = []

    def collect_specs(prim_spec):
        for attr_spec in prim_spec.attributes.values():
            if _is_remote_aluminum_cast_mdl(attr_spec.default):
                remote_specs.append(attr_spec)
        for child_spec in prim_spec.nameChildren.values():
            collect_specs(child_spec)

    for layer in layers:
        collect_specs(layer.pseudoRoot)
    return remote_specs


def _spec_layer_identifiers(specs) -> str:
    identifiers = sorted({spec.layer.identifier for spec in specs})
    return ", ".join(identifiers)


def redirect_aluminum_cast_asset_layer(usd_path: str, repo_root: str | Path) -> int:
    """Patch a camera-stand layer immediately before it is referenced.

    Scene resets can reopen the asset and discard an earlier registry edit.
    Running this beside ``add_reference_to_stage`` guarantees that Hydra first
    observes the local MDL source. Only the process-local Sdf layer is changed;
    the USD file on disk is never saved.
    """
    normalized_path = str(usd_path).replace("\\", "/")
    camera_stand_suffix = "/Assets/Object/RoboDojo/Geometry/camera_stand/00000/object.usd"
    if not normalized_path.endswith(camera_stand_suffix):
        return 0

    from pxr import Sdf

    identifier = Path(usd_path).absolute().as_posix()
    layer = Sdf.Layer.FindOrOpen(identifier)
    if layer is None:
        print(f"[WARN] Cannot open camera-stand asset for material redirect: {usd_path}")
        return 0

    local_mdl = _local_aluminum_cast_mdl(repo_root)
    if local_mdl is None:
        return 0
    remote_specs = _remote_asset_specs([layer])
    if not remote_specs:
        _PREPARED_LAYERS[identifier] = layer
        return 0

    with Sdf.ChangeBlock():
        for attr_spec in remote_specs:
            attr_spec.default = local_mdl

    _PREPARED_LAYERS[identifier] = layer
    print(
        f"[INFO] Redirected {len(remote_specs)} camera-stand Aluminum_Cast shader source(s) "
        f"before USD reference to bundled material {local_mdl.path}"
    )
    return len(remote_specs)


def redirect_remote_aluminum_cast(stage, repo_root: str | Path) -> int:
    """Point remote Aluminum_Cast shaders at the bundled MDL and textures.

    The NVIDIA MDL referenced by some scene USDs loads its textures from S3.
    A failed texture fetch leaves the material compiling and makes every render
    pass retry it.  RoboDojo's asset bundle contains the same MDL and textures,
    so changing only the shader source keeps the existing material inputs and
    geometry bindings intact.

    Returns the number of shader source attributes changed.
    """
    remote_attrs = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        for attr_name in ("info:mdl:sourceAsset", "info:sourceAsset"):
            attr = prim.GetAttribute(attr_name)
            if attr.IsValid() and _is_remote_aluminum_cast_mdl(attr.Get()):
                remote_attrs.append(attr)

    local_mdl = _local_aluminum_cast_mdl(repo_root)
    if local_mdl is None:
        return 0

    from pxr import Sdf

    # Referenced asset layers can contain material specs that a normal composed
    # stage traversal misses. Rewrite matching specs in every used layer as a
    # fallback for stages assembled outside TaskEnv.
    remote_specs = _remote_asset_specs(stage.GetUsedLayers())

    if not remote_attrs and not remote_specs:
        return 0

    with Sdf.ChangeBlock():
        for attr in remote_attrs:
            attr.Set(local_mdl)
        for attr_spec in remote_specs:
            attr_spec.default = local_mdl

    changed = len(remote_attrs) + len(remote_specs)
    print(
        f"[INFO] Redirected {changed} Aluminum_Cast shader source(s) "
        f"from {_spec_layer_identifiers(remote_specs)} to bundled material {local_mdl.path}"
    )
    return changed
