"""Compatibility boundaries can be checked without initializing the SDK/GPU."""
import importlib.metadata
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from utils import isaacsim_compat as compat


TEMPLATE = '''[dependencies]
"isaacsim.asset.browser" = {}
"isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}
[settings.app.extensions]
registryEnabled = true
[settings.app.exts]
folders = ["${app}", "${app}/../source"]
[settings.physics]
useFastCache = false
'''


class RuntimeTests(unittest.TestCase):
    def test_sim5_does_not_change_imports_or_experience(self):
        args = SimpleNamespace(experience="")
        before = list(sys.meta_path)
        with patch.object(compat.importlib.metadata, "version", return_value="5.1.0.0"):
            compat.install_sim6_compatibility()
            compat.configure_sim6_app(args)
        self.assertEqual(before, sys.meta_path)
        self.assertEqual(args.experience, "")

    def test_sim6_alias_finder_is_idempotent_and_narrow(self):
        with patch.object(compat.importlib.metadata, "version", return_value="6.0.0.1"), \
                patch.object(sys, "meta_path", list(sys.meta_path)):
            compat.install_sim6_compatibility()
            compat.install_sim6_compatibility()
            finders = [f for f in sys.meta_path if isinstance(f, compat._TensorAliasFinder)]
            self.assertEqual(len(finders), 1)
            self.assertIsNone(finders[0].find_spec("omni.physics.tensors.other"))
            self.assertIsNotNone(finders[0].find_spec("omni.physics.tensors.impl.api"))

    def test_standalone_version_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "VERSION").write_text("6.0.0-rc.59+release")
            spec = SimpleNamespace(origin=str(root / "__init__.py"))
            with patch.object(compat.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError), \
                    patch.object(compat.importlib.util, "find_spec", return_value=spec):
                self.assertTrue(compat.is_sim6())

    def test_explicit_experience_is_respected(self):
        args = SimpleNamespace(experience="custom.kit")
        with patch.object(compat, "is_sim6", return_value=True), \
                patch.object(compat, "_lab_apps_directory") as locate:
            compat.configure_sim6_app(args)
        self.assertEqual(args.experience, "custom.kit")
        locate.assert_not_called()

    def test_experience_relocates_lab_folders_and_reuses_content_addressed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            apps = root / "lab with spaces" / "apps"
            apps.mkdir(parents=True)
            (apps / "isaaclab.python.kit").write_text(TEMPLATE)
            args = SimpleNamespace(experience="")
            with patch.object(compat, "is_sim6", return_value=True), \
                    patch.object(compat, "_lab_apps_directory", return_value=apps), \
                    patch.dict(compat.os.environ, {"XDG_CACHE_HOME": str(root / "cache")}):
                compat.configure_sim6_app(args)
                destination = Path(args.experience)
                original_mtime = destination.stat().st_mtime_ns
                other = SimpleNamespace(experience="")
                compat.configure_sim6_app(other)
            text = destination.read_text()
            self.assertTrue(destination.name.startswith("robodojo.sim6."))
            self.assertEqual(args.experience, other.experience)
            self.assertEqual(original_mtime, destination.stat().st_mtime_ns)
            self.assertIn('"isaacsim.gui.content_browser"', text)
            self.assertIn('version = "3.11.0"', text)
            self.assertIn("registryEnabled = false", text)
            self.assertIn(apps.as_posix(), text)
            self.assertIn((apps.parent / "source").as_posix(), text)
            self.assertNotIn("${app}", text)
            self.assertIn("useFastCache = false", text)

    def test_unknown_lab_pin_fails_instead_of_loading_a_wrong_extension(self):
        with self.assertRaisesRegex(RuntimeError, "URDF importer pin"):
            compat._experience_text(TEMPLATE.replace("2.4.31", "99.0"), Path("/lab/apps"))

    def test_aliases_are_real_sdk_objects_and_missing_schema_fails(self):
        actual = ModuleType("omni.physics.tensors.api")
        actual.DeformableBodyView = type("Body", (), {})
        actual.DeformableMaterialView = type("Material", (), {})
        actual.create_simulation_view = object()
        physx = ModuleType("omni.physx")
        native_getter = object()
        ifaces = SimpleNamespace(acquire_physx_interface=native_getter)
        schema_type = object()
        pxr = ModuleType("pxr")
        pxr.PhysxSchema = SimpleNamespace()
        pxr.Usd = SimpleNamespace(SchemaRegistry=SimpleNamespace(GetTypeFromSchemaTypeName=lambda _: schema_type))
        modules = {"omni.physics.tensors.api": actual, "omni.physx": physx,
                   "omni.physx.scripts.ifaces": ifaces, "pxr": pxr}
        with patch.dict(sys.modules, modules):
            alias = ModuleType("omni.physics.tensors.impl.api")
            compat._TensorAliasLoader(actual.__name__).exec_module(alias)
            self.assertIs(alias.create_simulation_view, actual.create_simulation_view)
            self.assertIs(alias.SoftBodyView, actual.DeformableBodyView)
            self.assertIs(alias.SoftBodyMaterialView, actual.DeformableMaterialView)
            self.assertIs(physx.acquire_physx_interface, native_getter)
            self.assertIs(pxr.PhysxSchema.PhysxDeformableBodyAPI, schema_type)
            self.assertFalse(hasattr(alias, "create_soft_body_view"))
            del pxr.PhysxSchema.PhysxDeformableBodyAPI
            pxr.Usd.SchemaRegistry.GetTypeFromSchemaTypeName = lambda _: None
            with self.assertRaisesRegex(RuntimeError, "schema is not registered"):
                compat._TensorAliasLoader(actual.__name__).exec_module(ModuleType("alias"))


if __name__ == "__main__":
    unittest.main()
