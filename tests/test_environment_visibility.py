"""Exercise Sim 5 camera isolation against the USD session layer."""

from contextlib import nullcontext
import unittest
from unittest.mock import patch

try:
    from pxr import Sdf, Usd, UsdGeom, UsdLux
except ImportError:
    Usd = None


@unittest.skipUnless(Usd is not None, "USD Python bindings are not available")
class TestEnvironmentVisibility(unittest.TestCase):
    def test_sim5_preserves_session_layer_stack_and_opinions(self):
        from env.camera_manager.capture import environment_visibility

        stage = Usd.Stage.CreateInMemory()
        session = stage.GetSessionLayer()
        prior_layer = Sdf.Layer.CreateAnonymous("existing.usda")
        session.subLayerPaths = [prior_layer.identifier]
        original_edit_layer = stage.GetEditTarget().GetLayer()

        owner = UsdLux.RectLight.Define(stage, "/World/envs/env_0/light").GetPrim()
        other = "/World/envs/env_1"
        peer = UsdLux.RectLight.Define(stage, other + "/light").GetPrim()
        peer_path = peer.GetPath()
        authored = UsdGeom.Xform.Define(stage, other + "/authored").GetVisibilityAttr()
        inherited = UsdGeom.Xform.Define(stage, other + "/inherited").GetVisibilityAttr()
        with Usd.EditContext(stage, session):
            authored.Set(UsdGeom.Tokens.inherited)
            peer.SetActive(True)
            owner.SetCustomDataByKey("unrelated", "preserved")

        with patch.object(environment_visibility, "is_sim6", return_value=False):
            visibility = environment_visibility.EnvironmentVisibility(stage, 2)
        for abort in (False, True):
            with self.subTest(abort=abort):
                capture = self.assertRaisesRegex(ValueError, "capture interrupted") if abort else nullcontext()
                with capture, visibility.only(0):
                    self.assertEqual(session.subLayerPaths, [prior_layer.identifier])
                    self.assertEqual(authored.Get(), UsdGeom.Tokens.invisible)
                    self.assertEqual(inherited.Get(), UsdGeom.Tokens.invisible)
                    self.assertFalse(stage.GetPrimAtPath(peer_path).IsActive())
                    self.assertTrue(owner.IsActive())
                    if abort:
                        raise ValueError("capture interrupted")

                self.assertEqual(stage.GetEditTarget().GetLayer(), original_edit_layer)
                self.assertEqual(session.subLayerPaths, [prior_layer.identifier])
                self.assertTrue(stage.GetPrimAtPath(peer_path).IsActive())
                self.assertTrue(session.GetObjectAtPath(peer_path).GetInfo("active"))
                self.assertEqual(session.GetObjectAtPath(authored.GetPath()).GetInfo("default"),
                                 UsdGeom.Tokens.inherited)
                inherited_spec = session.GetObjectAtPath(inherited.GetPath())
                self.assertFalse(inherited_spec and inherited_spec.HasInfo("default"))
                self.assertEqual(owner.GetCustomDataByKey("unrelated"), "preserved")
