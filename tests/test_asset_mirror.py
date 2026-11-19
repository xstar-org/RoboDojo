import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('asset_mirror', Path(__file__).resolve().parents[1]/'src/eval_client/asset_mirror.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AssetMirrorTests(unittest.TestCase):
    def test_unconfigured_does_not_import_kit(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(module.install_asset_mirror())

    def test_local_and_resolved_bytes_must_both_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b'unchanged material bytes'
            (root/'material.mdl').write_bytes(content)
            item = dict(url='https://assets.example/material.mdl', bytes=len(content),
                        sha256=hashlib.sha256(content).hexdigest())
            config = root/'config.json'
            data = dict(prefix='https://assets.example/', root=str(root), files=[item])
            config.write_text(json.dumps(data))
            aliases = []
            client = SimpleNamespace(Result=SimpleNamespace(OK=0),
                set_alias=lambda *args: aliases.append(args),
                read_file=lambda url: (0, '', content))
            with patch.dict(os.environ, {'ROBODOJO_ASSET_MIRROR_CONFIG':str(config)}, clear=True):
                self.assertEqual(module.install_asset_mirror(client)['files'], [item])
                self.assertEqual(aliases, [('https://assets.example/',root.as_uri()+'/')])
                client.read_file = lambda url: (0, '', b'changed resolver bytes')
                with self.assertRaises(RuntimeError):
                    module.install_asset_mirror(client)
                (root/'material.mdl').write_bytes(b'changed local bytes')
                with self.assertRaises(ValueError):
                    module.install_asset_mirror(client)

    def test_manifest_cannot_escape_directory_or_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'cache'
            cache.mkdir()
            (root/'secret').write_bytes(b'outside')
            config = root/'config.json'
            for url in ['https://other.example/secret','https://assets.example/%2e%2e/secret']:
                config.write_text(json.dumps(dict(prefix='https://assets.example/',root=str(cache),
                    files=[dict(url=url,bytes=7,sha256=hashlib.sha256(b'outside').hexdigest())])))
                with patch.dict(os.environ, {'ROBODOJO_ASSET_MIRROR_CONFIG':str(config)}, clear=True):
                    with self.assertRaises(ValueError):
                        module.install_asset_mirror(SimpleNamespace())
