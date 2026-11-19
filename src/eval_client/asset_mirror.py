"""Optional exact-byte asset transport for hosts without NVIDIA CDN access."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, unquote


def install_asset_mirror(client=None):
    """Call after AppLauncher, before creating the environment; default is a no-op.

    A trusted deployment config specifies one HTTPS prefix, a local directory,
    and the expected SHA-256/size of every mirrored file. No USD is rewritten.
    """
    config_path = os.environ.get('ROBODOJO_ASSET_MIRROR_CONFIG')
    if not config_path:
        return None
    config = json.loads(Path(config_path).read_text())
    prefix = config['prefix']
    parsed = urlsplit(prefix)
    if (parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment
            or not prefix.endswith('/')):
        raise ValueError('Asset mirror prefix must be an HTTPS directory URL')
    root = Path(config['root']).resolve(strict=True)
    if not root.is_dir() or not config['files']:
        raise ValueError('Asset mirror requires a directory and nonempty file manifest')
    verified = []
    for item in config['files']:
        url = item['url']
        if not url.startswith(prefix):
            raise ValueError('Asset URL is outside the mirror prefix')
        relative = unquote(url[len(prefix):])
        if not relative or '?' in relative or '#' in relative or relative.startswith('/'):
            raise ValueError('Asset URL must resolve to a relative file')
        path = (root/relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Asset path escapes the mirror directory')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if type(item['bytes']) is not int or len(content) != item['bytes'] or digest != item['sha256']:
            raise ValueError('Local asset bytes differ from the mirror manifest')
        verified.append(dict(url=url, sha256=digest, bytes=len(content)))
    if client is None:
        import omni.client as client
    client.set_alias(prefix, root.as_uri()+'/')
    for item in verified:
        result, _, content = client.read_file(item['url'])
        if (result != client.Result.OK or len(content) != item['bytes']
                or hashlib.sha256(bytes(content)).hexdigest() != item['sha256']):
            raise RuntimeError('Asset resolver did not return the verified mirror bytes')
    audit = dict(prefix=prefix, target=root.as_uri()+'/', files=verified)
    audit_path = os.environ.get('ROBODOJO_ASSET_MIRROR_AUDIT')
    if audit_path:
        Path(audit_path).write_text(json.dumps(audit, indent=2))
    print('[assets] Exact-byte local mirror verified: '+str(len(verified))+' files', flush=True)
    return audit
