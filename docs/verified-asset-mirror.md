# Verified local asset mirror

Some compute environments cannot fetch the NVIDIA material CDN during scene
creation. A blocked fetch can stall the first PhysX reset before the first
policy observation. An optional exact-byte mirror resolves the existing URLs
to local files without editing USD, textures, physics or evaluator criteria.

Set `ROBODOJO_ASSET_MIRROR_CONFIG` to a trusted JSON file before evaluation:

```json
{
  "prefix": "https://omniverse-content-production.s3.us-west-2.amazonaws.com/",
  "root": "/private/cache/nvidia-materials",
  "files": [
    {"url": "https://omniverse-content-production.s3.us-west-2.amazonaws.com/path/material.mdl",
     "sha256": "sha256-of-original-file", "bytes": 1234}
  ]
}
```

Place each original file under the root at its URL-relative path. Generate the
manifest from the original downloaded bytes. After AppLauncher, before scene
creation, the evaluator verifies every local file, installs an OmniClient alias,
then reads every listed URL through the resolver and verifies the returned bytes.
Missing, changed or incorrectly resolved files abort startup. With no config,
startup is unchanged. Optionally set `ROBODOJO_ASSET_MIRROR_AUDIT` to save the
verified URL/hash/size mapping. This configuration belongs to trusted deployment;
it must not be provided by the evaluated agent. Keep it identical across arms.
