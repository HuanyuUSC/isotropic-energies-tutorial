"""Build the interactive viewer for spherical maps of genus-0 meshes.

Each mesh becomes one gzip-compressed binary blob (vertices, faces, every spherical map); the
browser decompresses a mesh only when it is selected, and computes per-face distortion itself.

    python figures/make_viewer.py single <out.html> [mesh ...]   # one file, blobs inlined
    python figures/make_viewer.py folder <out-dir>               # index.html + data/<mesh>.bin

Sources: the Stanford cache (assets/fig/cache) and the batch cache (assets/viewer/cache100).
The folder build fetches its blobs, so open it through a web server (or publish it).
"""
import base64
import glob
import gzip
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

from distortion.render import earth_texture

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "..", "assets")
SOURCES = [("Benchmark (100 meshes)", os.path.join(ASSETS, "viewer", "cache100")),
           ("Talk meshes (Stanford, thesis)", os.path.join(ASSETS, "fig", "cache"))]
TALK_SET = ["skull", "max", "buddha", "GardenPig", "santa-20000", "filledPF", "fandisk", "bunnyih-20000-normal", "cow_thesis", "hand_thesis", "spot", "bunny", "armadillo", "hand", "horse1", "cow", "ant1", "octopus1", "fish", "plane",
            "bunnyih-20000-normal", "gargoyle100K", "lucy100k", "camel1", "dragon", "flamingo", "woman"]
# thesis meshes that are identical to benchmark meshes (same vertices to 1e-7): show the benchmark copy only
DUPLICATES = {"gardenpig", "santa", "bunny_thesis"}
MAP_KEYS = ["one_hole", "naive", "reference", "conformal", "symdir"]


def blob(npz, info):
    d = np.load(npz)
    V, F = d["V"].astype(np.float32), d["F"].astype(np.uint32)
    parts, offsets = [], {}
    keys = [k for k in MAP_KEYS if k in d] + sorted(k for k in d.files if k.startswith(("nu_", "en_")))
    maps = {k: d[k].astype(np.float32) for k in keys}
    header = dict(n=len(V), m=len(F), info=info)
    # two passes: header size depends on offsets, offsets on header size
    for _ in range(2):
        h = json.dumps(dict(header, V=offsets.get("V", 0), F=offsets.get("F", 0),
                            maps={k: offsets.get(k, 0) for k in maps})).encode()
        start = 4 + len(h)
        start += (-start) % 4
        off = start
        offsets["V"] = off
        off += V.nbytes
        offsets["F"] = off
        off += F.nbytes
        for k, S in maps.items():
            offsets[k] = off
            off += S.nbytes
        h2 = json.dumps(dict(header, V=offsets["V"], F=offsets["F"], maps={k: offsets[k] for k in maps})).encode()
        if len(h2) == len(h):
            break
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(h2)))
    buf.write(h2)
    buf.write(b"\0" * ((-(4 + len(h2))) % 4))
    buf.write(V.tobytes())
    buf.write(F.tobytes())
    for S in maps.values():
        buf.write(S.tobytes())
    return gzip.compress(buf.getvalue(), 6)


def catalog():
    out = {}
    for group, folder in SOURCES:
        log = {}
        if os.path.exists(os.path.join(folder, "batch_log.json")):
            log = {r["name"]: r for r in json.load(open(os.path.join(folder, "batch_log.json")))}
        for npz in sorted(glob.glob(os.path.join(folder, "*.npz")), key=str.lower):
            name = os.path.basename(npz)[:-4]
            if name in DUPLICATES:
                continue
            info = {k: v for k, v in log.get(name, {}).items() if k in ("symdir_iters", "symdir_time", "symdir_start")}
            out[name] = dict(group=group, npz=npz, info=info, label=name)
    return out


def texture_uri():
    buf = io.BytesIO()
    Image.fromarray(earth_texture()).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cube_table(N=257):
    """Conformal cubed-sphere lookup (see conformal_cube.py) as base64 half floats."""
    from conformal_cube import table
    return dict(n=N, half=base64.b64encode(table(N).astype(np.float16).tobytes()).decode())


def write(html_path, meshes, default):
    data = dict(meshes=meshes, texture=texture_uri(), cube=cube_table(), default=default)
    html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf8").read().replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    open(html_path, "w", encoding="utf8").write(html)
    print(f"wrote {html_path} ({os.path.getsize(html_path) / 1e6:.1f} MB, {len(meshes)} meshes)")


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    cat = catalog()
    if mode == "single":
        names = sys.argv[3:] or [n for n in TALK_SET if n in cat]
        meshes = {n: dict(group=cat[n]["group"], label=n, info=cat[n]["info"],
                          inline=base64.b64encode(blob(cat[n]["npz"], cat[n]["info"])).decode()) for n in names}
        write(out, meshes, names[0])
    else:
        os.makedirs(os.path.join(out, "data"), exist_ok=True)
        meshes = {}
        for n, c in cat.items():
            open(os.path.join(out, "data", n + ".bin"), "wb").write(blob(c["npz"], c["info"]))
            meshes[n] = dict(group=c["group"], label=n, info=c["info"], url=f"data/{n}.bin")
        write(os.path.join(out, "index.html"), meshes, "hand" if "hand" in cat else next(iter(cat)))
