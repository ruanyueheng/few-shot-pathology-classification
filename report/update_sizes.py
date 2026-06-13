import json
from pathlib import Path
from PIL import Image
d = Path(__file__).parent / "figures"
sizes = {}
for p in sorted(d.glob("*.png")):
    sizes[p.name] = list(Image.open(p).size)
(d / "sizes.json").write_text(json.dumps(sizes, indent=2))
for k, v in sizes.items():
    print(k, v)
