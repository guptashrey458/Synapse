import time
import json
import glob
from pathlib import Path

for f in sorted(glob.glob("logs/reasoning/*.log")):
    print(">>", f)

p = Path("logs/reasoning")
seen = set()

while True:
    for lf in p.glob("*.log"):
        if lf not in seen:
            seen.add(lf)
            print(f"\n=== {lf} ===")
        try:
            for line in lf.open():
                try:
                    obj = json.loads(line)
                    status = obj.get("status","").upper()
                    print(("🟢" if status=="SUCCESS" else "🟡" if status=="WARN" else "🔴"), obj.get("msg",""))
                except:
                    print(line.strip())
        except FileNotFoundError:
            pass
    time.sleep(2)