from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from DataPipeline.config import LANDING_DIR,RAW_FILES,find_landing_file

def run():
    print(f"Landing: {LANDING_DIR}")
    found={}
    for name in RAW_FILES:
        p=find_landing_file(name); found[name]=str(p); print(f"OK {name}: {p.name} ({p.stat().st_size/1024/1024:.1f} MB)")
    return found
if __name__=="__main__":run()
