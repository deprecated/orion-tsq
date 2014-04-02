from pathlib import Path
from astropy.table import Table
import argh
import json


positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"
wavrange_dir = Path("Manu-Data") / "WavRanges"

lines = ["4639", "4649", "4651", "4662"]

def main():
    positions_paths = positions_dir.glob("*.json")
    tables = {line: Table() for line in lines}
    for path in positions_paths:
        with path.open() as f:
            data = json.load(f)
        position_id = path.stem

        for line in lines: 
            fitpath = fit_dir / position_id / (line + ".json")
            with fitpath.open() as f:
                fitdata = json.load(f) 
            tables[line].add_row(data['x'], data['y'], fitdata['Flux'])


if __name__ == "__main__":
    argh.dispatch_command(main)
