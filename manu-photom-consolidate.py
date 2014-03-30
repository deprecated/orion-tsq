import json
from pathlib import Path
import argh

positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"

def main():
    positions_paths = positions_dir.glob("*.json")
    db = {}
    for path in positions_paths:
        with path.open() as f:
            data = json.load(f)
        position_id = path.stem

        # First, get the per-position data such as x, y
        db[position_id] = data

        # Second, get the per-(position + line) data
        fit_subdir = fit_dir / position_id
        line_paths = fit_subdir.glob("*.json")
        for linepath in line_paths:
            with linepath.open() as f:
                linedata = json.load(f)
            line_id = linepath.stem
            data[line_id] = linedata
    with open("manu_spectral_fit_db.json", "w") as f:
        json.dump(db, f, indent=2)



if __name__ == "__main__":
    argh.dispatch_command(main)
