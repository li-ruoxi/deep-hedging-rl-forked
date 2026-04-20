import json

with open('/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "train final" in source.lower() or "final cfg" in source.lower() or "final_cfg" in source.lower():
            print(f"=== Cell {i} ===")
            print(source)
