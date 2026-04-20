import json

with open('/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "BASE_TRAIN_KW" in source:
            print("===========================")
            print(f"Cell {i} source:")
            print("\n".join([line for line in source.split("\n") if "BASE_TRAIN_KW" in line or "train_end" in line or "window" in line or "lr=" in line or "steps=" in line or "entropy" in line]))
