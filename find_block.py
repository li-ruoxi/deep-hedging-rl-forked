import json

file_path = '/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb'
with open(file_path, 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        if "state_cols = [" in "".join(cell.get('source', [])):
            print(f"Code cell index: {i}")
            if i > 0:
                print("Previous cell: ", "".join(nb['cells'][i-1].get('source', [])))
            print("First line of this cell: ", cell['source'][0] if cell['source'] else "")
            break
