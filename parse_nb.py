import json

with open('/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "new param sweep 1" in source or "Parameter sweep: cadence vs slippage" in source:
            print(f"--- Cell {i} ---")
            print("SOURCE:")
            print(source)
            if 'outputs' in cell:
                for out in cell['outputs']:
                    if out['output_type'] == 'stream':
                        print("OUTPUT STREAM (first 500 chars):")
                        print("".join(out['text'])[:500])
            print("----------------\n")
