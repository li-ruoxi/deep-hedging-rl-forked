import json

file_path = '/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb'
with open(file_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if "state_cols = [" in "".join(cell.get('source', [])):
            new_source = []
            for line in cell['source']:
                if '"vix","rate_10y","rv_21d","hvol_30d","hvol_91d"' in line:
                    new_source.append('        "vix","rate_10y","rv_21d","hvol_30d","hvol_91d"\\n",\n')
                elif '"ret_1d","close_spy_z60", "rv_60d"' in line:
                    pass # remove
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)
