import json

with open('/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/notebooks/03_rl_training.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if "New parameter sweep 2:" in source:
            print(f"CELL INDEX: {i}")
            for out in cell.get('outputs', []):
                if out['output_type'] == 'display_data':
                    print("Found DataFrame:")
                    try:
                        print("".join(out['data']['text/plain']))
                    except:
                        pass
                elif out['output_type'] == 'stream':
                    print("Found Stream Output:")
                    text = "".join(out['text'])
                    if "Best configuration candidate:" in text:
                        print(text)
