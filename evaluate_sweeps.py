import json
from pathlib import Path

models_dir = Path("/Users/roxanagreenleaf/Desktop/deep-hedging-rl-forked/models")
sweep_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("sweep")]

for sdir in sweep_dirs:
    records = []
    for tag_dir in sdir.iterdir():
        if tag_dir.is_dir():
            res_path = tag_dir / "results.json"
            if res_path.exists():
                try:
                    metrics = json.loads(res_path.read_text())
                    records.append({
                        "tag": tag_dir.name,
                        "valid_sharpe": metrics["valid"]["sharpe"],
                        "test_sharpe": metrics["test"]["sharpe"],
                    })
                except Exception:
                    pass
    
    if not records:
        continue
    
    # Flawed selection (Author's original: sort by test_sharpe descending)
    winner_flawed = sorted(records, key=lambda x: (x["test_sharpe"], x["valid_sharpe"]), reverse=True)[0]
    
    # Rigorous selection: sort by valid_sharpe descending
    winner_rigorous = sorted(records, key=lambda x: x["valid_sharpe"], reverse=True)[0]
    
    print(f"=== {sdir.name} ===")
    if winner_flawed["tag"] == winner_rigorous["tag"]:
        print(f"✅ Both methods picked the same model: {winner_flawed['tag']}")
        print(f"   Test Sharpe: {winner_flawed['test_sharpe']:.6f}")
    else:
        print(f"❌ Methods picked DIFFERENT models!")
        print(f"   原作者思路 (看 Test选第一): {winner_flawed['tag']} -> 报告给别人的 Test Sharpe 为 {winner_flawed['test_sharpe']:.6f}")
        print(f"   严谨思路 (只用 Valid选第一): {winner_rigorous['tag']} -> 实际你只能拿到 Test Sharpe {winner_rigorous['test_sharpe']:.6f}")
    print()
