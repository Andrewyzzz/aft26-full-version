import pandas as pd, numpy as np
from pathlib import Path
Path("results/tables").mkdir(parents=True, exist_ok=True)
ETH = 3708; WEI = 1e18

for name, fname, winner_col in [
    ("mechanical",     "mechanical_replay.csv",     "winner_changed"),
    ("behavior_aware", "behavior_aware_replay.csv",  "winner_exited"),
    ("adversarial",    "adversarial_replay.csv",     None),
]:
    r = pd.read_csv(f"results/tables/{fname}")
    act = r[r["reserve_active"]==True]
    non = r[r["reserve_active"]==False]

    rent_col = "delta_rent"
    def usd(s): return s / WEI * ETH

    print(f"\n=== {name} ===")
    print(f"  n_total:          {len(r):,}")
    print(f"  activation_rate:  {r['reserve_active'].mean():.1%}")
    if rent_col in r.columns:
        print(f"  overall delta_rent_usd:       ${usd(r[rent_col]).mean():,.2f}")
        print(f"  activated delta_rent_usd:     ${usd(act[rent_col]).mean():,.2f}" if len(act) else "  (no activated)")
        print(f"  non-activated delta_rent_usd: ${usd(non[rent_col]).mean():,.2f}" if len(non) else "  (no non-activated)")
        check = r["reserve_active"].mean() * usd(act[rent_col]).mean() + \
                (1-r["reserve_active"].mean()) * usd(non[rent_col]).mean() if len(act) and len(non) else np.nan
        print(f"  weighted check:               ${check:,.2f}  (should ≈ overall)")

rows = []
for name, fname, _ in [
    ("mechanical","mechanical_replay.csv",None),
    ("behavior_aware","behavior_aware_replay.csv",None),
    ("adversarial","adversarial_replay.csv",None),
]:
    r = pd.read_csv(f"results/tables/{fname}")
    act = r[r["reserve_active"]==True]
    non = r[r["reserve_active"]==False]
    row = {"replay_mode": name,
           "n_eval": len(r),
           "activation_rate": r["reserve_active"].mean()}
    if "delta_rent" in r.columns:
        row["overall_delta_rent_usd"]       = (r["delta_rent"]/WEI*ETH).mean()
        row["activated_delta_rent_usd"]     = (act["delta_rent"]/WEI*ETH).mean() if len(act) else np.nan
        row["non_activated_delta_rent_usd"] = (non["delta_rent"]/WEI*ETH).mean() if len(non) else np.nan
    if "theorem1_holds" in r.columns:
        row["theorem1_holds_frac"] = r["theorem1_holds"].mean()
    rows.append(row)

result = pd.DataFrame(rows)
result.to_csv("results/tables/replay_decomposition.csv", index=False)
print("\n✓ replay_decomposition.csv saved")
