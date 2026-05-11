import pandas as pd, numpy as np, sys, yaml
from pathlib import Path
sys.path.insert(0, "experiments/mechanism")
sys.path.insert(0, "experiments/counterfactual")
from shadow_reserve import ShadowReserveBank, ReserveParams
from sr_auction import ShadowReserveAuction, ReserveMode

Path("results/tables").mkdir(parents=True, exist_ok=True)
ETH_PRICE = 3708

def run_replay(df, params):
    bank = ShadowReserveBank(params)
    cutoff = int(len(df)*0.20)
    for _, r in df.iloc[:cutoff].iterrows():
        bank.update(r.to_dict())
    auc = ShadowReserveAuction(bank, ReserveMode.HARD)
    rows = []
    for _, r in df.iloc[cutoff:].iterrows():
        d = r.to_dict()
        orig, sr = auc.run(d)
        bank.update(d)
        rent_delta = (sr.solver_rent - orig.solver_rent) / 1e18 * ETH_PRICE
        rows.append({"reserve_active": sr.reserve_active,
                     "delta_rent_usd": rent_delta,
                     "winner_changed": sr.winner_solver != orig.winner_solver})
    res = pd.DataFrame(rows)
    return {
        "activation_rate":    res["reserve_active"].mean(),
        "mean_delta_rent_usd":res["delta_rent_usd"].mean(),
        "active_delta_rent":  res[res["reserve_active"]]["delta_rent_usd"].mean() if res["reserve_active"].any() else np.nan,
    }

df = pd.read_parquet("data/processed/auctions_main_usd.parquet")
df = df.sort_values("block_timestamp").reset_index(drop=True)

# Baseline
base = ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.10,delta_down=0.05,delta_up=0.10)

configs = [
    ("baseline",           ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.10,delta_down=0.05,delta_up=0.10)),
    ("tau=0.10",           ReserveParams(L=200,H=10,k=2,tau=0.10,eta=0.10,delta_down=0.05,delta_up=0.10)),
    ("tau=0.20",           ReserveParams(L=200,H=10,k=2,tau=0.20,eta=0.10,delta_down=0.05,delta_up=0.10)),
    ("tau=0.40",           ReserveParams(L=200,H=10,k=2,tau=0.40,eta=0.10,delta_down=0.05,delta_up=0.10)),
    ("delta_down=0.02",    ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.10,delta_down=0.02,delta_up=0.10)),
    ("delta_down=0.10",    ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.10,delta_down=0.10,delta_up=0.10)),
    ("eta=0.05",           ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.05,delta_down=0.05,delta_up=0.10)),
    ("eta=0.20",           ReserveParams(L=200,H=10,k=2,tau=0.25,eta=0.20,delta_down=0.05,delta_up=0.10)),
]

rows = []
for name, p in configs:
    print(f"Running {name}...", end=" ", flush=True)
    ShadowReserveBank._global_limiter = None  # reset class state
    # Reset class-level limiter
    r = run_replay(df, p)
    r["config"] = name
    rows.append(r)
    print(f"activation={r['activation_rate']:.1%}  delta_rent=${r['mean_delta_rent_usd']:,.2f}")

result = pd.DataFrame(rows)[["config","activation_rate","mean_delta_rent_usd","active_delta_rent"]]
result.to_csv("results/tables/reserve_sensitivity.csv", index=False)
print("\n✓ reserve_sensitivity.csv saved")
