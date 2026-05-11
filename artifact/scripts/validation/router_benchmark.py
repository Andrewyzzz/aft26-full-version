"""
P0-E: Blue-chip router benchmark

For each blue-chip auction with valid clearing prices, computes:
  CoW execution rate = price[sell_token] / price[buy_token] (from clearingPrices)
  DEX reference rate = DeFiLlama hourly price ratio

CoW improvement bps = 10^4 * (CoW rate - DEX rate) / DEX rate
Positive = CoW is better than public DEX quote.

Then regresses improvement on fragility bucket to test whether high-fragility
auctions have weaker CoW advantage.

Output: results/tables/router_benchmark.csv
        paper/tables/router_benchmark.tex
"""
import json, math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

BLUE = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": ("WETH",  18),
    "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee": ("ETH",   18),
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": ("USDC",  6),
    "0xdac17f958d2ee523a2206206994597c13d831ec7": ("USDT",  6),
    "0x6b175474e89094c44da98b954eedeac495271d0f": ("DAI",   18),
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": ("WBTC",  8),
}
BLUE_ADDRS = set(BLUE.keys())

OUT = Path("results/tables")
TEX = Path("paper/tables")

# ── 1. Extract clearing prices from raw JSONL ─────────────────
print("Extracting blue-chip clearing prices from JSONL...")
records = []
with open("data/raw/cow_mainnet_full.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        auction_id = rec.get("auctionId")
        block      = rec.get("auctionStartBlock", 0)
        ts         = rec.get("_block_timestamp", 0)

        for sol in rec.get("solutions", []):
            if not sol.get("isWinner", False) or sol.get("filteredOut", False):
                continue
            cp = sol.get("clearingPrices") or {}
            if not cp:
                continue
            tokens = [k.lower() for k in cp.keys()]
            # Only 2-token blue-chip pairs
            if len(tokens) != 2 or not all(t in BLUE_ADDRS for t in tokens):
                continue
            t0, t1 = tokens[0], tokens[1]
            p0, p1 = int(cp[t0]), int(cp[t1])
            if p0 == 0 or p1 == 0:
                continue
            # CoW rate: how many t1 per t0 (normalized by decimals)
            d0, d1 = BLUE[t0][1], BLUE[t1][1]
            # clearingPrices are in wei units: price[t] = value of 1 wei of t in numeraire
            # rate t0→t1 = price[t0] / price[t1] (in numeraire units), adjusted for decimals
            cow_rate = (p0 / p1) * (10 ** (d1 - d0))

            orders = sol.get("orders", [])
            sell_amt = int(orders[0].get("sellAmount", "0")) if orders else 0
            buy_amt  = int(orders[0].get("buyAmount", "0"))  if orders else 0

            records.append({
                "auction_id": auction_id,
                "block":      block,
                "timestamp":  ts,
                "token_sell": t0,
                "token_buy":  t1,
                "symbol_sell": BLUE[t0][0],
                "symbol_buy":  BLUE[t1][0],
                "cow_rate":   cow_rate,
                "sell_amt":   sell_amt,
                "buy_amt":    buy_amt,
            })

cp_df = pd.DataFrame(records)
cp_df["dt"] = pd.to_datetime(cp_df["timestamp"], unit="s", utc=True)
print(f"  Found {len(cp_df):,} blue-chip auctions with clearing prices")
print(f"  Pairs: {cp_df['symbol_sell'].value_counts().head(8).to_dict()}")

# ── 2. Load DeFiLlama hourly prices ──────────────────────────
prices = pd.read_parquet("data/processed/token_prices_full.parquet")
prices["date"] = pd.to_datetime(prices["date"])
# Build hourly price lookup by flooring to nearest hour
def get_price(token_addr, dt_utc):
    day = dt_utc.normalize()
    row = prices[(prices["token_addr"]==token_addr) & (prices["date"]==day)]
    return row["price_usd"].values[0] if len(row) > 0 else np.nan

# Vectorized: merge on date
prices_daily = prices.copy()
prices_daily["date_only"] = prices_daily["date"].dt.date

cp_df["date_only"] = cp_df["dt"].dt.date
cp_df["date_str"]  = cp_df["date_only"].astype(str)

# Merge sell token price
p_sell = prices[["token_addr","date","price_usd"]].copy()
p_sell["date_only"] = p_sell["date"].dt.date.astype(str)
p_sell = p_sell.rename(columns={"price_usd":"price_sell","token_addr":"token_sell"})

p_buy = prices[["token_addr","date","price_usd"]].copy()
p_buy["date_only"] = p_buy["date"].dt.date.astype(str)
p_buy = p_buy.rename(columns={"price_usd":"price_buy","token_addr":"token_buy"})

# Also handle ETH (use WETH price as proxy)
weth_prices = prices[prices["token_addr"]=="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"].copy()
weth_prices["token_addr"] = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
prices_with_eth = pd.concat([prices, weth_prices])

p_sell2 = prices_with_eth[["token_addr","date","price_usd"]].copy()
p_sell2["date_only"] = p_sell2["date"].dt.date.astype(str)
p_sell2 = p_sell2.rename(columns={"price_usd":"price_sell","token_addr":"token_sell"})

p_buy2 = prices_with_eth[["token_addr","date","price_usd"]].copy()
p_buy2["date_only"] = p_buy2["date"].dt.date.astype(str)
p_buy2 = p_buy2.rename(columns={"price_usd":"price_buy","token_addr":"token_buy"})

cp_df = (cp_df
    .merge(p_sell2[["token_sell","date_only","price_sell"]], on=["token_sell","date_only"], how="left")
    .merge(p_buy2[["token_buy","date_only","price_buy"]],   on=["token_buy","date_only"],  how="left"))

# DEX reference rate: price_sell / price_buy (USD per sell token / USD per buy token)
# This gives: how many buy tokens per sell token at market rate
cp_df["dex_rate"] = cp_df["price_sell"] / cp_df["price_buy"]

# CoW improvement bps
cp_df["improvement_bps"] = np.where(
    (cp_df["dex_rate"] > 0) & cp_df["dex_rate"].notna(),
    1e4 * (cp_df["cow_rate"] - cp_df["dex_rate"]) / cp_df["dex_rate"],
    np.nan)

valid = cp_df["improvement_bps"].notna()
print(f"\n  Valid comparisons: {valid.sum():,} ({valid.mean():.1%})")
print(f"  Median improvement: {cp_df.loc[valid,'improvement_bps'].median():.1f} bps")
print(f"  Share where CoW beats DEX: {(cp_df.loc[valid,'improvement_bps']>0).mean():.1%}")

# ── 3. Merge fragility ────────────────────────────────────────
frag = pd.read_parquet("data/processed/auctions_full_usd.parquet",
    columns=["auction_id","fragility"])
cp_df = cp_df.merge(frag, on="auction_id", how="left")

# ── 4. By fragility bucket ────────────────────────────────────
def winsorize(s, lo=0.01, hi=0.99):
    l,h = s.quantile([lo,hi]); return s.clip(l,h)

sub = cp_df[cp_df["improvement_bps"].notna() & cp_df["fragility"].notna()].copy()
sub["improvement_bps_w"] = winsorize(sub["improvement_bps"])
sub["frag_bucket"] = pd.cut(sub["fragility"],
    bins=[0, 0.1, 0.3, 0.5, 0.75, 1.01],
    labels=["0–0.1\n(very low)","0.1–0.3\n(low)","0.3–0.5\n(mid)",
            "0.5–0.75\n(high)","0.75–1.0\n(very high)"])

agg = sub.groupby("frag_bucket", observed=True).agg(
    n=("auction_id","count"),
    median_improvement_bps=("improvement_bps_w","median"),
    mean_improvement_bps=("improvement_bps_w","mean"),
    share_beaten_by_router=("improvement_bps",lambda x:(x<0).mean()),
    mean_fragility=("fragility","mean")).reset_index()

print("\n=== Router benchmark by fragility ===")
print(agg.to_string(index=False))
agg.to_csv(OUT/"router_benchmark.csv", index=False)

# Regression: improvement_bps ~ high_fragility + pair_FE
import statsmodels.formula.api as smf
sub["pair"] = sub["symbol_sell"] + "/" + sub["symbol_buy"]
sub["high_frag"] = (sub["fragility"] > 0.5).astype(int)
sub["imp_w"] = sub["improvement_bps_w"]
m = smf.ols("imp_w ~ high_frag + C(pair)", data=sub).fit(cov_type="HC3")
coef_hf = m.params.get("high_frag", np.nan)
pval_hf = m.pvalues.get("high_frag", np.nan)
print(f"\nRegression: improvement_bps ~ high_fragility + pair_FE")
print(f"  high_frag coef = {coef_hf:.2f}  p = {pval_hf:.4f}  R² = {m.rsquared:.4f}  N = {int(m.nobs):,}")

# LaTeX
tex = f"""\\begin{{table}}[t]
\\centering\\footnotesize
\\caption{{Blue-chip router benchmark. CoW Protocol execution compared with
contemporaneous DeFiLlama daily DEX reference price.
Positive improvement = CoW execution better than public DEX.
Sample: blue-chip token pairs (ETH/WETH, USDC, USDT, DAI, WBTC) with
valid clearing prices. DeFiLlama daily price is an approximate (not
block-exact) DEX reference; caption acknowledges this limitation.}}
\\label{{tab:router-benchmark}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Fragility bucket & $N$ & Med.\ CoW improvement (bps) & Share beaten by DEX & Mean fragility \\\\
\\midrule
"""
for _, r in agg.iterrows():
    tex += (f"{str(r['frag_bucket']).replace(chr(10),' ')} & "
            f"{int(r['n']):,} & "
            f"{r['median_improvement_bps']:.1f} & "
            f"{r['share_beaten_by_router']*100:.1f}\\% & "
            f"{r['mean_fragility']:.3f} \\\\\n")
tex += f"""\\midrule
\\multicolumn{{5}}{{l}}{{Regression: CoW improvement $\\sim$ high fragility + pair FE: """
tex += f"$\\hat\\beta_{{\\text{{high frag}}}} = {coef_hf:.2f}$, $p = {pval_hf:.4f}$"
tex += r"""}} \\
\bottomrule
\end{tabular}
\footnotesize Reference price is DeFiLlama daily (not block-exact); directional comparison only.
\end{table}
"""
(TEX/"router_benchmark.tex").write_text(tex)
print(f"\n✓ router_benchmark.csv + router_benchmark.tex")
