from simulator.build_panel import build_sim_panel

# Compose the simulation panel with guarded forward-fills and explicit ret_fwd
panel, state_cols = build_sim_panel(
    market_df=market,
    spx_clean_dir=SPX_CLEAN,
    include_spy=False,
    ffill_limit=2, # IV features forward-fill at most 2 calendar days
)
