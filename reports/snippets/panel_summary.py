# EDA excerpt (Notebook 01)
summary = (panel[["close_spy", "vix", "rv_21d", "hvol_30d", "rate_10y"]]
              .pct_change()
              .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
              .T[["mean", "std", "min", "1%", "5%", "50%", "95%", "99%", "max"]])
