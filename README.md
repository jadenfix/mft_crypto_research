## Mid-Frequency-Trading: 1 Minute Crypto Data
 ** This Repo is Dedicated to the research of algorithms and statistical methods to predict crypto currency prices **

## Primary focuses: ML, DL, Physics, Stochastic Methods
---

** Strategies Tested: **

-Convolutionary Neural Networks:

-Recurrent Neural Networks

-Physics Informed Neural Networks



## Explicit research data boundary

The GPU notebook previously replaced missing private market CSVs with random dummy data automatically. It now raises `FileNotFoundError` by default. Only `MFT_ALLOW_SYNTHETIC=1` permits the computational-smoke fixture, which writes a `.synthetic.json` provenance sidecar next to each generated CSV. Results from that mode must remain labeled synthetic and cannot be used as market-performance evidence.

```sh
python -m pip install -r requirements-smoke.txt
python -m unittest discover -s tests -v
```

Three credential-free tests execute the notebook's actual generator, verify the missing-data rejection and supplied-file behavior, and run the explicitly synthetic CSV through a tiny RandomForest fit/predict flow. The real training/backtest CSVs are not bundled, and full GPU/DL/backtesting experiments remain unverified.

Cached notebook outputs were cleared after changing the data boundary; rerun explicitly rather than treating prior displayed outputs as validation of the revised workflow.
