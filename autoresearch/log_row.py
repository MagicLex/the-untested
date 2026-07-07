"""Log one experiment row to the autoresearch leaderboard FG.

The FG needs a real timestamp and the CLI cannot write one (recipe quirk), so
insert through the SDK. Creates the FG on first call.

    python3 autoresearch/log_row.py <commit> <val_metric> <peak_gb> <status> <desc> <iso_ts>
"""
import sys

import hopsworks
import pandas as pd

FG = "autoresearch_experiments_amr"


def main():
    commit, val, mem, status, desc, ts = sys.argv[1:7]
    fs = hopsworks.login().get_feature_store()
    fg = fs.get_or_create_feature_group(
        FG, version=1,
        description="Autoresearch leaderboard for the AMR QSAR model search. One "
                    "row per experiment (keep/discard/crash); val_metric = mean "
                    "AMR ROC-AUC, maximize.",
        primary_key=["commit"], event_time="ts",
        online_enabled=False, statistics_config=False)
    fg.insert(pd.DataFrame([{
        "commit": commit, "val_metric": float(val), "peak_memory_gb": float(mem),
        "status": status, "description": desc[:500],
        "ts": pd.to_datetime(ts, utc=True)}]), wait=True)
    print(f"logged {commit} val={val} status={status}", flush=True)


if __name__ == "__main__":
    main()
