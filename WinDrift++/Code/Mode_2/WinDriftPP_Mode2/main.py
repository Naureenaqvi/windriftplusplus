"""
main.py — WinDrift++ Mode 2 Pipeline
======================================
Full dataset suite: C1-C30
  C1-C17  : Synthetic univariate
  C18-C22 : Real-world univariate
  C23-C25 : Real-world multivariate
  C26-C30 : Synthetic multivariate

Pipeline:
  1. Generate all datasets               [skipped if REGENERATE_DATA=False]
  2. Run WinDrift++ detection (Mode I + Mode II)
  3. Compare 5 voting methods (univariate C1-C22)
  4. Summarise detection statistics
  5. Generate visualisations             [skipped if GENERATE_PLOTS=False]
"""


# **** Shaping up provided code with parameters and correct indentation ****
import os, sys, time, warnings
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "core_algorithm"))
sys.path.insert(0, os.path.join(ROOT, "data_generation"))
sys.path.insert(0, os.path.join(ROOT, "visualisation"))

from config import (
    WIN_LEV_SIZE, MAX_CYC_LEN, ALPHA, DATASET_REGISTRY,
    GROUPS, UNIVARIATE_IDS, MULTIVARIATE_IDS,
    DATASET_DIR, RESULTS_DIR, TABLES_DIR, LOGS_DIR, PLOTS_DIR,
)
from data_generator              import generate_all_datasets
from data_generator_real         import generate_real_univariate
from data_generator_multivariate import generate_multivariate, load_multivariate
from windrift_pp                 import WinDriftPP
from windrift_pp_multivariate    import WinDriftPPMultivariate
from voting                      import (majority_vote, plurality_vote, borda_count_vote,
                                         plurality_elimination_vote, pairwise_vote)
from evaluation                  import summarise_dataset
import visualisation as vis

# **** Pipeline toggles ****
REGENERATE_DATA = False   # True = regenerate all CSVs from scratch
GENERATE_PLOTS  = False   # True = produce all charts after detection

for d in [DATASET_DIR, TABLES_DIR, LOGS_DIR, PLOTS_DIR,
          os.path.join(TABLES_DIR, "per_dataset")]:
    os.makedirs(d, exist_ok=True)


def run():
    t0 = time.time()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 65)
    log("  WinDrift++ Mode 2 - Full Pipeline (C1-C30)")
    log("=" * 65)

    # **** Step 1: Data ****
    log("\n[1] Datasets")
    if REGENERATE_DATA:
        log("  -> Synthetic univariate (C1-C17)...")
        uni_syn, _ = generate_all_datasets(save=True)
        log("  -> Real-world univariate (C18-C22)...")
        uni_real   = generate_real_univariate(save=True)
        log("  -> Multivariate (C23-C30)...")
        multi      = generate_multivariate(save=True)
    else:
        uni_syn  = {}
        uni_real = {}
        for cid in UNIVARIATE_IDS:
            path = os.path.join(DATASET_DIR, f"{cid}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Set REGENERATE_DATA=True to generate.")
            (uni_syn if int(cid[1:]) <= 17 else uni_real)[cid] = pd.read_csv(path)
        multi = {cid: load_multivariate(cid, DATASET_DIR)
                 for cid in MULTIVARIATE_IDS}
        log(f"  + Loaded {len(UNIVARIATE_IDS)} univariate + "
            f"{len(MULTIVARIATE_IDS)} multivariate datasets")

    all_univariate = {**uni_syn, **uni_real}

    # **** Step 2: Detection ****
    log("\n[2] WinDrift++ detection")
    all_results = {}

    for cid in sorted(UNIVARIATE_IDS, key=lambda x: int(x[1:])):
        detector = WinDriftPP(WIN_LEV_SIZE, MAX_CYC_LEN, ALPHA, verbose=False)
        detector.detect(all_univariate[cid], dataset_id=cid)
        df      = detector.to_dataframe()
        all_results[cid] = df
        m2      = df[df["mode"] == 2]
        flagged = (m2["majority_flag"] == 1).sum()
        total   = len(m2)
        log(f"  + {cid:4s}  Mode II: {flagged}/{total} comparisons flagged")
        df.to_csv(os.path.join(TABLES_DIR, "per_dataset", f"{cid}_results.csv"),
                  index=False)

    for cid in sorted(MULTIVARIATE_IDS, key=lambda x: int(x[1:])):
        detector = WinDriftPPMultivariate(WIN_LEV_SIZE, MAX_CYC_LEN, ALPHA, verbose=False)
        df       = detector.detect(multi[cid], dataset_id=cid)
        all_results[cid] = df
        m2      = df[df["mode"] == 2]
        flagged = (m2["majority_flag"] == 1).sum()
        total   = len(m2)
        nf      = DATASET_REGISTRY[cid]["n_features"]
        log(f"  + {cid:4s}  Mode II: {flagged}/{total} comparisons flagged  ({nf} features)")
        save_cols = [c for c in df.columns if not c.startswith("flag_")]
        df[save_cols].to_csv(
            os.path.join(TABLES_DIR, "per_dataset", f"{cid}_results.csv"), index=False)

    # **** Step 3: Voting comparison (univariate only) ****
    log("\n[3] Voting methods comparison (C1-C22)")
    from statistical_tests import TestResult
    voting_rows = []
    for cid in sorted(UNIVARIATE_IDS, key=lambda x: int(x[1:])):
        m2    = all_results[cid][all_results[cid]["mode"] == 2]
        total = len(m2)
        for method_name, vote_fn in [
            ("majority", majority_vote), ("plurality", plurality_vote),
            ("borda", borda_count_vote), ("elimination", plurality_elimination_vote),
            ("pairwise", pairwise_vote),
        ]:
            flags = []
            for _, row in m2.iterrows():
                trs = [
                    TestResult("KS",     row.get("ks_stat",0),     row.get("ks_crit",1),     row.get("ks_pval",1),     row.get("ks_flag",0)),
                    TestResult("Kuiper", row.get("kuiper_stat",0), row.get("kuiper_crit",1), row.get("kuiper_pval",1), row.get("kuiper_flag",0)),
                    TestResult("CVM",    row.get("cvm_stat",0),    row.get("cvm_crit",1),    row.get("cvm_pval",1),    row.get("cvm_flag",0)),
                    TestResult("AD",     row.get("ad_stat",0),     row.get("ad_crit",1),     row.get("ad_pval",1),     row.get("ad_flag",0)),
                    TestResult("EMD",    row.get("emd_stat",0),    row.get("emd_crit",1),    row.get("emd_pval",1),    row.get("emd_flag",0)),
                ]
                flags.append(vote_fn(trs).drift_flag)
            flagged = sum(1 for f in flags if f == 1)
            voting_rows.append({
                "dataset_id": cid,
                "method":     method_name,
                "flagged":    flagged,
                "total":      total,
                "flag_rate":  round(flagged / total, 4) if total else 0,
            })

    voting_df = pd.DataFrame(voting_rows)
    pivot = voting_df.pivot_table(
        index="dataset_id", columns="method", values="flag_rate").round(4)
    pivot.to_csv(os.path.join(TABLES_DIR, "voting_methods_comparison.csv"))
    log("  + Saved voting_methods_comparison.csv")

    # **** Step 4: Detection statistics ****
    log("\n[4] Detection statistics")
    rows = []
    for cid in sorted(all_results.keys(), key=lambda x: int(x[1:])):
        meta = DATASET_REGISTRY[cid]
        try:
            m = summarise_dataset(cid, all_results[cid])
            rows.append({"group": meta["group"], "label": meta["label"],
                         "multivariate": meta["multivariate"],
                         "n_features": meta["n_features"], **m})
        except Exception as e:
            rows.append({"dataset_id": cid, "group": meta["group"],
                         "label": meta["label"], "error": str(e)})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(TABLES_DIR, "detection_summary.csv"), index=False)
    log(f"  + Saved detection_summary.csv ({len(rows)} datasets)")

    # **** Step 5: Visualisations ****
    if GENERATE_PLOTS:
        log("\n[5] Visualisations")
        _run_plots(all_univariate, all_results, voting_df, summary_df)
    else:
        log("\n[5] Visualisations skipped  (set GENERATE_PLOTS=True to enable)")

    elapsed = time.time() - t0
    log(f"\n{'='*65}")
    log(f"  Mode 2 complete in {elapsed:.1f}s  |  30 datasets  |  Results -> {RESULTS_DIR}")
    log(f"{'='*65}\n")

    log("\nDetection summary by group (Mode II):")
    for group_name, cids in GROUPS.items():
        log(f"  {group_name}")
        for cid in cids:
            if cid not in all_results:
                continue
            m2      = all_results[cid][all_results[cid]["mode"] == 2]
            flagged = (m2["majority_flag"] == 1).sum()
            total   = len(m2)
            log(f"    {cid:<6}  {flagged}/{total} flagged")

    with open(os.path.join(LOGS_DIR, "run_log_mode2.txt"), "w") as f:
        f.write("\n".join(log_lines))

    return all_results, summary_df, voting_df


def _run_plots(all_univariate, all_results, voting_df, summary_df):
    meta     = DATASET_REGISTRY
    uni_only = {k: v for k, v in all_results.items()
                if not meta[k]["multivariate"]}
    for label, fn, args in [
        ("Dataset overview",   vis.plot_dataset_means,
         ({k: all_univariate[k] for k in [f"C{i}" for i in range(1,11)]}, meta)),
        ("ECDF comparisons",   vis.plot_ecdf_comparisons,     (all_univariate,)),
        ("C11-C17 dists",      vis.plot_c11_c17_distributions, (all_univariate,)),
        ("Drift heatmap",      vis.plot_drift_heatmap,         (uni_only, meta)),
        ("Test contributions", vis.plot_test_contributions,    (uni_only,)),
        ("Detection rates",    vis.plot_performance_comparison, (summary_df,)),
        ("Voting comparison",  vis.plot_voting_comparison,     (voting_df,)),
    ]:
        try:
            fn(*args)
            print(f"  + {label}")
        except Exception as e:
            print(f"  ! {label}: {e}")


if __name__ == "__main__":
    run()
