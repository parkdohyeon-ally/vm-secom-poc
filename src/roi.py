"""Business case — estimate $ impact of deploying Virtual Metrology.

VM catches likely-bad wafers early (rework/scrap avoidance + less full
metrology). This converts model recall into a monthly dollar figure.
"""


def roi(wafers_per_month, wafer_value_usd, baseline_scrap_rate,
        detection_recall, vm_coverage=1.0, inspection_cost_per_wafer=0.0):
    """Return monthly impact estimate.

    wafers_per_month        : throughput
    wafer_value_usd         : value (scrap/rework cost avoided) per caught wafer
    baseline_scrap_rate     : current fail fraction (e.g. 0.066)
    detection_recall        : model recall at chosen threshold (fails caught)
    vm_coverage             : fraction of line covered by VM
    inspection_cost_per_wafer: cost of running VM per wafer (compute, ops)
    """
    fails = wafers_per_month * baseline_scrap_rate
    caught = fails * detection_recall * vm_coverage
    gross = caught * wafer_value_usd
    cost = wafers_per_month * inspection_cost_per_wafer
    net = gross - cost
    return {
        "fails_per_month": round(fails),
        "caught_per_month": round(caught),
        "gross_saved_usd": round(gross),
        "vm_run_cost_usd": round(cost),
        "net_saved_usd": round(net),
        "net_saved_usd_per_year": round(net * 12),
    }


if __name__ == "__main__":
    # Example: 50k wafers/mo, $400 avoided per caught wafer, 6.6% scrap,
    # model recall 0.40 at the chosen operating threshold.
    r = roi(wafers_per_month=50_000, wafer_value_usd=400,
            baseline_scrap_rate=0.066, detection_recall=0.40,
            inspection_cost_per_wafer=0.02)
    for k, v in r.items():
        print(f"{k:28s}: {v:>12,}")
