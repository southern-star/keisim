"""G3 metrics on common and native display grids. Only the scorer sees ground truth or world geometry."""
import math

import cv2
import numpy as np

from keisim.config import N_SEM, SEM
from .metrics import local_ssim


def sample_region(x):
    return x[1::2,1::2]


def region_masks(support):
    return dict(all=np.ones(support.shape,bool),visible=support==1,past=support==2,unconfirmed=support==0,unknown=support==255)


def confusion(truth, predicted, mask=None):
    pair=truth.astype(np.int32)*N_SEM+predicted
    if mask is not None:
        pair=pair[mask]
    return np.bincount(pair.ravel(),minlength=N_SEM*N_SEM).reshape(N_SEM,N_SEM)


def measure(values):
    return float(values.sum(dtype=np.float64)),values.size


def region_scores(low, gt, predicted_seg, seg, masks):
    diff=low-gt
    l1=np.abs(diff).mean(-1)
    mse=np.square(diff).mean(-1)
    sim=local_ssim(low,gt)
    result={}
    for name,mask in masks.items():
        for metric,values in (("l1",l1),("mse",mse),("ssim",sim)):
            result[name+"_"+metric]=measure(values[mask])
    static=(seg>=1)&(seg<=7)
    dynamic=seg>=SEM["vehicle"]
    predicted=predicted_seg>=SEM["vehicle"]
    result["phantom_fpr"]=measure(predicted[static])
    result["unconfirmed_phantom_fpr"]=measure(predicted[static&masks["unconfirmed"]])
    for name in ("visible","unconfirmed","all"):
        result[name+"_dynamic_miss"]=measure((~predicted)[dynamic&masks[name]])
    for name in ("vehicle","pedestrian","road","tl_red","tl_green","marking"):
        mask=(seg==SEM[name])&masks["visible"]
        result["visible_"+name+"_l1"]=measure(l1[mask])
    return result


def projected_scores(frame, previous, warp):
    (overlap,old_overlap),(vacated,old_vacated)=warp
    error,truth,label=frame
    old_error,old_truth,_=previous
    a=error.reshape(-1,3)
    b=old_error.reshape(-1,3)
    ghost_label=label.ravel()[vacated]>=SEM["vehicle"]
    current_color=truth.reshape(-1,3)[vacated]
    old_color=old_truth.reshape(-1,3)[old_vacated]
    output=current_color+a[vacated]
    distinct=np.abs(current_color-old_color).mean(-1)>.10
    resembles_old=np.abs(output-old_color).mean(-1)+.02 < np.abs(output-current_color).mean(-1)
    return dict(warped_residual_l1=measure(np.abs(a[overlap]-b[old_overlap]).mean(-1)),
                vacated_rgb_l1=measure(np.abs(a[vacated]).mean(-1)),vacated_actor_rate=measure(ghost_label),
                vacated_old_color_rate=measure(resembles_old[distinct]))


def display_image(low):
    return cv2.resize(low,(256,128),interpolation=cv2.INTER_LINEAR)


def finalize(total):
    report=total.report()
    for region in ("all","visible","past","unconfirmed","unknown"):
        mse=report["metrics"].get(region+"_mse")
        report["metrics"][region+"_psnr_db"]=-10*math.log10(mse) if mse else None
    return report


def g3_decision(cfg, scores, profile, calibration, coverage):
    gate=cfg["gate"]
    result={}
    def value(group,mode,key):
        return scores[group][mode]["metrics"].get(key)
    for nfe in cfg["test_nfe"]:
        mode=f"nfe{nfe}"
        def ratio(group,key):
            x,y=value(group,mode,key),value(group,"phase2",key)
            return x/y if x is not None and y else None
        def delta(group,key):
            x,y=value(group,mode,key),value(group,"phase2",key)
            return x-y if x is not None and y is not None else None
        def within(number,threshold):
            return number is not None and number<=threshold
        l1=ratio("fixed","all_l1")
        ssim=delta("fixed","all_ssim")
        p=profile["methods"]
        speed=p[mode]["total_ms"]["p50"]/p["phase2"]["total_ms"]["p50"]
        first,last=ratio("quarter0","all_l1"),ratio("quarter3","all_l1")
        drift=last/first if last is not None and first else None
        benefit=dict(l1=within(l1,gate["quality_l1_gain_ratio"]),
                     ssim=ssim is not None and ssim>=gate["quality_ssim_gain"],
                     latency=within(speed,gate["compute_gain_p50_ratio"]))
        criteria=dict(full_rollouts=coverage,probe_calibration=calibration,
                      benefit=any(benefit.values()),common_l1=within(l1,gate["common_l1_noninferiority_ratio"]),
                      native_l1=within(ratio("fixed","native_l1"),gate["native_display_l1_ratio_max"]),
                      flicker=within(ratio("fixed","warped_residual_l1"),gate["static_warped_residual_ratio_max"]),
                      vacated=within(delta("fixed","vacated_rgb_l1"),gate["vacated_rgb_l1_increase_max"]),
                      phantom=within(delta("fixed","phantom_fpr"),gate["static_phantom_fpr_increase_max"]),
                      late_l1=within(last,gate["last_quarter_l1_ratio_max"]),
                      late_drift=within(drift,gate["relative_late_vs_early_drift_max"]),
                      latency=within(p[mode]["total_ms"]["p95"],gate["end_to_end_p95_ms_max"]))
        result[mode]=dict(status="PASS" if all(criteria.values()) else "FAIL",criteria=criteria,
                          benefit=benefit,common_l1_ratio=l1,ssim_gain=ssim,latency_p50_ratio=speed,drift=drift)
    return result
