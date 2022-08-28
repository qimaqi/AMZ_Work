#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Thierry Backes <tbackes@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


def aggregate_statistics(data_by_matching_distance, bucket_size, plot: bool) -> None:
    """
    Aggregates statistics over multiple frames and groups them by matching distance
    Output is precision, recall, F1 score in stdout and optional plot
    """
    #pylint: disable=too-many-statements,too-many-locals
    
    legend_lines = []
    pr_curve_data = []
    pr_curve_data_2 = []

    print(f'Statistics for {bucket_size[0]}m - {bucket_size[1]}m :')

    counter = 0
    for distance, frame_data in data_by_matching_distance.items():
        agg_tpr = []  # True positive rate/sensitivity/recall: "true positives"/"GTMD cones"
        aggr_fnr = []  # False negative rate: "FN/(FN+TP)" = "(GTMD WITHOUT MATCH)/(TOTAL GTMD)"

        agg_pr = []  # Precision/positive predicted value: "TP/(TP+FP)" = "(MATCH WITH CORRECT COLOR)/(TP+FP)" where
        # FP = (GTMD with multiple matches) + (cone with no match) + (cone match with wrong color)

        all_tps = []
        all_fps = []
        all_fns = []
        all_errors_x = []
        all_errors_y = []

        for _, data in frame_data.items():
            # Extract data
            gtmd_cones_dict = data['gtmd_cones_dict']
            total_gtmd = len(gtmd_cones_dict['pos_with_color'])
            false_positive = data['unmatched_cones'] + data['cone_match_wrong_color']  + data['gtmd_multiple_match']
            true_positive = data['correct_matches']
            false_negative = data['gtmd_no_match']
            error_x = data['offset_x']
            error_y = data['offset_y']

            all_tps.append(true_positive)
            all_fps.append(false_positive)
            all_fns.append(false_negative)
            all_errors_x.extend(error_x)
            all_errors_y.extend(error_y)

            # Compute true positive rate/sensitivity/recall
            if true_positive + false_negative == 0:
                agg_tpr.append(0)
            else:
                agg_tpr.append(true_positive / (true_positive + false_negative))

            # Compute false negative rate
            if total_gtmd == 0:
                aggr_fnr.append(0)
            else:
                aggr_fnr.append(false_negative / total_gtmd)

            # compute precision
            if (true_positive + false_positive) == 0:
                agg_pr.append(0)
            else:
                agg_pr.append(true_positive / (true_positive + false_positive))

        pr_curve_data.append([np.mean(agg_pr), np.mean(agg_tpr)])

        tp_tmp = np.mean(all_tps)
        fn_tmp = np.mean(all_fns)
        fp_tmp = np.mean(all_fps)
        error_np = np.array(all_errors_x)**2 + np.array(all_errors_y)**2 
        rms_err = np.sqrt(np.mean((error_np)))

        pr_curve_data_2.append([np.mean(tp_tmp / (tp_tmp + fp_tmp)), np.mean(tp_tmp / (tp_tmp + fn_tmp))])

        counter += 1
        print(f"\t{distance}m")
        print(f"\t\tF1 score\t:{2*(np.mean(agg_tpr)*np.mean(agg_pr))/(np.mean(agg_tpr)+np.mean(agg_pr)):.2f}")
        print(f"\t\tPrecision\t:{np.mean(agg_pr):.2f}")
        print(f"\t\tRecall\t\t:{np.mean(agg_tpr):.2f}")
        print(f"\t\tRMS error\t\t:{rms_err:.2f}")

