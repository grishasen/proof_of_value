from typing import List

import numpy as np
import polars as pl
import scipy.stats.distributions as dist
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats.contingency import odds_ratio


def proportions_ztest(c1, c2, n1, n2, two_tailed=True) -> dict[str, float]:
    """
    Calculate the test statistic for a z-test on 2 proportions from independent samples
    c1, c2: number of successes in group 1 and 2
    n1, n2: total number of observations in group 1 and 2
    Returns: test statistic (z), and p-value
    """
    if (c1 == 0) | (c2 == 0) | (n1 == 0) | (n2 == 0):
        return {"z_score": 0.0, "z_p_val": 0.0}

    avg_p = (c1 + c2) / (n1 + n2)
    z_val = (c1 / n1 - c2 / n2) / np.sqrt(avg_p * (1 - avg_p) * (1 / n1 + 1 / n2))
    z_prob = dist.norm.cdf(-np.abs(z_val))

    if two_tailed:
        return {"z_score": z_val, "z_p_val": 2 * z_prob}

    else:
        return {"z_score": z_val, "z_p_val": z_prob}


def g_test_proportion(c1, c2, n1, n2) -> dict[str, float]:
    """
    Performs a g-test comparing the proportion of successes in two groups.
    Returns: test statistic, p value
    """
    if (c1 == 0) | (c2 == 0) | (n1 == 0) | (n2 == 0):
        return {"g_test_stat": 0.0, "g_p_val": 0.0}

    failures_1 = n1 - c1
    failures_2 = n2 - c2
    total = n1 + n2

    expected_control_success = n1 * (c1 + c2) / total
    expected_control_failure = n1 * (failures_1 + failures_2) / total
    expected_variant_success = n2 * (c1 + c2) / total
    expected_variant_failure = n2 * (failures_1 + failures_2) / total

    g1 = c1 * np.log(c1 / expected_control_success)
    g2 = failures_1 * np.log(failures_1 / expected_control_failure)
    g3 = c2 * np.log(c2 / expected_variant_success)
    g4 = failures_2 * np.log(failures_2 / expected_variant_failure)

    g_test_stat = 2 * (g1 + g2 + g3 + g4)
    p_value = 1 - chi2.cdf(g_test_stat, 1)

    return {"g_stat": g_test_stat, "g_p_val": p_value}


def chi2_test(args: List[pl.Series]) -> pl.Struct:
    df = pl.DataFrame(args)
    df = df.transpose(include_header=False, column_names="column_0")
    contingency_table = df.to_numpy()
    g, p, dof, expected = chi2_contingency(contingency_table, correction=False)
    odds_ratio_stat, odds_ratio_ci_low, odds_ratio_ci_high = 0.0, 0.0, 0.0
    if contingency_table.shape == (2, 2):
        res = odds_ratio(contingency_table, kind='sample')
        odds_ratio_stat = res.statistic
        odds_ratio_ci_low, odds_ratio_ci_high = res.confidence_interval(confidence_level=0.95)
    return {"chi2_stat": g, "chi2_dof": dof, "chi2_p_val": p,
            "chi2_odds_ratio_stat": odds_ratio_stat, "chi2_odds_ratio_ci_low": odds_ratio_ci_low,
            "chi2_odds_ratio_ci_high": odds_ratio_ci_high}


def g_test(args: List[pl.Series]) -> pl.Struct:
    df = pl.DataFrame(args)
    df = df.transpose(include_header=False, column_names="column_0")
    contingency_table = df.to_numpy()
    g, p, dof, expected = chi2_contingency(contingency_table, lambda_="log-likelihood", correction=False)
    odds_ratio_stat, odds_ratio_ci_low, odds_ratio_ci_high = 0.0, 0.0, 0.0
    if contingency_table.shape == (2, 2):
        res = odds_ratio(contingency_table, kind='sample')
        odds_ratio_stat = res.statistic
        odds_ratio_ci_low, odds_ratio_ci_high = res.confidence_interval(confidence_level=0.95)
    return {"g_stat": g, "g_dof": dof, "g_p_val": p,
            "g_odds_ratio_stat": odds_ratio_stat, "g_odds_ratio_ci_low": odds_ratio_ci_low,
            "g_odds_ratio_ci_high": odds_ratio_ci_high}


def z_test(args: List[pl.Series]) -> pl.Struct:
    df = pl.DataFrame(args)
    df = df.transpose(include_header=False, column_names="column_0")
    if df.shape == (2, 2):
        return proportions_ztest(df[0, 0], df[0, 1], df[1, 0], df[1, 1])
    else:
        return {"z_score": 0.0, "z_p_val": 0.0}
