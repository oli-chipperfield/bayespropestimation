#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest

from bayespropestimation.bayespropestimation import BayesProportionsEstimation
from bayespropestimation.bayesprophelpers import _calculate_kde, _calculate_map
from bayespropestimation.bayespropplotters import (
    _get_centre_lines,
    _get_intervals,
    _make_area_go,
    _make_delta_line,
    _make_density_go,
    _make_histogram_go,
    _make_line_go,
)


def compare_dictionaries(p, z):
    k = list(z.keys())
    r = []
    for i in k:
        r.append(np.any(p[i], z[i]))
    return np.any(r)


# Define Initialisation fixtures


@pytest.fixture
def make_a_list():
    return [10, 50]


@pytest.fixture
def make_a_numpy():
    return np.array([10, 50])


@pytest.fixture
def make_a_series():
    return pd.DataFrame({"a": [10, 50]})["a"]


@pytest.fixture
def make_b_list():
    return [20, 50]


@pytest.fixture
def make_b_numpy():
    return np.array([20, 50])


@pytest.fixture
def make_b_series():
    return pd.DataFrame({"a": [20, 50]})["a"]


@pytest.fixture
def make_explicit_prior():
    return 0.1


@pytest.fixture
def make_explicit_n():
    return 3


@pytest.fixture
def make_explicit_seed():
    return 1000


# Define bad inputs


@pytest.fixture
def make_a_str():
    return "foo"


@pytest.fixture
def make_a_bad_proportion():
    return [50, 10]


@pytest.fixture
def make_invalid_prior():
    return -1


@pytest.fixture
def make_invalid_n():
    return -1


@pytest.fixture
def make_invalid_seed():
    return 1.2


# Define fixtures for results testing


@pytest.fixture
def make_quantile_summary_results():
    return np.array(
        [
            [0.10594123533474246, 0.20127340652543865, 0.32294649108461715],
            [0.2744412955144984, 0.3999900504080393, 0.5396945573617622],
            [0.019031409978063265, 0.1974645057248544, 0.3708045045911511],
        ]
    )


@pytest.fixture
def make_get_posterior_results():
    return np.array(
        [
            [0.15608103, 0.17790474, 0.24626163],
            [0.35279345, 0.3636633, 0.51329111],
            [0.19671241, 0.18575856, 0.26702948],
        ]
    )


@pytest.fixture
def make_hdi_summary_results():
    return np.array(
        [
            [0.1038032606788284, 0.19275171884176634, 0.320526260831967],
            [0.2691767369659035, 0.3986262017019868, 0.5338200338157294],
            [0.0242652248218867, 0.19609368609983302, 0.3745174565063844],
        ]
    )


# Define fixtures for delta inference testing


@pytest.fixture
def make_infer_delta_probability_result():
    return (0.9863, "almost certainly")


@pytest.fixture
def make_infer_delta_bayes_factor_result():
    return (71.99270072992677, "very strong")


# Define fixtures for bayeprophelpers testing


@pytest.fixture
def make_draw():
    np.random.seed(1000)
    return np.random.beta(2, 5, 100)


@pytest.fixture
def _calculate_kde_results():
    return (
        np.array([0.03857532, 0.43082818, 0.82308105]),
        np.array([1.14415602, 1.12407264, 0.11679198]),
    )


@pytest.fixture
def _calculate_map_results():
    return 0.038575315797957296


# Run helper tests


def test__calculate_kde_returns_correct_values(make_draw, _calculate_kde_results):
    x, kde_density = _calculate_kde(make_draw, num=3)
    assert np.allclose(x, _calculate_kde_results[0])
    assert np.allclose(kde_density, _calculate_kde_results[1])


def test__calculate_map_results(make_draw, _calculate_map_results):
    assert np.isclose(_calculate_map_results, _calculate_map(make_draw, num=3))


# Run initialisation tests


def test_BayesProportionsEstimation_with_make_a_list_and_make_b_list_initialises(
    make_a_list, make_b_list
):
    try:
        BayesProportionsEstimation(a=make_a_list, b=make_b_list)
    except:
        raise pytest.fail()


def test_BayesProportionsEstimation_with_make_a_numpy_and_make_b_numpy_initialises(
    make_a_numpy, make_b_numpy
):
    try:
        BayesProportionsEstimation(a=make_a_numpy, b=make_b_numpy)
    except:
        raise pytest.fail()


def test_BayesProportionsEstimation_with_make_a_series_and_make_b_series_initialises(
    make_a_series, make_b_series
):
    try:
        BayesProportionsEstimation(a=make_a_series, b=make_b_series)
    except:
        raise pytest.fail()


def test_BayesProportionsEstimation_with_make_explicit_prior_alpha_and_make_explicit_prior_beta_intialises(
    make_a_list, make_b_list, make_explicit_prior
):
    try:
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, prior_alpha=make_explicit_prior
        )
    except:
        raise pytest.fail()


def test_BayesProportionsEstimation_with_mmake_explicit_n_initialises(
    make_a_list, make_b_list, make_explicit_n
):
    try:
        BayesProportionsEstimation(a=make_a_list, b=make_b_list, n=make_explicit_n)
    except:
        raise pytest.fail()


def test_BayesProportionsEstimation_with_mmake_explicit_seed_initialises(
    make_a_list, make_b_list, make_explicit_seed
):
    try:
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, seed=make_explicit_seed
        )
    except:
        raise pytest.fail()


# Run ValueError tests


def test_BayesProportionsEstimation_with_a_str_returns_ValueError(
    make_a_str, make_b_list
):
    with pytest.raises(ValueError) as e:
        BayesProportionsEstimation(a=make_a_str, b=make_b_list)
    assert (
        str(e.value)
        == "type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'DataFrame'"
    )


def test_BayesProportionsEstimation_with_a_bad_proportion_returns_ValueError(
    make_a_bad_proportion, make_b_list
):
    with pytest.raises(ValueError) as e:
        BayesProportionsEstimation(a=make_a_bad_proportion, b=make_b_list)
    assert (
        str(e.value)
        == "the count of successes for a and/or b exceeds the number of trials"
    )


def test_BayesProportionsEstimation_with_make_invalid_prior_returns_ValueError(
    make_a_list, make_b_list, make_invalid_prior
):
    with pytest.raises(ValueError) as e:
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, prior_alpha=make_invalid_prior
        )
    assert str(e.value) == "the prior_alpha and/or prior_beta parameters must be > 0"


def test_BayesProportionsEstimation_with_make_invalid_n_returns_ValueError(
    make_a_list, make_b_list, make_invalid_n
):
    with pytest.raises(ValueError) as e:
        BayesProportionsEstimation(a=make_a_list, b=make_b_list, n=make_invalid_n)
    assert str(e.value) == "n must be a positive integer"


def test_BayesProportionsEstimation_with_make_invalid_seed_returns_ValueError(
    make_a_list, make_b_list, make_invalid_seed
):
    with pytest.raises(ValueError) as e:
        BayesProportionsEstimation(a=make_a_list, b=make_b_list, seed=make_invalid_seed)
    assert str(e.value) == "seed must be a positive integer or None"


# Run results tests


def test_BayesProportionsEstimation_quantile_summary_returns_correct_results(
    make_a_list, make_b_list, make_explicit_seed, make_quantile_summary_results
):
    test = np.array(
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, seed=make_explicit_seed
        ).quantile_summary()
    )[:, 0:3]
    assert np.array_equal(test, make_quantile_summary_results)


def test_BayesProportionsEstimation_hdi_summary_returns_correct_results(
    make_a_list,
    make_b_list,
    make_explicit_seed,
    make_quantile_summary_results,
    make_hdi_summary_results,
):
    test = np.array(
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, seed=make_explicit_seed
        ).hdi_summary()
    )[:, 0:3]
    assert np.array_equal(test, make_hdi_summary_results)


def test_BayesProportionsEstimation_get_posteriors_returns_correct_results(
    make_a_list,
    make_b_list,
    make_explicit_n,
    make_explicit_seed,
    make_get_posterior_results,
):
    test = BayesProportionsEstimation(
        make_a_list, make_b_list, n=make_explicit_n, seed=make_explicit_seed
    ).get_posteriors()
    test = np.array(pd.DataFrame(test))
    assert np.allclose(test, make_get_posterior_results)


# Run delta inference tests


def test_infer_delta_probability_returns_correct_values(
    make_a_list, make_b_list, make_explicit_seed, make_infer_delta_probability_result
):
    p, i = BayesProportionsEstimation(
        a=make_a_list, b=make_b_list, seed=make_explicit_seed
    ).infer_delta_probability()
    assert np.isclose(p, make_infer_delta_probability_result[0])
    assert i == make_infer_delta_probability_result[1]


def test_infer_delta_bayes_factor_returns_correct_values(
    make_a_list, make_b_list, make_explicit_seed, make_infer_delta_bayes_factor_result
):
    bf, i = BayesProportionsEstimation(
        a=make_a_list, b=make_b_list, seed=make_explicit_seed
    ).infer_delta_bayes_factor()
    assert np.isclose(bf, make_infer_delta_bayes_factor_result[0])
    assert i == make_infer_delta_bayes_factor_result[1]


# RUn bayespropplotters tests


def test__get_centre_lines_runs_without_error(make_draw):
    try:
        _get_centre_lines(make_draw, method="hdi")
    except:
        raise pytest.fail()


def test__get_intervals_runs_with_hdi_without_error(make_draw):
    try:
        _get_intervals(make_draw, method="hdi", bounds=0.95)
    except:
        raise pytest.fail()


def test__get_intervals_runs_with_quantile_without_error(make_draw):
    try:
        _get_intervals(make_draw, method="quantile", bounds=[0.025, 0.975])
    except:
        raise pytest.fail()


def test__make_density_go_without_error(make_draw):
    try:
        _make_density_go(make_draw, name="dummy")
    except:
        raise pytest.fail()


def test__make_histogram_go_without_error(make_draw):
    try:
        _make_histogram_go(make_draw, name="dummy")
    except:
        raise pytest.fail()


def test__make_line_go_with_without_error(make_draw):
    try:
        cl = _get_centre_lines(make_draw, method="hdi")
        _make_line_go(cl, name="dummy")
    except:
        raise pytest.fail()


def test__make_area_go_without_error(make_draw):
    try:
        intervals = _get_intervals(make_draw, method="hdi", bounds=0.95)
        _make_area_go(intervals, name="dummy")
    except:
        raise pytest.fail()


# Run plot_posterior method test


def test_plot_posterior_with_error(make_a_list, make_b_list, make_explicit_seed):
    try:
        BayesProportionsEstimation(
            a=make_a_list, b=make_b_list, seed=make_explicit_seed
        ).posterior_plot()
    except:
        raise pytest.fail()
