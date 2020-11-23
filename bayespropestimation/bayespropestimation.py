import arviz as az
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from bayespropestimation.bayesprophelpers import _calculate_map
from bayespropestimation.bayespropplotters import (
    _get_centre_lines,
    _get_intervals,
    _make_area_go,
    _make_delta_line,
    _make_density_go,
    _make_histogram_go,
    _make_line_go,
)


class BayesProportionsEstimation:
    def __init__(self, a, b, prior_alpha=0.5, prior_beta=0.5, n=10000, seed=None):
        """
        Initialises the BayesProportionsEstimation class and samples from the posterior distribution
        Parameters
        ----------
        a: list, ndarray or Series [successes, trials]:  array describing results from sample a
        b: list, ndarray or Series [successes, trials]:  array describing results from sample b
        prior_alpha: float, alpha parameter for the Beta prior distribution, default = 0.5 (Jeffreys prior)
        prior_beta: float, beta parameter for the Beta prior distribution, default = 0.5 (Jeffreys prior)
        m: integer, number of samples to take from the posterior distribution, default = 10000
        seed: integer, set random seed at the start of the initialisation, default = None
        """
        self.a = a
        self.b = b
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.n = n
        self.seed = seed
        self._check_inputs()
        self._sample_posteriors()

    def _check_inputs(self):
        # Checks that parameters are in the correct format
        types = ["list", "ndarray", "Series"]
        if (type(self.a).__name__ not in types) or (type(self.b).__name__ not in types):
            raise ValueError(
                "type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'DataFrame'"
            )
        if (self.a[0] > self.a[1]) or (self.b[0] > self.b[1]):
            raise ValueError(
                "the count of successes for a and/or b exceeds the number of trials"
            )
        if (self.prior_alpha <= 0) or (self.prior_beta <= 0):
            raise ValueError("the prior_alpha and/or prior_beta parameters must be > 0")
        if self.n <= 0:
            raise ValueError("n must be a positive integer")
        if self.seed is not None and str(self.seed).isdigit() == False:
            raise ValueError("seed must be a positive integer or None")

    def _posterior_function(self, d):
        # Defines the posterior
        return np.random.beta(
            d[0] + self.prior_alpha, d[1] - d[0] + self.prior_beta, self.n
        )

    def _sample_posteriors(self):
        # Draws from posterior
        np.random.seed(self.seed)
        a_draw = self._posterior_function(self.a)
        b_draw = self._posterior_function(self.b)
        d_draw = b_draw - a_draw
        self.a_draw = a_draw
        self.b_draw = b_draw
        self.d_draw = d_draw

    def get_posteriors(self):
        """
        Retrieves random draws from the posterior
        Returns
        -------
        tuple:
            - np.array[n] draws from the posterior of theta_a
            - np.array[n] draws from the posterior of theta_b
            - np.array[n] draws from the posterior of theta_b minus draws from the posterior of theta_a
        """
        return (self.a_draw, self.b_draw, self.d_draw)

    def _calculate_quantiles(self, d, mean, quantiles):
        # Calculate mean and quantiles
        q = np.quantile(d, quantiles)
        if mean is True:
            q = np.append(q, np.mean(d))
        return q

    def quantile_summary(self, mean=True, quantiles=[0.025, 0.5, 0.975], names=None):
        """
        Summarises the properties of the estimated posterior using quantiles
        Parameters
        ----------
        mean:  boolean, default True, calculates the mean of the draws from the posterior.  Default True
        quantiles: list, calculates the quantiles of the draws from the posterior.  Default [0.025, 0.5, 0.975]
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        pd.DataFrame:
            'theta_a':  summaries of the posterior of theta_a
            'theta_b':  summaries of the posterior of theta_b
            'delta':  summaries of the posterior of theta_b - theta_a
        """
        if quantiles is None:
            raise ValueError("quantiles must be a list of length > 0")
        draws = [self.a_draw, self.b_draw, self.d_draw]
        if names is None:
            names = ["theta_a", "theta_b", "delta"]
        if len(names) > 3:
            raise ValueError("names must be a list of length 3")
        q = []
        for i in draws:
            q.append(self._calculate_quantiles(i, mean, quantiles))
        df = pd.DataFrame(np.array(q))
        if mean is True:
            df.columns = list(map(str, quantiles)) + ["mean"]
        else:
            df.columns = list(map(str, quantiles))
        df["parameter"] = names
        return df

    def _calculate_hdi_and_map(self, d, mean, interval):
        # Calculate HDI interval and MAP
        q = az.hdi(d, hdi_prob=interval)
        m = _calculate_map(d)
        q = np.array([q[0], m, q[1]])
        if mean is True:
            q = np.append(q, np.mean(d))
        return q

    def hdi_summary(self, mean=True, interval=0.95, names=None):
        """
        Summarises the properties of the estimated posterior using the MAP and HDI
        Parameters
        ----------
        mean:  boolean, calculates the mean of the draws from the posterior.  Default True
        interval: float, defines the HDI interval.  Default = 0.95 (i.e. 95% HDI interval)
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        pd.DataFrame:
            'theta_a':  summaries of the posterior of theta_a
            'theta_b':  summaries of the posterior of theta_b
            'delta':  summaries of the posterior of theta_b - theta_a
        """
        if interval is None or interval <= 0 or interval >= 1:
            raise ValueError("interval must be a float > 0 and < 1")
        draws = [self.a_draw, self.b_draw, self.d_draw]
        if names is None:
            names = ["theta_a", "theta_b", "delta"]
        if len(names) > 3:
            raise ValueError("names must be a list of length 3")
        q = []
        for i in draws:
            q.append(self._calculate_hdi_and_map(i, mean, interval))
        df = pd.DataFrame(np.array(q))
        col_names = [
            "%.5g" % ((1 - interval) / 2),
            "MAP",
            "%.5g" % (interval + ((1 - interval) / 2)),
        ]
        if mean is True:
            df.columns = col_names + ["mean"]
        else:
            df.columns = col_names
        df["parameter"] = names
        return df

    def _probability_interpretation_guide(self, p):
        # Interpretation guide for probabilities using:
        # https://www.cia.gov/library/center-for-the-study-of-intelligence/csi-publications/books-and-monographs/sherman-kent-and-the-board-of-national-estimates-collected-essays/6words.html
        if p >= 0 and p <= 0.13:
            i = "almost certainly not"
        elif p > 0.13 and p <= 0.4:
            i = "probably not"
        elif p > 0.4 and p <= 0.6:
            i = "about equally likely"
        elif p > 0.6 and p <= 0.86:
            i = "probably"
        elif p > 0.86 and p <= 1:
            i = "almost certainly"
        else:
            raise ValueError("p must be >= 0 and <= 1")
        return i

    def _print_inference_probability(self, p, i, direction, value, names):
        # Combines inference values into a readable string
        s = "The probability that " + names[1] + " is " + direction + " " + names[0]
        if value != 0:
            s = s + " by more than " + str(value)
        s = s + " is " + ("%.5g" % (p * 100)) + "%."
        s = s + " Therefore " + names[1] + " is " + i + " " + direction + " " + names[0]
        if value != 0:
            s = s + " by more than " + str(value)
        s = s + "."
        return s

    def infer_delta_probability(
        self, direction="greater than", value=0, print_inference=True, names=None
    ):
        """
        Provides a guide to making inferences on the posterior delta, based on proportion of
        draws to the right or left of a given value.
        Parameters
        ----------
        direction: str, defines the direction of the inference, options 'greater than' or 'less than'.  Default is 'greater than'.
        value: float,  defines the value about which to make the inference.  Default = 0.
        print_inference:  boolean, prints a readable string.  Default is True.
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        tuple
            - float, probability that b > (a + value) or b < (a + value).
            - str, string interpretation of that probabiliyu
        """
        dir_opts = ["greater than", "less than"]
        if direction not in dir_opts:
            raise ValueError("direction must be 'greater than' or 'less than'")
        if direction == "greater than":
            p = len(self.d_draw[self.d_draw > 0]) / len(self.d_draw)
        else:
            p = len(self.d_draw[self.d_draw < 0]) / len(self.d_draw)
        i = self._probability_interpretation_guide(p)
        if names is None:
            names = ["theta_a", "theta_b", "delta"]
        if len(names) > 3:
            raise ValueError("names must be a list of length 3")
        if print_inference is True:
            print(self._print_inference_probability(p, i, direction, value, names))
        return p, i

    def _bayes_factor_interpretation_guide(self, bf):
        # Interpretation guide for bayes factors using:
        # Jeffreys guide (https://en.wikipedia.org/wiki/Bayes_factor#cite_note-9)
        if np.isinf(bf) or bf > np.power(10, 2):
            i = "decisive"
        elif bf > np.power(10, 3 / 2) and bf <= np.power(10, 2):
            i = "very strong"
        elif bf > 10 and bf <= np.power(10, 3 / 2):
            i = "strong"
        elif bf > np.power(10, 1 / 2) and bf <= 10:
            i = "substantial"
        elif bf >= 1 and bf <= np.power(10, 1 / 2):
            i = "barely worth mentioning"
        elif bf < 1:
            i = "negative"
        else:
            raise ValueError("bf did not satisfy range of criteria")
        return i

    def _print_inference_bayes_factor(self, bf, i, direction, value, names):
        s = (
            "The calculated bayes factor for the hypothesis that "
            + names[1]
            + " is "
            + direction
            + " "
            + names[0]
        )
        if value != 0:
            s = s + " by more than " + str(value)
        s = (
            s
            + " versus the hypothesis that "
            + names[0]
            + " is "
            + direction
            + " "
            + names[0]
        )
        if value != 0:
            s = s + " by more than " + str(value)
        s = s + " is "
        if np.isinf(bf) is True:
            s = s + "more than 100"
        else:
            s = s + ("%.5g" % bf)
        s = s + ". Therefore the strength of evidence for this hypothesis is " + i + "."
        return s

    def _estimate_bayes_factor(self, p_h1, p_h2):
        # Estimates bayes Factor
        if p_h2 == 0:
            k = np.Infinity
        else:
            k = p_h1 / p_h2
        return k

    def infer_delta_bayes_factor(
        self, direction="greater than", value=0, print_inference=True, names=None
    ):
        """
        Provides a guide to making inferences on the posterior delta, based on the Bayes Factor by estimating
        P(D|H1) / P(D|H2) for the hypotheses H1: b>(a + value) vs H2: (a + value)>b (or vice versa).
        Where D denotes the observed data.
        Parameters
        ----------
        direction: str, defines the direction of the inference, options 'greater than' or 'less than'.  Default is 'greater than'.
        value: float,  defines the value about which to make the inference.  Default = 0.
        print_inference:  boolean, prints a readable string.  Default is True.
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        tuple
            - float, bayes factor for P(D|H1) / P(D|H2) for the hypotheses H1: b>(a + value) vs H2: (a + value)>b (or vice versa).
            - str, string interpretation of that bayes factor
        """
        dir_opts = ["greater than", "less than"]
        if direction not in dir_opts:
            raise ValueError("direction must be 'greater than' or 'less than'")
        if direction == "greater than":
            p_h1 = len(self.d_draw[self.d_draw > value]) / len(self.d_draw)
            p_h2 = 1 - p_h1
            bf = self._estimate_bayes_factor(p_h1, p_h2)
        else:
            p_h1 = len(self.d_draw[self.d_draw < value]) / len(self.d_draw)
            p_h2 = 1 - p_h1
            bf = self._estimate_bayes_factor(p_h1, p_h2)
        i = self._bayes_factor_interpretation_guide(bf)
        if names is None:
            names = ["theta_a", "theta_b", "delta"]
        if len(names) > 3:
            raise ValueError("names must be a list of length 3")
        if print_inference is True:
            print(self._print_inference_bayes_factor(bf, i, direction, value, names))
        return bf, i

    def posterior_plot(
        self,
        method="hdi",
        delta_line=0,
        col="#1f77b4",
        bounds=None,
        names=None,
        fig_size=None,
    ):
        """
        Plots the density of the draws from the posterior distribution
        Parameters
        ----------
        method: str, defines method for interval estimate and central tendency.  Default = 'hdi'
            - 'hdi':  Uses HDI and maximum aposteriori
            - 'quantile': Uses credible intervals and median
        delta_line: float, position of the vertical line on the delta plot
        col:  str, colour of plots.  Default = '#1f77b4' (muted-blue)
        bounds:  float or list, defines the boundaries of the interval
            - if method = 'hdi': float, defines the interval of the HDI. Default = 0.95
            - if method = 'quantile': list, defines the credible interval.  Default = [0.025, 0.975]
        names: list of length 3, parameter names for the plot.  Default ['theta_a', 'theta_b', 'delta']
        fig_size:  tuple(width, height), dimensions of plot.  Default is None
        """
        valid_methods = ["hdi", "quantile"]
        if method not in valid_methods:
            raise ValueError("method must be 'hdi' or 'quantile'")
        if method == "hdi" and bounds is None:
            bounds = 0.95
        if method == "quantile" and bounds is None:
            bounds = [0.025, 0.975]
        if method == "hdi" and (bounds <= 0 or bounds >= 1):
            raise ValueError(
                "if method is 'hdi' then bounds must be a float between 0 and 1"
            )
        if method == "quantiles" and len(bounds) != 2:
            raise ValueError("quantiles must be a list of length 2")
        if names is None:
            names = ["theta_a", "theta_b", "delta"]
        if len(names) > 3:
            raise ValueError("names must be a list of length 3")
        if method == "hdi":
            interval_name = "hdi"
            centre_line_name = "map"
        else:
            interval_name = "credible interval"
            centre_line_name = "median"
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_xaxes=False,
            shared_yaxes=False,
            subplot_titles=tuple(names),
        )
        draws = [self.a_draw, self.b_draw, self.d_draw]
        for i in range(0, 3):
            cl = _get_centre_lines(draws[i], method=method)
            intervals = _get_intervals(draws[i], method=method, bounds=bounds)
            fig.add_trace(
                _make_density_go(draws[i], name="posterior density", col=col), 1, i + 1
            )
            fig.add_trace(
                _make_histogram_go(draws[i], name="posterior draws", col=col), 1, i + 1
            )
            fig.add_trace(_make_line_go(cl, name=centre_line_name, col=col), 1, i + 1)
            fig.add_trace(
                _make_area_go(intervals, name=interval_name, col=col), 1, i + 1
            )
        fig.update_layout(shapes=[_make_delta_line(self.d_draw, delta_line=delta_line)])
        fig.update_yaxes(title_text="density", row=1, col=1)
        name_set = set()
        fig.for_each_trace(
            lambda trace: trace.update(showlegend=False)
            if (trace.name in name_set)
            else name_set.add(trace.name)
        )
        if fig_size is not None:
            fig.update_layout(height=fig_size[1], width=fig_size[0])
        return fig
