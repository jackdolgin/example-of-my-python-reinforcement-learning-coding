#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import numpy as np
from pandas import read_csv, DataFrame
from random import choice
from scipy.optimize import minimize
from simulate_and_model import Agent, mle


def extract_key_variables(func):

    rocket_pairs = ["one", "two"]
    pair_sides = ["a", "b"]
    planets = ["red", "purple"]

    def procedure(**kwargs):
        return func(rocket_pairs, pair_sides, planets, **kwargs)

    return procedure


@extract_key_variables
def simulate(rocket_pairs, pair_sides, planets, α, β, λ, π, ρ, w_high, w_faux_high, w_faux_low, w_low, trials):

    agent = Agent("simulate", rocket_pairs, planets, α, β, λ, π, ρ, (w_high, w_faux_high, w_faux_low, w_low))

    for trial in range(int(trials)):
        og_pair = choice(rocket_pairs)
        stake = "high" if og_pair == rocket_pairs[0] else "low"
        stake = stake if np.random.uniform() <= 2/3 else "faux_" + stake

        pair_sides = choice(pair_sides)
        agent.trial(trial, og_pair, stake, pair_sides)

    pandas_df = DataFrame(agent.log, columns=tuple(agent.log.keys()))
    return pandas_df


@extract_key_variables
def model(rocket_pairs, pair_sides, planets, data_directory, sub_path, include_priors,
          α_0, α_lb, α_ub, β_0, β_lb, β_ub, λ_0, λ_lb, λ_ub,
          π_0, π_lb, π_ub, ρ_0, ρ_lb, ρ_ub,
          w_high_0, w_high_lb, w_high_ub,
          w_faux_high_0, w_faux_high_lb, w_faux_high_ub,
          w_faux_low_0, w_faux_low_lb, w_faux_low_ub,
          w_low_0, w_low_lb, w_low_ub):

    if type(sub_path) == str:
        sub_df = read_csv(path.join(data_directory, "Spliced", sub_path))
    else:
        sub_df = sub_path

    fit = minimize(
        mle,
        np.array([α_0, β_0, λ_0, π_0, ρ_0, w_high_0,
                  w_faux_high_0, w_faux_low_0, w_low_0]),
        args=(rocket_pairs, planets, sub_df, include_priors),
        method='L-BFGS-B',
        bounds=((α_lb, α_ub), (β_lb, β_ub), (λ_lb, λ_ub),
                (π_lb, π_ub), (ρ_lb, ρ_ub),
                (w_high_lb, w_high_ub),
                (w_faux_high_lb, w_faux_high_ub),
                (w_faux_low_lb, w_faux_low_ub),
                (w_low_lb, w_low_ub))
    )

    if type(sub_path) != str:
        fit = [list(fit.keys()), fit, sub_df['completed_trial'].sum()]
    else:
        fit['trials'] = sub_df['completed_trial'].sum()

    return fit
