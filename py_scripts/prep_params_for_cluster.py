#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from siuba import _, filter, mutate, if_else, case_when, group_by, ungroup, select
from wrapper import extract_key_variables
from run_wrapper import _thisDir, data_dir
import os
from pathlib import Path


@extract_key_variables
def splice_raw(rocket_pairs, pair_sides, planets):

    # remove any files currently in the directory, since we will ultimately model every file left in the directory
    [f.unlink() for f in Path(os.path.join(data_dir, "Spliced")).glob("*") if f.is_file()]

    all_raw = pd.read_csv(os.path.join(data_dir, "Raw_Data.csv"))

    all_mutated = (all_raw
      >> filter(_.practice == 0)
      >> mutate(
           og_pair = if_else(_.state1 == 1, rocket_pairs[0], rocket_pairs[1]),
           pair_sides = case_when(_, {
             _.stim_left == 1: pair_sides[0],
             _.stim_left == 3: pair_sides[0],
             True: pair_sides[1]
           }),
           preset_planet=case_when(_, {
             _.state2 == 1: planets[0],
             _.state2 == 2: planets[1],
             True: "NA"
           }),
           completed_trial = if_else(_.rt_2 == -1, 0, 1),
         )
      )

    all_subs = all_mutated.assignment_id.unique()

    for sub_id in all_subs:

        neutral_stake = 1 if sys.argv[2] == "first_go" else 3

        sub_df = (all_mutated
          >> filter(_.assignment_id == sub_id)
          >> group_by("state1")
          >> mutate(
               stake_mean = _.stake.mean(),
               stake_group = if_else(_.stake_mean > neutral_stake, "high", "low"),
               stake_type = if_else(_.stake == neutral_stake, "faux_" + _.stake_group, _.stake_group)
             )
          >> ungroup()
          # to keep this code consistent with my r code, i could set trial_index column equal to actual row number, now that practice trials have been filtered; not a biggie, though
          >> select(
               _.trial_index,
               _.og_pair,
               _.stake_type,
               _.pair_sides,
               _.preset_planet,
               _.completed_trial,
               _.points
             )
          )

        # if sub_df['completed_trial'].sum() > 240:  # filter out participants with a lot of no responses
        if ((sys.argv[2] == 'first_go' and sub_df['completed_trial'].sum() > 240) or (sys.argv[2] == 'second_go' and sub_df['completed_trial'].sum() > 213)):
            output_path = os.path.join(data_dir, "Spliced", str(sub_id) + ".csv")
            sub_df.to_csv(output_path)


def generate_csv_of_params(func):
    table_of_params = {}

    def generating_func(recreate_indv_csvs, iterations, include_priors, **kwargs):

        if recreate_indv_csvs:
            splice_raw()

        func(table_of_params, iterations, include_priors, **kwargs)
        df = pd.DataFrame(table_of_params)
        df.to_csv(os.path.join(_thisDir, "..", sys.argv[2], "all_params_for_fitting.csv"))
    return generating_func


@generate_csv_of_params
def generate_params(param_dict, iterations, include_priors, **kwargs):

    spliced_dir = os.listdir(os.path.join(data_dir, "Spliced"))

    param_dict["sub_path"] = spliced_dir * iterations

    param_reps = iterations * len(spliced_dir)

    param_dict["include_priors"] = [include_priors] * param_reps

    for param, bounds in kwargs.items():

        param_dict[param + '_0'] = np.random.default_rng().uniform(*bounds, param_reps)

        param_dict[param + "_lb"] = bounds[0]
        param_dict[param + "_ub"] = bounds[1]

    return param_dict


if __name__ == '__main__':
    iters_per_sub = int(sys.argv[1])
    generate_params(True, iters_per_sub, True, α=(0.0001, .9999), β=(0.0001, 20), λ=(0.0001, .9999), π=(-20, 20), ρ=(-20, 20),
                    w_high=(0.0001, .9999), w_faux_high=(0.0001, .9999), w_faux_low=(0.0001, .9999), w_low=(0.0001, .9999))
