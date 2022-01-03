#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv
from os import path, chdir
import pandas as pd
from wrapper import model
from uuid import uuid4
from pickle import dump

_thisDir = path.dirname(path.abspath(__file__))
chdir(_thisDir)
data_dir = path.join(_thisDir, "..", "..", "Data", "second_go")

if __name__ == '__main__':

    df = pd.read_csv(path.join(_thisDir, "..", "second_go", "all_params_for_fitting.csv")) # reads in the csv of parameters for fitting
    index = int(argv[1]) - 1

    params = df.iloc[index, :][1:]                                              # these two lines grab a row of the csv based on argv[1]...
    params = pd.Series.to_dict(params)                                          # ... and then we convert that row to a dictionary (where column names are the keys and cell values are values)

    results = {**params, **model(data_directory=data_dir, **params)}            # creates a dictionary based on the dictionary of csv params, and also of the results from fitting the model (which gets fit based on feeding in params as the input)

    random_characters = uuid4().hex.upper()[0:10]                               # creates a string of random characters to be added to the end of the pickle file to distinguish multiple fits from the same participant

    filename = path.join(_thisDir, "..", "second_go", "fits", f"fit_{index}_{random_characters}.pickle")
    with open(filename, "wb") as handle:
        dump(results, handle)
