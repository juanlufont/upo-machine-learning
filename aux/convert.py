#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas

dataset_arff = "./dataset_etapas_class17_idx.arff"
dataset_csv = "./dataset_etapas_class17_idx.csv"
metadata_lines = 24

headers = [
    "NEtapa",
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
    "P08",
    "P09",
    "P10",
    "P11",
    "P12",
    "P13",
    "P14",
    "P15",
    "P16",
    "P18",
    "P19",
    "P17",
]

data = pandas.read_csv(dataset_arff, skiprows=metadata_lines, names=headers)

data.to_csv(dataset_csv, index=False)

df = pandas.SparseDataFrame()

# dropping "NEtapa Atribute"
# data = data.drop(columns=["NEtapa"])
