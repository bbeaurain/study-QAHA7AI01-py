import os

from .util import Util

pj = os.path.join


class Chapter_B:
    """"""

    def __init__(self, folder_sample="."):
        """"""
        self.folder_sample = folder_sample
        self.df = None

    def load_sample(self, filename):
        """"""
        t0 = Util.timer_start("load_sample")
        path = pj(self.folder_sample, filename)
        self.df = Util.unpickle(path)
        self.name = filename
        Util.timer_stop(t0)

    def overview_df(self):
        """"""
        print(f"overview dataframe {self.name}")
        n_row = len(self.df)
        n_col = len(self.df.columns)
        print(f"\t n_row = {n_row}")
        print(f"\t n_col = {n_col}")
