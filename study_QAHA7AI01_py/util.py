import gzip
import pickle
import re
from timeit import default_timer as timer


class Util:
    """ """

    def __init__(self):
        """ """
        pass

    @staticmethod
    def topickle(obj, path, compresslevel=9):
        """
        pickle obj to disk
        compresslevel from 0 to 9, 9 is default, slowest, most compressed
        """
        pickle.dump(
            obj=obj, file=gzip.open(path, "wb", compresslevel=1), protocol=pickle.HIGHEST_PROTOCOL
        )

    @staticmethod
    def unpickle(path):
        """
        unpickle obj from disk
        """
        return pickle.load(gzip.open(path, "rb"))

    @staticmethod
    def timer_start(name):
        """ """
        print(f"start {name}", end=" ")
        t0 = timer()
        return t0

    @staticmethod
    def timer_stop(t0):
        """ """
        t1 = timer()
        print(f"- done in  {t1-t0:.2f} s")

    @staticmethod
    def multiple_replace(text, dic):
        if len(dic) == 0:
            return text
        # Create a regex from the dict keys
        regex = re.compile("(%s)" % "|".join(map(re.escape, dic.keys())))
        # For each match, lookup corresponding value in dict
        return regex.sub(lambda mo: dic[mo.string[mo.start() : mo.end()]], text)