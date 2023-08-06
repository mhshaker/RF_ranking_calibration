import sys
sys.path.append('../') # to access the files in higher directories

import Data.data_provider as dp


def test_load():
    x, y = dp.load_data("spambase")
    assert len(x) > 0

