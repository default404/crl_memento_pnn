
import unittest

import tests.test_utils as util
import tests.test_continal_env as cenv

#note tests will be run in this order
TEST_CASES = [util, cenv]

if __name__ == '__main__':
    loader = unittest.TestLoader()

    SUITES = []
    for tm in TEST_CASES:
        suite = loader.loadTestsFromModule(tm)
        SUITES.append(suite)

    SUITE_WRAP = unittest.TestSuite(SUITES)
    res = unittest.TextTestRunner(verbosity=2).run(SUITE_WRAP)
    # print(res)