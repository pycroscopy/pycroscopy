"""
Unit Test created by Christopher Smith on 7/17/18
"""
from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import numpy as np
import sys
sys.path.append("../../../../pycroscopy/")

from pycroscopy.io.translators.df_utils.image_utils import unnest_parm_dicts, try_tag_to_string, no_bin

class TestImageUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_unnext_parm_dicts(self):
        nested_dict = {'level_1_a': {'level_2_a': {'level_3_a': '3a_things',
                                                   'level_3_b': '3b_things',
                                                   'level_3_c': {'level_4_a': '4a_stuff'}},
                                     'level_2_b': '2b_junk'},
                       'level_1_b': '1b_crap'}

        unnested_dict = {'level_1_a-level_2_a-level_3_a': '3a_things',
                         'level_1_a-level_2_a-level_3_b': '3b_things',
                         'level_1_a-level_2_a-level_3_c-level_4_a': '4a_stuff',
                         'level_1_a-level_2_b': '2b_junk',
                         'level_1_b': '1b_crap'}

        test_dict = unnest_parm_dicts(nested_dict)

        for key, val in unnested_dict.items():
            self.assertTrue(key in test_dict.keys())
            self.assertTrue(val == test_dict[key])

    def test_try_tag_to_string_success(self):
        pass
        # TODO Figure out how to turn strings into byte-arrays for the try_tag_to_string function
        # This doesn't work for some reason.  Need to investigate further.
        # test_strings = ['banana', 'bacon', 'sideways', 'question', 'discombobulated', '5w0rdf1sh']
        #
        # for string in test_strings:
        #     int_string = int(string.encode('utf-16').hex(), 16)
        #
        #     tmp_str = try_tag_to_string(int_string)
        #
        #     self.assertTrue(string == tmp_str)

    def test_try_tag_to_string_error(self):
        test_strings = []


        pass