# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 15:21:46 2018

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from os import path, remove
from collections import OrderedDict
import numpy as np
import h5py

from ...core.io.translator import Translator, generate_dummy_main_parms
from ...core.io.write_utils import Dimension
from ...core.io.hdf_utils import create_indexed_group, write_main_dataset, write_simple_attrs, write_ind_val_dsets

# packages specific to this kind of file
from .df_utils.gsf_read import gsf_read
import gwyfile


class GwyddionTranslator(Translator):

    def translate(self, filepath, *args, **kwargs):

        # Two kinds of files:
        # 1. Simple GSF files -> use metadata, data = gsf_read(filepath)
        # 2. Native .gwy files -> use the gwyfile package
        # I have a notebook that shows how such data can be read.
        raise NotImplementedError('This translator has not yet been implemented')

    def _translate_image_stack(self):
        """
        Use this function to write data corrsponding to a stack of scan images (most common)0
        Returns
        -------

        """
        pass

    def _translate_3d_spectroscopy(self):
        """
        Use this to translate force-maps, I-V spectroscopy etc.
        Returns
        -------

        """
        pass

    def _translate_spectra(self):
        """
        Use this to translate simple 1D data like force curves
        Returns
        -------

        """
        pass
