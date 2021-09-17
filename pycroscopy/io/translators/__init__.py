# -*- coding: utf-8 -*-
"""
A collection of Translators that extract data from custom & proprietary microscope formats and write them to
standardized USID HDF5 files.

Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""
from .AR_hdf5 import ARhdf5
from .bruker_afm import BrukerAFMTranslator
from .nanonis import NanonisTranslator, NanonisTranslatorCorrect
from .pifm import PiFMTranslator

__all__ = ['BrukerAFMTranslator', 'PiFMTranslator', 'NanonisTranslator',
           'ARhdf5']

afm_translators = [PiFMTranslator, BrukerAFMTranslator, ARhdf5]

stm_translators = [NanonisTranslatorCorrect]

stem_translators = []

misc_translators = []

all_translators = afm_translators + stm_translators + stem_translators + misc_translators