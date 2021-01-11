# -*- coding: utf-8 -*-
"""
A collection of Translators that extract data from custom & proprietary microscope formats and write them to
standardized USID HDF5 files.

Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""
from .igor_ibw import IgorIBWTranslator
from .ndata import NDataTranslator
from .oneview import OneViewTranslator
from .image_stack import PtychographyTranslator
from .image_stack import ImageStackTranslator
from .time_series import MovieTranslator
from .bruker_afm import BrukerAFMTranslator
from .nanonis import NanonisTranslator, NanonisTranslatorCorrect
from .image import ImageTranslator
from .pifm import PiFMTranslator
from .gwyddion import GwyddionTranslator
from .omicron_asc import AscTranslator

__all__ = ['MovieTranslator', 'IgorIBWTranslator',
           'OneViewTranslator', 'NDataTranslator', 'PtychographyTranslator',
           'BrukerAFMTranslator', 'ImageTranslator',
           'PiFMTranslator', 'NanonisTranslator', 'GwyddionTranslator', 'AscTranslator']

afm_translators = [IgorIBWTranslator, PiFMTranslator, BrukerAFMTranslator, GwyddionTranslator]

stm_translators = [NanonisTranslatorCorrect, AscTranslator]

stem_translators = [NDataTranslator, OneViewTranslator]

misc_translators = [ImageStackTranslator, MovieTranslator, ImageTranslator]

all_translators = afm_translators + stm_translators + stem_translators + misc_translators