# -*- coding: utf-8 -*-
"""
A collection of Translators that extract data from custom & proprietary microscope formats and write them to
standardized USID HDF5 files.

Created on Tue Jan 05 07:55:56 2016

@author: Suhas Somnath, Chris Smith
"""
from .be_odf import BEodfTranslator
from .be_odf_relaxation import BEodfRelaxationTranslator
from .beps_ndf import BEPSndfTranslator
from .general_dynamic_mode import GDMTranslator
from .gmode_iv import GIVTranslator
from .gmode_line import GLineTranslator
from .gmode_tune import GTuneTranslator
from .igor_ibw import IgorIBWTranslator
from .ndata import NDataTranslator
from .tr_kpfm import TRKPFMTranslator
from .oneview import OneViewTranslator
from .image_stack import PtychographyTranslator
from .image_stack import ImageStackTranslator
from .sporc import SporcTranslator
from .time_series import MovieTranslator
from .bruker_afm import BrukerAFMTranslator
from .beps_data_generator import FakeBEPSGenerator
from .labview_h5_patcher import LabViewH5Patcher
from .nanonis import NanonisTranslator, NanonisTranslatorCorrect
from .image import ImageTranslator
from .pifm import PiFMTranslator
from .gwyddion import GwyddionTranslator
from .omicron_asc import AscTranslator

__all__ = ['BEodfTranslator', 'BEPSndfTranslator', 'BEodfRelaxationTranslator',
           'GIVTranslator', 'GLineTranslator', 'GTuneTranslator', 'GDMTranslator',
           'SporcTranslator', 'MovieTranslator', 'IgorIBWTranslator',
           'OneViewTranslator', 'NDataTranslator', 'FakeBEPSGenerator', 'PtychographyTranslator',
           'LabViewH5Patcher', 'TRKPFMTranslator', 'BrukerAFMTranslator', 'ImageTranslator',
           'PiFMTranslator', 'NanonisTranslator', 'GwyddionTranslator', 'AscTranslator']

be_translators = [BEodfTranslator, BEodfRelaxationTranslator, BEPSndfTranslator, FakeBEPSGenerator, LabViewH5Patcher]

gmode_translators = [GDMTranslator, GIVTranslator, GLineTranslator, GTuneTranslator, TRKPFMTranslator, SporcTranslator]

afm_translators = [IgorIBWTranslator, PiFMTranslator, BrukerAFMTranslator, GwyddionTranslator]

stm_translators = [NanonisTranslatorCorrect, AscTranslator]

stem_translators = [NDataTranslator, OneViewTranslator]

misc_translators = [ImageStackTranslator, MovieTranslator, ImageTranslator]

all_translators = be_translators + gmode_translators + afm_translators + stm_translators + stem_translators + misc_translators