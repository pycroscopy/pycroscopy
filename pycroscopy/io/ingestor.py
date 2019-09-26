# -*- coding: utf-8 -*-
"""
Utilities for automated ingestion of data and metadata in proprietary file formats

Created on Fri May 25 11:14:25 2019

@author: Suhas Somnath
"""

from __future__ import division, unicode_literals, print_function, absolute_import
import sys
from warnings import warn
from . import translators

if sys.version_info.major == 3:
    unicode = str


def ingest(file_path, force_translator=None, unique_translator=True,
           verbose=False, **kwargs):
    """
    Translates raw data file(s) in proprietary file formats into a h5USID file

    Parameters
    ----------
    file_path : str
        Path to raw data file(s)
    unique_translator : bool, Optional. Default - True
        If True and multiple translators claim to be able to translate a given
        file, a ValueError will be raised
        If False, the last valid translator found will be used
    force_translator : str, Optional. Default - Ignored
        Name of the Translator class to instantiate.
        Use this if multiple translators claim to be able to translate a given
        file and the desired translator
        is not chosen automatically
    verbose : bool, Optional. Default = False
        Whether or not to print print statements for debugging
    kwargs : keyword arguments that will be passed on to the translate() function

    Returns
    -------
    str
        Absolute path to the h5USID file that resulted from the translation
    """
    if not isinstance(file_path, (str, unicode)):
        raise TypeError('file_path must be a string object')
    if isinstance(force_translator, (str, unicode)):
        trans_class = getattr(translators, force_translator)
        # Assumes that is_valid_file() would have returned file_path
        valid_translators = [trans_class(), file_path]
    else:
        # Could have used an OrderedDict as well.
        valid_translators = list()

        for trans_class in translators.all_translators:
            trans_obj = trans_class()
            # The following line is in place until the updated Translator class is pushed
            # after that time, all Translators will inherit this function
            func = getattr(trans_obj, "is_valid_file", None)
            if callable(func):
                try:
                    ret_path = func(file_path)
                    if verbose:
                        print(
                            '{} has implemented the is_valid_file() function'.format(
                                trans_class.__name__))
                except NotImplementedError:
                    continue
                if isinstance(ret_path, (str, unicode)):
                    if verbose:
                        print(
                            '\tThis translator is able to read the provided file(s)')
                    valid_translators.append([trans_obj, ret_path])
        if len(valid_translators) > 1:
            trans_names = '\n'.join([str(obj.__class__.__name__) for [obj, _] in valid_translators])
            mesg = 'The provided file can be read using the following Translators:\n' + trans_names

            if unique_translator:
                mesg += '\nConsider providing the name of desired translator class ' \
                        'via "force_translator" or setting unique_translator=False'
                raise ValueError(mesg)
            else:
                mesg += '\n\n{} will be used to translate the provided file(s)' \
                     '.'.format(valid_translators[-1][0].__class__.__name__)
                warn(mesg)
        elif len(valid_translators) == 0:
            raise NotImplementedError(
                'No translator available to translate: {}'.format(file_path))

        # Pick the last translator object given that multiple claim to be able to translate the file
        valid_translators = valid_translators[-1]

    # Finally translate the file:
    if verbose:
        print('{} will be provided to {} for translation'.format(valid_translators[1], valid_translators[0].__class__.__name__))
    return valid_translators[0].translate(valid_translators[1], **kwargs)
