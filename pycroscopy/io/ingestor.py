# -*- coding: utf-8 -*-
"""
Utilities for automated ingestion of data and metadata in proprietary file formats

Created on Fri May 25 11:14:25 2019

@author: Suhas Somnath
"""

from __future__ import division, unicode_literals, print_function, absolute_import
from warnings import warn
from .translators import all_translators


def ingest(file_path, *args, verbose=False, **kwargs):
    """
    Translates raw data file(s) in proprietary file formats into a h5USID file

    Parameters
    ----------
    file_path : str
        Path to raw data file(s)
    args : Arguments that will be passed on to the translator
    verbose : bool, Optional. Default = False
        Whether or not to print print statements for debugging
    kwargs

    Returns
    -------

    """
    valid_translators = []

    for trans_class in all_translators:
        trans_obj = trans_class()
        # The following line is in place until the updated Translator class is pushed
        # after that time, all Translators will inherit this function
        func = getattr(trans_obj, "is_valid_file", None)
        if callable(func):
            try:
                test = func(file_path)
                if verbose:
                    print(
                        '{} has implemented the is_valid_file() function'.format(
                            trans_obj))
            except NotImplementedError:
                continue
            if test:
                if verbose:
                    print(
                        '\tThis translator is able to read the provided file(s)')
                valid_translators.append(trans_obj)
    if len(valid_translators) > 1:
        trans_names = '\n'.join([str(obj) for obj in valid_translators])
        warn(
            'The provided file can be read using the following Translators:\n{}.'
            '\n\n{} will be used to translate the provided file(s)'
            '.'.format(trans_names, valid_translators[-1]))
    elif len(valid_translators) == 0:
        raise NotImplementedError(
            'No translator available to translate: {}'.format(file_path))

    # Pick the last translator object given that multiple claim to be able to translate the file
    valid_translators = valid_translators[-1]

    # Finally translate the file:
    if verbose:
        print('{} will be used for the translation'.format(valid_translators))
    return valid_translators.translate(file_path, *args, **kwargs)
