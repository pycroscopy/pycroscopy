import os
import numpy as np

_end_tags = dict(grid=':HEADER_END:', scan='SCANIT_END', spec='[DATA]')


class NanonisFile(object):
    """
    Base class for Nanonis data files (grid, scan, point spectroscopy).

    Handles methods and parsing tasks common to all Nanonis files.

    Parameters
    ----------
    fname : str
        Name of Nanonis file.

    Attributes
    ----------
    datadir : str
        Directory path for Nanonis file.
    basename : str
        Just the filename, no path.
    fname : str
        Full path of Nanonis file.
    filetype : str
        filetype corresponding to filename extension.
    byte_offset : int
        Size of header in bytes.
    header_raw : str
        Unproccessed header information.
    """

    def __init__(self, fname):
        """

        Parameters
        ----------
        fname
        """
        self.datadir, self.basename = os.path.split(fname)
        self.fname = fname
        self.filetype = self._determine_filetype()
        self.byte_offset = self.start_byte()
        self.header_raw = self.read_raw_header(self.byte_offset)

    def _determine_filetype(self):
        """
        Check last three characters for appropriate file extension,
        raise error if not.

        Returns
        -------
        str
            Filetype name associated with extension.

        Raises
        ------
        UnhandledFileError
            If last three characters of filename are not one of '3ds',
            'sxm', or 'dat'.
        """

        if self.fname[-3:] == '3ds':
            return 'grid'
        elif self.fname[-3:] == 'sxm':
            return 'scan'
        elif self.fname[-3:] == 'dat':
            return 'spec'
        else:
            raise UnhandledFileError('{} is not a supported filetype or does not exist'.format(self.basename))

    def read_raw_header(self, byte_offset):
        """
        Return header as a raw string.

        Everything before the end tag is considered to be part of the header.
        the parsing will be done later by subclass methods.

        Parameters
        ----------
        byte_offset : int
            Size of header in bytes. Read up to this point in file.

        Returns
        -------
        str
            Contents of filename up to byte_offset as a decoded binary
            string.
        """

        with open(self.fname, 'rb') as f:
            return f.read(byte_offset).decode()

    def start_byte(self):
        """
        Find first byte after end tag signalling end of header info.

        Caveat, I believe this is the first byte after the end of the
        line that the end tag is found on, not strictly the first byte
        directly after the end tag is found. For example in Scan
        __init__, byte_offset is incremented by 4 to account for a
        'start' byte that is not actual data.

        Returns
        -------
        int
            Size of header in bytes.
        """

        with open(self.fname, 'rb') as f:
            tag = _end_tags[self.filetype]

            # Set to a default value to know if end_tag wasn't found
            byte_offset = -1

            for line in f:
                # Convert from bytes to str
                entry = line.strip().decode()
                if tag in entry:
                    byte_offset = f.tell()
                    break

            if byte_offset == -1:
                raise FileHeaderNotFoundError(
                    'Could not find the {} end tag in {}'.format(tag, self.basename)
                )

        return byte_offset


class Grid(NanonisFile):
    """
    Nanonis grid file class.

    Contains data loading method specific to Nanonis grid file. Nanonis
    3ds files contain a header terminated by '\r\n:HEADER_END:\r\n'
    line, after which big endian encoded binary data starts. A grid is
    always recorded in an 'up' direction, and data is recorded
    sequentially starting from the first pixel. The number of bytes
    corresponding to a single pixel will depend on the experiment
    parameters. In general the size of one pixel will be a sum of

        - # fixed parameters
        - # experimental parameters
        - # sweep signal points (typically bias).

    Hence if there are 2 fixed parameters, 8 experimental parameters,
    and a 512 point bias sweep, a pixel will account 4 x (522) = 2088
    bytes of data. The class intuits this from header info and extracts
    the data for you and cuts it up into each channel, though normally
    this should be just the current.

    Currently cannot accept grids that are incomplete.

    Parameters
    ----------
    fname : str
        Filename for grid file.

    Attributes
    ----------
    header : dict
        Parsed 3ds header. Relevant fields are converted to float,
        otherwise most are string values.
    signals : dict
        Dict keys correspond to channel name, with values being the
        corresponding data array.

    Raises
    ------
    UnhandledFileError
        If fname does not have a '.3ds' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='3ds')
        super(Grid, self).__init__(fname)
        self.header = _parse_3ds_header(self.header_raw)
        self.signals = self._load_data()
        self.signals['sweep_signal'] = self._derive_sweep_signal()
        self.signals['topo'] = self._extract_topo()

    def _load_data(self):
        """
        Read binary data for Nanonis 3ds file.

        Returns
        -------
        dict
            Channel name keyed dict of 3d array.
        """
        # load grid params
        nx, ny = self.header['dim_px']
        num_sweep = self.header['num_sweep_signal']
        num_param = self.header['num_parameters']
        num_chan = self.header['num_channels']
        data_dict = dict()

        # open and seek to start of data
        f = open(self.fname, 'rb')
        f.seek(self.byte_offset)
        data_format = '>f4'
        griddata = np.fromfile(f, dtype=data_format)
        f.close()

        # pixel size in bytes
        exp_size_per_pix = num_param + num_sweep * num_chan

        # reshape from 1d to 3d
        griddata_shaped = griddata.reshape((nx, ny, exp_size_per_pix))

        # experimental parameters are first num_param of every pixel
        params = griddata_shaped[:, :, :num_param]
        data_dict['params'] = params

        # extract data for each channel
        for i, chann in enumerate(self.header['channels']):
            start_ind = num_param + i * num_sweep
            stop_ind = num_param + (i + 1) * num_sweep
            data_dict[chann] = griddata_shaped[:, :, start_ind:stop_ind]

        return data_dict

    def _derive_sweep_signal(self):
        """
        Computer sweep signal.

        Based on start and stop points of sweep signal in header, and
        number of sweep signal points.

        Returns
        -------
        numpy.ndarray
            1d sweep signal, should be sample bias in most cases.
        """
        # find sweep signal start and end from a given pixel value
        sweep_start, sweep_end = self.signals['params'][0, 0, :2]
        num_sweep_signal = self.header['num_sweep_signal']

        return np.linspace(sweep_start, sweep_end, num_sweep_signal, dtype=np.float32)

    def _extract_topo(self):
        """
        Extract topographic map based on z-controller height at each
        pixel.

        The data is already extracted, though it lives in the signals
        dict under the key 'parameters'. Currently the 4th column is the
        Z (m) information at each pixel, should update this to be more
        general in case the fixed/experimental parameters are not the
        same for other Nanonis users.

        Returns
        -------
        numpy.ndarray
            Copy of already extracted data to be more easily accessible
            in signals dict.
        """
        return self.signals['params'][:, :, 4]


class Scan(NanonisFile):
    """
    Nanonis scan file class.

    Contains data loading methods specific to Nanonis sxm files. The
    header is terminated by a 'SCANIT_END' tag followed by the \1A\04
    code. The NanonisFile header parse method doesn't account for this
    so the Scan __init__ method just adds 4 bytes to the byte_offset
    attribute so as to not include this as a datapoint.

    Data is structured a little differently from grid files, obviously.
    For each pixel in the scan, each channel is recorded forwards and
    backwards one after the other.

    Currently cannot take scans that do not have both directions
    recorded for each channel, nor incomplete scans.

    Parameters
    ----------
    fname : str
        Filename for scan file.

    Attributes
    ----------
    header : dict
        Parsed sxm header. Some fields are converted to float,
        otherwise most are string values.
    signals : dict
        Dict keys correspond to channel name, values correspond to
        another dict whose keys are simply forward and backward arrays
        for the scan image.

    Raises
    ------
    UnhandledFileError
        If fname does not have a '.sxm' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='sxm')
        super(Scan, self).__init__(fname)
        self.header = _parse_sxm_header(self.header_raw)

        # data begins with 4 byte code, add 4 bytes to offset instead
        self.byte_offset += 4

        # load data
        self.signals = self._load_data()

    def _load_data(self):
        """
        Read binary data for Nanonis sxm file.

        Returns
        -------
        dict
            Channel name keyed dict of each channel array.
        """
        channs = list(self.header['data_info']['Name'])
        nchanns = len(channs)
        nx, ny = self.header['scan_pixels']

        # assume both directions for now
        ndir = 2

        data_dict = dict()

        # open and seek to start of data
        f = open(self.fname, 'rb')
        f.seek(self.byte_offset)
        data_format = '>f4'
        scandata = np.fromfile(f, dtype=data_format)
        f.close()

        # reshape
        scandata_shaped = scandata.reshape(nchanns, ndir, nx, ny)

        # extract data for each channel
        for i, chann in enumerate(channs):
            chann_dict = dict(forward=scandata_shaped[i, 0, :, :],
                              backward=scandata_shaped[i, 1, :, :])
            data_dict[chann] = chann_dict

        return data_dict


class Spec(NanonisFile):
    """
    Nanonis point spectroscopy file class.

    These files are a little easier to handle since they are stored in
    ascii format.

    Parameters
    ----------
    fname : str
        Filename for spec file.

    Attributes
    ----------
    header : dict
        Parsed dat header.

    Raises
    ------
    UnhandledFileError
        If fname does not have a '.dat' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='dat')
        super(Spec, self).__init__(fname)
        self.header = _parse_dat_header(self.header_raw)
        self.signals = self._load_data()

    def _load_data(self):
        """
        Loads ascii formatted .dat file.

        Header ended by '[DATA]' tag.

        Returns
        -------
        dict
            Keys correspond to each channel recorded, including
            saved/filtered versions of other channels.
        """

        # done differently since data is ascii, not binary
        f = open(self.fname, 'r')
        f.seek(self.byte_offset)
        data_dict = dict()

        column_names = f.readline().strip('\n').split('\t')
        f.close()
        header_lines = len(self.header) + 4
        specdata = np.genfromtxt(self.fname, delimiter='\t', skip_header=header_lines)

        for i, name in enumerate(column_names):
            data_dict[name] = specdata[:, i]

        return data_dict


class UnhandledFileError(Exception):
    """
    To be raised when unknown file extension is passed.
    """
    pass


class FileHeaderNotFoundError(Exception):
    """
    To be raised when no header information could be determined.
    """
    pass


def _parse_3ds_header(header_raw):
    """
    Parse raw header string.

    Empirically done based on Nanonis header structure. See Grid
    docstring or Nanonis help documentation for more details.

    Parameters
    ----------
    header_raw : str
        Raw header string from read_raw_header() method.

    Returns
    -------
    dict
        Channel name keyed dict of 3d array.
    """
    # cleanup string and remove end tag as entry
    header_entries = header_raw.split('\r\n')
    header_entries = header_entries[:-2]

    # Convert the strings to a dictionary.
    raw_dict = dict()
    for entry in header_entries:
        key, val = _split_header_entry(entry)
        raw_dict[key] = val

    # Transfer parameters from raw_dict to header_dict
    # Get the expected parameters first
    header_dict = dict()

    # grid dimensions in pixels
    dim_px_str = raw_dict.pop('Grid dim', '1 x 1')
    header_dict['dim_px'] = [int(val) for val in dim_px_str.split(' x ')]

    # grid frame center position, size, angle
    grid_str = raw_dict.pop('Grid settings', [0, 0, 0, 0])
    header_dict['pos_xy'] = [float(val) for val in grid_str[:2]]
    header_dict['size_xy'] = [float(val) for val in grid_str[2:4]]
    header_dict['angle'] = float(grid_str[-1])

    # sweep signal
    header_dict['sweep_signal'] = raw_dict.pop('Sweep Signal', 'Bias (V)')

    # fixed parameters
    header_dict['fixed_parameters'] = raw_dict.pop('Fixed parameters', [''])

    # experimental parameters
    header_dict['experimental_parameters'] = raw_dict.pop('Experiment parameters', [''])

    # number of parameters (each 4 bytes)
    header_dict['num_parameters'] = int(raw_dict.pop('# Parameters (4 byte)',
                                                     len(header_dict['fixed_parameters']) +
                                                     len(header_dict['experimental_parameters'])))

    # experiment size in bytes
    header_dict['experiment_size'] = int(raw_dict.pop('Experiment size (bytes)', header_dict['num_parameters'] * 2))

    # number of points of sweep signal
    header_dict['num_sweep_signal'] = int(raw_dict.pop('Points', 1))

    # channel names
    header_dict['channels'] = raw_dict.pop('Channels', ['Input 1'])
    header_dict['num_channels'] = len(header_dict['channels'])

    # measure delay
    header_dict['measure_delay'] = float(raw_dict.pop('Delay before measuring (s)', 0))

    # metadata
    header_dict['experiment_name'] = raw_dict.pop('Experiment', 'Experiment')
    header_dict['start_time'] = raw_dict.pop('Start time', '')
    header_dict['end_time'] = raw_dict.pop('End time', '')
    header_dict['user'] = raw_dict.pop('User', 'User')
    header_dict['comment'] = raw_dict.pop('Comment', '')

    # Add all remaining parameters to header_dict unchanged
    header_dict.update(raw_dict)

    return header_dict


def _parse_sxm_header(header_raw):
    """
    Parse raw header string.

    Empirically done based on Nanonis header structure. See Scan
    docstring or Nanonis help documentation for more details.

    Parameters
    ----------
    header_raw : str
        Raw header string from read_raw_header() method.

    Returns
    -------
    dict
        Channel name keyed dict of each channel array.
    """
    header_entries = header_raw.split('\n')
    header_entries = header_entries[:-3]

    header_dict = dict()
    entries_to_be_split = ['scan_offset',
                           'scan_pixels',
                           'scan_range',
                           'scan_time']

    entries_to_be_floated = ['scan_offset',
                             'scan_range',
                             'scan_time',
                             'bias',
                             'acq_time']

    entries_to_be_inted = ['scan_pixels']

    for i, entry in enumerate(header_entries):
        if entry == ':DATA_INFO:' or entry == ':Z-CONTROLLER:':
            count = 1
            for j in range(i + 1, len(header_entries)):
                if header_entries[j].startswith(':'):
                    break
                if header_entries[j][0] == '\t':
                    count += 1
            header_dict[entry.strip(':').lower()] = _parse_scan_header_table(header_entries[i + 1:i + count])
            continue
        if entry.startswith(':'):
            header_dict[entry.strip(':').lower()] = header_entries[i + 1].strip()

    for key in entries_to_be_split:
        header_dict[key] = header_dict[key].split()

    for key in entries_to_be_floated:
        if isinstance(header_dict[key], list):
            header_dict[key] = np.asarray(header_dict[key], dtype=np.float)
        else:
            header_dict[key] = np.float(header_dict[key])
    for key in entries_to_be_inted:
        header_dict[key] = np.asarray(header_dict[key], dtype=np.int)

    return header_dict


def _parse_dat_header(header_raw):
    """
    Parse point spectroscopy header.

    Each key-value pair is separated by '\t' characters. Values may be
    further delimited by more '\t' characters.

    Returns
    -------
    dict
        Parsed point spectroscopy header.
    """
    header_entries = header_raw.split('\r\n')
    header_entries = header_entries[:-3]
    header_dict = dict()
    for entry in header_entries:
        key, val, _ = entry.split('\t')
        header_dict[key] = val

    return header_dict


def _clean_sxm_header(header_dict):
    """
    Cleanup header dicitonary key-value pairs.

    Parameters
    ----------
    header_dict : dict
        Should be dict returned from _parse_sxm_header method.

    Returns
    -------
    clean_header_dict : dict
        Cleaned header dictionary.
    """
    pass


def _split_header_entry(entry):
    """
    Split 3ds header entries by '=' character. If multiple values split
    those by ';' character.
    """

    key_str, val_str = entry.split("=", 1)

    if ';' in val_str:
        return key_str, (val_str.strip('"').split(';'))
    else:
        return key_str, val_str.strip('"')


def save_array(file, arr, allow_pickle=True):
    """
    Wrapper to numpy.save method for arrays.

    The idea would be to use this to save a processed array for later
    use in a matplotlib figure generation scripts. See numpy.save
    documentation for details.

    Parameters
    ----------
    file : file or str
        File or filename to which the data is saved.  If file is a file-
        object, then the filename is unchanged.  If file is a string, a
        ``.npy`` extension will be appended to the file name if it does
        not already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can
        execute arbitrary code) and portability (pickled objects may not
        be loadable on different Python installations, for example if
        the stored objects require libraries that are not available, and
        not all pickled data is compatible between Python 2 and Python
        3). Default: True
    """
    np.save(file, arr, allow_pickle=allow_pickle)


def load_array(file, allow_pickle=True):
    """
    Wrapper to numpy.load method for binary files.

    See numpy.load documentation for more details.

    Parameters
    ----------
    file : file or str
        The file to read. File-like objects must support the
    ``seek()`` and ``read()`` methods. Pickled files require that the
    file-like object support the ``readline()`` method as well.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons
        for disallowing pickles include security, as loading pickled
        data can execute arbitrary code. If pickles are disallowed,
        loading object arrays will fail. Default: True

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned
        instance of NpzFile class must be closed to avoid leaking file
        descriptors.
    """
    return np.load(file)


def _parse_scan_header_table(table_list):
    """
    Parse scan file header entries whose values are tab-separated
    tables.
    """
    table_processed = []
    for row in table_list:
        # strip leading \t, split by \t
        table_processed.append(row.strip('\t').split('\t'))

    # column names are first row
    keys = table_processed[0]
    values = table_processed[1:]

    zip_vals = zip(*values)

    return dict(zip(keys, zip_vals))


def _is_valid_file(fname, ext):
    """
    Detect if invalid file is being initialized by class.
    """
    if fname[-3:] != ext:
        raise UnhandledFileError('{} is not a {} file'.format(fname, ext))
