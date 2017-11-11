import unittest
import tempfile
import os
import numpy as np

import nanonispy as nap

class TestNanonisFileBaseClass(unittest.TestCase):
    """
    Testing class for NanonisFile base class.
    """
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_is_instance_nanonis_file(self):
        """
        Check for correct instance of NanonisFile object.
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.3ds',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b':HEADER_END:')
        f.close()
        NF = nap.read.NanonisFile(f.name)

        self.assertIsInstance(NF, nap.read.NanonisFile)


    def test_unsupported_filetype(self):
        """
        Handle unsupported file gracefully.
        """
        with self.assertRaises(nap.read.UnhandledFileError):
            f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.txt',
                                        dir=self.temp_dir.name,
                                        delete=False)
            f.close()
            NF = nap.read.NanonisFile(f.name)

    def test_3ds_suffix_parsed(self):
        """
        3ds file recognized.
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.3ds',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b':HEADER_END:')
        f.close()
        NF = nap.read.NanonisFile(f.name)
        self.assertEqual(NF.filetype, 'grid')

    def test_sxm_suffix_parsed(self):
        """
        Sxm file recognized.
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.sxm',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'SCANIT_END')
        f.close()
        NF = nap.read.NanonisFile(f.name)
        self.assertEqual(NF.filetype, 'scan')

    def test_dat_suffix_parsed(self):
        """
        Dat file recognized.
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.dat',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'[DATA]')
        f.close()
        NF = nap.read.NanonisFile(f.name)
        self.assertEqual(NF.filetype, 'spec')

    def test_find_start_byte(self):
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.3ds',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'header_entry\n:HEADER_END:\n')
        f.close()

        NF = nap.read.NanonisFile(f.name)
        byte_offset = NF.start_byte()

        self.assertEqual(byte_offset, 26)

    def test_no_header_tag_found(self):
        with self.assertRaises(nap.read.FileHeaderNotFoundError):
            f = tempfile.NamedTemporaryFile(mode='wb',
                                            suffix='.3ds',
                                            dir=self.temp_dir.name,
                                            delete=False)
            f.close()
            NF = nap.read.NanonisFile(f.name)

    def test_header_raw_is_str(self):
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix='.3ds',
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'header_entry\n:HEADER_END:\n')
        f.close()

        NF = nap.read.NanonisFile(f.name)
        self.assertIsInstance(NF.header_raw, str)


class TestGridFile(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_dummy_grid_data(self, suffix='3ds'):
        """
        return tempfile file object with dummy header info
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix=suffix,
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'Grid dim="230 x 230"\r\nGrid settings=4.026839E-8;-4.295725E-8;1.500000E-7;1.500000E-7;0.000000E+0\r\nSweep Signal="Bias (V)"\r\nFixed parameters="Sweep Start;Sweep End"\r\nExperiment parameters="X (m);Y (m);Z (m);Z offset (m);Settling time (s);Integration time (s);Z-Ctrl hold;Final Z (m)"\r\n# Parameters (4 byte)=10\r\nExperiment size (bytes)=2048\r\nPoints=512\r\nChannels="Input 3 (A)"\r\nDelay before measuring (s)=0.000000E+0\r\nExperiment="Grid Spectroscopy"\r\nStart time="21.10.2014 16:48:06"\r\nEnd time="23.10.2014 10:42:19"\r\nUser=\r\nComment=\r\n:HEADER_END:\r\n')
        a = np.linspace(0, 100.0, 230*230*(10+512))
        b = np.asarray(a, dtype='>f4')
        b.tofile(f)
        f.close()

        return f

    def create_dummy_grid_data_v2(self, suffix='3ds'):
        """
        return tempfile file object with dummy header info
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix=suffix,
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b'Grid dim="230 x 230"\r\nGrid settings=4.026839E-8;-4.295725E-8;1.500000E-7;1.500000E-7;0.000000E+0\r\nFiletype=Linear\r\nSweep Signal="Bias (V)"\r\nFixed parameters="Sweep Start;Sweep End"\r\nExperiment parameters="X (m);Y (m);Z (m);Z offset (m);Settling time (s);Integration time (s);Z-Ctrl hold;Final Z (m)"\r\n# Parameters (4 byte)=10\r\nExperiment size (bytes)=2048\r\nPoints=512\r\nChannels="Input 3 (A)"\r\nDelay before measuring (s)=0.000000E+0\r\nExperiment="Grid Spectroscopy"\r\nStart time="21.10.2014 16:48:06"\r\nEnd time="23.10.2014 10:42:19"\r\nUser=\r\nComment=\r\n:HEADER_END:\r\n')
        a = np.linspace(0, 100.0, 230*230*(10+512))
        b = np.asarray(a, dtype='>f4')
        b.tofile(f)
        f.close()

        return f

    def test_is_instance_grid_file(self):
        """
        Check for correct instance of Grid object.
        """
        f = self.create_dummy_grid_data()
        GF = nap.read.Grid(f.name)

        self.assertIsInstance(GF, nap.read.Grid)

    def test_data_has_right_shape(self):
        f = self.create_dummy_grid_data()
        GF = nap.read.Grid(f.name)

        self.assertEqual(GF.signals['Input 3 (A)'].shape, (230, 230, 512))

    def test_sweep_signal_calculated(self):
        f = self.create_dummy_grid_data()
        GF = nap.read.Grid(f.name)

        self.assertEqual(GF.signals['sweep_signal'].shape, (512,))

    def test_raises_correct_instance_error(self):
        with self.assertRaises(nap.read.UnhandledFileError):
            f = self.create_dummy_grid_data(suffix='sxm')
            GF = nap.read.Grid(f.name)

    def test_header_entries(self):
        f = self.create_dummy_grid_data()
        GF = nap.read.Grid(f.name)

        test_dict = {'angle': '0.0',
                     'channels': "['Input 3 (A)']",
                     'comment': '',
                     'dim_px': '[230, 230]',
                     'end_time': '23.10.2014 10:42:19',
                     'experiment_name': 'Grid Spectroscopy',
                     'experiment_size': '2048',
                     'experimental_parameters': "['X (m)', 'Y (m)', 'Z (m)', 'Z offset (m)', 'Settling time (s)', 'Integration time (s)', 'Z-Ctrl hold', 'Final Z (m)']",
                     'fixed_parameters': "['Sweep Start', 'Sweep End']",
                     'measure_delay': '0.0',
                     'num_channels': '1',
                     'num_parameters': '10',
                     'num_sweep_signal': '512',
                     'pos_xy': '[4.026839e-08, -4.295725e-08]',
                     'size_xy': '[1.5e-07, 1.5e-07]',
                     'start_time': '21.10.2014 16:48:06',
                     'sweep_signal': 'Bias (V)',
                     'user': ''}
        for key in GF.header:
            a = ''.join(sorted(str(GF.header[key])))
            b = ''.join(sorted(test_dict[key]))
            self.assertEqual(a, b)

    def test_both_header_formats(self):
        f = self.create_dummy_grid_data()
        f2 = self.create_dummy_grid_data_v2()
        GF = nap.read.Grid(f.name)
        GF2 = nap.read.Grid(f2.name)

        self.assertEqual(GF.header, GF2.header)


class TestScanFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_dummy_scan_data(self, suffix='.sxm'):
        """
        return tempfile file object with dummy header info
        """
        f = tempfile.NamedTemporaryFile(mode='wb',
                                        suffix=suffix,
                                        dir=self.temp_dir.name,
                                        delete=False)
        f.write(b':NANONIS_VERSION:\n2\n:SCANIT_TYPE:\n              FLOAT            MSBFIRST\n:REC_DATE:\n 21.11.2014\n:REC_TIME:\n17:19:32\n:REC_TEMP:\n      290.0000000000\n:ACQ_TIME:\n       470.3\n:SCAN_PIXELS:\n       64       64\n:SCAN_FILE:\nC:\\STM data\\2014-11\\2014-11-21\\ScanAg111_November2014_001.sxm\n:SCAN_TIME:\n             3.533E+0             3.533E+0\n:SCAN_RANGE:\n           1.500000E-7           1.500000E-7\n:SCAN_OFFSET:\n             7.217670E-8         2.414175E-7\n:SCAN_ANGLE:\n            0.000E+0\n:SCAN_DIR:\nup\n:BIAS:\n            -5.000E-2\n:Z-CONTROLLER:\n\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const\n\tCurrent #3\t1\t1.000E-10 A\t7.000E-12 m\t3.500E-9 m/s\t2.000E-3 s\n:COMMENT:\n\n:NanonisMain>Session Path:\nC:\\STM data\\2014-11\\2014-11-21\n:NanonisMain>SW Version:\nGeneric 4\n:NanonisMain>UI Release:\n3180\n:NanonisMain>RT Release:\n3130\n:NanonisMain>RT Frequency (Hz):\n5E+3\n:NanonisMain>Signals Oversampling:\n10\n:NanonisMain>Animations Period (s):\n20E-3\n:NanonisMain>Indicators Period (s):\n300E-3\n:NanonisMain>Measurements Period (s):\n500E-3\n:DATA_INFO:\n\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n\t14\tZ\tm\tboth\t-3.480E-9\t0.000E+0\n\t2\tInput_3\tA\tboth\t1.000E-9\t0.000E+0\n\t20\tLIX_1_omega\tA\tboth\t1.000E+0\t0.000E+0\n\t21\tLIY_1_omega\tA\tboth\t1.000E+0\t0.000E+0\n\n:SCANIT_END:\n')
        a = np.linspace(0, 100.0, 1+4*2*64*64)
        b = np.asarray(a, dtype='>f4')
        b.tofile(f)
        f.close()

        return f

    def test_header_entries(self):
        f = self.create_dummy_scan_data()
        SF = nap.read.Scan(f.name)

        test_dict = {'acq_time': '470.3',
                     'bias': '-0.05',
                     'comment': '',
                     'data_info': "{'Channel': ('14', '2', '20', '21'), 'Unit': ('m', 'A', 'A', 'A'), 'Direction': ('both', 'both', 'both', 'both'), 'Offset': ('0.000E+0', '0.000E+0', '0.000E+0', '0.000E+0'), 'Name': ('Z', 'Input_3', 'LIX_1_omega', 'LIY_1_omega'), 'Calibration': ('-3.480E-9', '1.000E-9', '1.000E+0', '1.000E+0')}",
                     'nanonis_version': '2',
                     'nanonismain>animations period (s)': '20E-3',
                     'nanonismain>indicators period (s)': '300E-3',
                     'nanonismain>measurements period (s)': '500E-3',
                     'nanonismain>rt frequency (hz)': '5E+3',
                     'nanonismain>rt release': '3130',
                     'nanonismain>session path': 'C:\\STM data\\2014-11\\2014-11-21',
                     'nanonismain>signals oversampling': '10',
                     'nanonismain>sw version': 'Generic 4',
                     'nanonismain>ui release': '3180',
                     'rec_date': '21.11.2014',
                     'rec_temp': '290.0000000000',
                     'rec_time': '17:19:32',
                     'scan_angle': '0.000E+0',
                     'scan_dir': 'up',
                     'scan_file': 'C:\\STM data\\2014-11\\2014-11-21\\ScanAg111_November2014_001.sxm',
                     'scan_offset': '[  7.21767000e-08   2.41417500e-07]',
                     'scan_pixels': '[64 64]',
                     'scan_range': '[  1.50000000e-07   1.50000000e-07]',
                     'scan_time': '[ 3.533  3.533]',
                     'scanit_type': 'FLOAT            MSBFIRST',
                     'z-controller': "{'P-gain': ('7.000E-12 m',), 'Setpoint': ('1.000E-10 A',), 'on': ('1',), 'T-const': ('2.000E-3 s',), 'Name': ('Current #3',), 'I-gain': ('3.500E-9 m/s',)}"}

        for key in SF.header:
            a = ''.join(sorted(str(SF.header[key])))
            b = ''.join(sorted(test_dict[key]))
            self.assertEqual(a, b)

    def test_raises_correct_instance_error(self):
        with self.assertRaises(nap.read.UnhandledFileError):
            f = self.create_dummy_scan_data(suffix='.3ds')
            SF = nap.read.Scan(f.name)

class TestSpecFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_dummy_spec_data(self, suffix='dat'):
        base = os.path.dirname(__file__)
        f = open(base+'/Bias-Spectroscopy002.dat', 'rb')
        f.close()
        return f

    def test_header_entries(self):
        f = self.create_dummy_spec_data()
        SP = nap.read.Spec(f.name)

        test_dict = {'Cutoff frq': '',
                     'Date': '04.08.2015 08:49:41',
                     'Experiment': 'bias spectroscopy',
                     'Filter type': 'Gaussian',
                     'Final Z (m)': 'N/A',
                     'Integration time (s)': '200E-6',
                     'Order': '6',
                     'Settling time (s)': '200E-6',
                     'User': '',
                     'X (m)': '-19.4904E-9',
                     'Y (m)': '-73.1801E-9',
                     'Z (m)': '-13.4867E-9',
                     'Z offset (m)': '-250E-12',
                     'Z-Ctrl hold': 'TRUE'}

        for key in SP.header:
            a = ''.join(sorted(str(SP.header[key])))
            b = ''.join(sorted(test_dict[key]))
            self.assertEqual(a, b)


class TestUtilFunctions(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_arr_roundtrip(self):
        fname = self.temp_dir.name + '/test_roundtrip.npy'
        a = np.linspace(0, 1.00, dtype='>f4')
        nap.read.save_array(fname, a)
        b = nap.read.load_array(fname)

        np.testing.assert_array_equal(a, b)

if __name__ == '__main__':
    unittest.main()
