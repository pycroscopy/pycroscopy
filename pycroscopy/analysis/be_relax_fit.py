import numpy as np
import pyUSID as usid
from scipy.optimize import curve_fit

def exp(x, a, k, c):
    return (a * np.exp(-(x/k))) + c

def fit_exp_curve(x, y):
    popt, _ = curve_fit(exp, x, y, maxfev=25000)
    return popt

def double_exp(x, a, k, a2, k2, c):
    return (a * np.exp(-k*x)) + (a2 * np.exp(-k2*x) + c )

def fit_double_exp(x, y):
    """
    Fit spectrum,y, to double exp using differential evolution
    :param x: values for x-axis
    :param y: values for y-axis
    :return: best fit parameters for a double exp.
    """
    time_ax = x
    spectrum = y
    def cost_func_double_exp(params):
        a = params[0]; k = params[1]; a2 = params[2]; k2 = params[3]; c = params[4]
        double_exp_model = double_exp(time_ax, a, k, a2, k2, c)
        return np.sum((spectrum - double_exp_model)**2)
    popt = differential_evolution(cost_func_double_exp,
                                  bounds=([-100,100],[-100, 100],[-200,200],[-100,100],[-200,200])).x
    return popt

def str_exp(x, a, k, c):
    return a * np.exp(x ** k) + c

def fit_str_exp(x,y):
    popt, _ = curve_fit(str_exp, x, y, maxfev=25000, bounds=([-np.inf,0,-np.inf], [np.inf,1,np.inf]))
    return popt

def sigmoid(x, A, K, B, v, Q, C):
    return A + (K-A)/(C+Q*np.exp(-B*x)**(1/v))

def fit_sigmoid(x, y):
    popt, pcov = curve_fit(sigmoid, x, y, maxfev=2500)
    return popt

class BERelaxFit(usid.Process):
    def __init__(self, h5_main, variables=None, fit_method='Exponential', sens=1, phase_off=0,
                 starts_with='write', **kwargs):
        """
        This instantiation reads and calculates parameters in the data file necessary for reading, writing, analyzing,
        and visualizing the data. It writes these parameters to attributes to be referenced.

        :param h5_main: h5py.Dataset object from pycroscopy.analysis.BESHOfitter
        :param variables: list(string), Default ['Frequency']
        Lists of attributes that h5_main should possess so that it may be analyzed by Model.
        :param fit_method: fit_method for berelaxfit fit, can be 'Exponential', 'Double_Exp', 'Str_Exp' or 'Logistic'
        :param sens: tip sensitivity in pm/V. Default: 1, the data are not scaled
        :param phase_off: to apply to phase data. Default: 0, the data are not offset.
        :param starts_with: 'write' or 'read' , depending on whether the first step is a read or write step. Default:
        'write'

        **Currently, the BE software does not consistently encode whether spectra start with a read or write step
        """
        if h5_main == None:
            h5_main = self.h5_main
        super(BERelaxFit, self).__init__(h5_main, variables, **kwargs)
        self.starts_with = starts_with
        self.raw_data = h5_main.parent.parent['Raw_Data']
        self.raw_amp = np.abs(self.raw_data)
        self.raw_phase = np.angle(self.raw_data)
        self.h5_main_usid = usid.USIDataset(h5_main)
        self.raw_amp_reshape = self.raw_amp.reshape(self.h5_main_usid.pos_dim_sizes[0],
                                                    self.h5_main_usid.pos_dim_sizes[1],
                                                    h5_main.parent.parent.parent.attrs['num_steps'],-1)
        self.raw_phase_reshape = self.raw_phase.reshape(self.h5_main_usid.pos_dim_sizes[0],
                                                        self.h5_main_usid.pos_dim_sizes[1],
                                                        h5_main.parent.parent.parent.attrs['num_steps'],-1)
        self.fit_method = fit_method
        self.no_read_steps = self.h5_main.parent.parent.parent.attrs['VS_num_meas_per_read_step']
        self.no_write_steps = self.h5_main.parent.parent.parent.attrs['VS_num_meas_per_write_step']
        self.sensitivity = sens
        self.phase_offset = phase_off
        self.no_time_steps = self.h5_main.parent.parent.parent.attrs['num_steps']
        self.time_elapsed_per_step = self.h5_main.parent.parent.parent.attrs['BE_pulse_duration_[s]']
        self.time_elapsed_per_spectrum = (self.no_read_steps) * self.time_elapsed_per_step
        self.all_dc_offset_values = self.h5_main.h5_spec_vals[1,np.argwhere(self.h5_main.h5_spec_inds[0]==0)]
        self.dc_offset_expand = self.h5_main.h5_spec_vals[1,:]
        #make list of indices of read/write steps
        self.no_rs_spectra = int(len(np.argwhere(self.h5_main.h5_spec_inds[0, :] == 0)) / 2)
        self.read_inds_split = []
        self.write_inds_split = []
        self.all_inds_split = np.array_split(np.arange(0, self.no_time_steps, step=1), self.no_rs_spectra)
        self.write_spectra = []
        if self.starts_with == 'write':
            for i in range(self.no_rs_spectra):
                self.read_inds_split.append(self.all_inds_split[i][self.no_write_steps:])
                self.write_dc_offset_values = self.all_dc_offset_values[::2]

                #if there is only one RS spectrum
                if type(self.write_dc_offset_values) == np.float32:
                    self.write_dc_offset_values = [self.write_dc_offset_values]

        if self.starts_with == 'read':
            for i in range(self.no_rs_spectra):
                self.read_inds_split.append(self.all_inds_split[i][:-int(self.no_write_steps)])
                self.write_dc_offset_values = self.h5_main.h5_spec_vals[1,
                                                                        np.argwhere(self.h5_main.h5_spec_vals[
                                                                                        0] == self.no_read_steps)]
                # if there is only one RS spectrum
                if type(self.write_dc_offset_values) == np.float32:
                    self.write_dc_offset_values = [self.write_dc_offset_values]

        self.no_read_offset = len(self.all_dc_offset_values) - self.no_rs_spectra
        self.write_inds_split = np.split(np.setxor1d(self.all_inds_split, self.read_inds_split),
                                                       self.no_rs_spectra)

    def _map_function(self, spectra, *args, **kwargs):
        """

        Function that manipulates the data in a single pixel. This is used by self.test.
        :param spectra: relaxation spectrum to fit
        :return: array containing optimal fit values for the parameters
        """
        x = np.arange(0, self.time_elapsed_per_spectrum, step=self.time_elapsed_per_step)
        y = spectra
        if self.fit_method == 'Exponential':
            scalar = 1000
            popt_init = fit_exp_curve(x * scalar, y)
            a_init = popt_init[0];
            tau_init = popt_init[1] / scalar;
            c_init = popt_init[2]
            popt, _ = curve_fit(exp, x, y, maxfev=2500, p0=[a_init, tau_init, c_init])
        if self.fit_method == 'Double_Exp':
            popt = fit_double_exp(x, y)
        if self.fit_method == 'Str_Exp':
            popt = fit_str_exp(x, y)
        if self.fit_method == 'Logistic':
            popt = fit_sigmoid(x, y)
        return popt

    def test(self, pixel_ind):
        """
        This function manipulates the data within one pixel as defined by _map_function.
        :param pixel_ind: pixel index of the data we wish to test
        :return: array from fit of data to self.fit_method
        """
        amplitude_to_reshape = self.h5_main['Amplitude [V]'][pixel_ind, :]
        phase_to_reshape = self.h5_main['Phase [rad]'][pixel_ind, :]
        mixed_signal = []
        for j in range(self.no_read_offset):
            mixed_signal.append(amplitude_to_reshape[self.read_inds_split[j]] *
                                np.cos(phase_to_reshape[self.read_inds_split[j]]))
        spectra = mixed_signal
        return self._map_function(spectra)

    def _read_data_chunk(self):
        """
        Reads and loads relaxation spectroscopy data files from V3 beta 2 acquisition software on Cypher into self.data
        """
        super(BERelaxFit, self)._read_data_chunk()
        if self._start_pos < self.h5_main.shape[0]:
            # The above line makes the base Process class read X pixels from the data set into self.data
            amplitude_to_reshape = self.data['Amplitude [V]']
            phase_to_reshape = self.data['Phase [rad]']
            amplitude_reshaped = []
            phase_reshaped = []
            amplitude_write = []
            phase_write = []
            mixed_signal = []
            mixed_signal_write = []
            for i in range(self.h5_main.shape[0]):
                for j in range(self.no_read_offset):
                    amplitude_reshaped.append(amplitude_to_reshape[i, self.read_inds_split[j]])
                    phase_reshaped.append(phase_to_reshape[i, self.read_inds_split[j]])
                    mixed_signal = np.array([amp*self.sensitivity*np.cos(phase + self.phase_offset)
                                             for amp, phase in zip(amplitude_reshaped, phase_reshaped)])
                    #amplitude, phase, and mixed signal for write steps
                    if self.starts_with == 1:
                        amplitude_write.append(amplitude_to_reshape[i, :self.no_write_steps])
                        phase_write.append(phase_to_reshape[i, :self.no_write_steps])
                        mixed_signal_write = np.array([amp*self.sensitivity*np.cos(phase + self.phase_offset)
                                                 for amp, phase in zip(amplitude_write, phase_write)])
                    if self.starts_with == 0:
                        amplitude_write.append(amplitude_to_reshape[i, self.no_read_steps:])
                        phase_write.append(phase_to_reshape[i, self.no_read_steps:])
                        mixed_signal_write = np.array([amp * self.sensitivity * np.cos(phase + self.phase_offset)
                                                       for amp, phase in zip(amplitude_write, phase_write)])

            self.data = np.array(mixed_signal)
            self.phase = np.array(phase_reshaped)
            self.amplitude = np.array(amplitude_reshaped)
            self.write_spectra = np.array(mixed_signal_write)

        else:
            self.data = None

    def _create_results_datasets(self):
        """
        Creates hdf5 datasets for to-be computed results, writes parameters to the datasets, and
        links ancillary datasets.
        """
        if self.fit_method == 'Exponential':
            self.process_name = 'Exp_Fit'
        if self.fit_method == 'Double_Exp':
            self.process_name = 'Double_Exp'
        if self.fit_method == 'Str_Exp':
            self.process_name = 'Str_Exp'
        if self.fit_method == 'Logistic':
            self.process_name = 'Logistic_Fit'
        # 1. make HDF5 group to hold results
        self.h5_results_grp = usid.hdf_utils.create_results_group(self.h5_main, self.process_name)
        if self.verbose:
            print('Results to be written to Group: {}'.format(self.h5_results_grp))
        # 2. write relevant meta data to group
        usid.hdf_utils.write_simple_attrs(self.h5_results_grp,
                                          {'last_pixel': 0, 'algorithm': str(self.fit_method)})
        # define all inputs to write_main_dataset
        # results shape
        results_shape = (self.h5_main.shape[0], self.no_read_offset)
        pos_dims = None
        spec_dims = usid.write_utils.Dimension('Bias', 'V', self.write_dc_offset_values)

        # 3. make empty hdf5 group to store fit information:
        if self.fit_method == 'Exponential':
            field_names = ['Amplitude [pm]', 'Time_Constant [s]', 'Offset [pm]']
            results_dset_name = 'Exponential_Fit'
            results_quantity = 'None'
            results_units = 'pm'

        if self.fit_method == 'Double_Exp':
            field_names = ['Amplitude [pm]', 'Time_Constant [s]',
                           'Amplitude 2 [pm]', 'Time_Constant 2 [s]', 'Offset [pm]']
            results_dset_name = 'Double_Exp_Fit'
            results_quantity = 'None'
            results_units = 'pm'

        if self.fit_method == 'Str_Exp':
            field_names = ['Amplitude [pm]', 'Beta', 'Offset [pm]']
            results_dset_name = 'Str_Exp_Fit'
            results_quantity = 'None'
            results_units = 'pm'


        if self.fit_method == 'Logistic':
            field_names = ['A', 'K', 'B', 'v', 'Q', 'C']
            results_dset_name = 'Logistic_Fit'
            results_quantity = 'None'
            results_units = 'pm'

        berelaxfit32 = np.dtype({'names': field_names,
                            'formats': [np.float32 for name in field_names]})
        self.h5_results = usid.hdf_utils.write_main_dataset(self.h5_results_grp, results_shape, results_dset_name,
                                                            results_quantity, results_units, pos_dims, spec_dims,
                                                            dtype=berelaxfit32, h5_pos_inds=self.h5_main.h5_pos_inds,
                                                            h5_pos_vals=self.h5_main.h5_pos_vals)
        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        """
        Extracts references to the existing datasets with results (if any)
        """
        if self.fit_method == 'Exponential' or 'Double_Exp':
            self.h5_amplitude = self. h5_results_grp['Amplitude [pm]']
            self.h5_time_const = self.h5_results_grp['Time_Constant [s]']
            self.h5_offset = self.h5_results_grp['Offset [pm']
        if self.fit_method == 'Double_Exp':
            self.h5_amplitude2 = self. h5_results_grp['Amplitude 2 [pm]']
            self.h5_time_const2 = self.h5_results_grp['Time_Constant 2 [pm]']
        if self.fit_method == 'Str_Exp':
            self.h5_amplitude = self.h5_results_grp['Amplitude [pm]']
            self.h5_beta = self.h5_results_grp['Beta']
            self.h5_offset = self.h5_results_grp['Offest [pm]']
        if self.fit_method == 'Logistic':
            self.h5_A = self.h5_results_grp['Amplitude [pm]']
            self.h5_K = self.h5_results_grp['A']
            self.h5_B = self.h5_results_grp['K']
            self.h5_v = self.h5_results_grp['v']
            self.h5_Q = self.h5_results['Q']
            self.h5_C = self.h5_results['C']

    def _write_results_chunk(self):
        """
        Writes computed results into appropriate datasets.
        """
        if self.fit_method == 'Exponential':
            field_names = ['Amplitude [pm]', 'Time_Constant [s]', 'Offset [pm]']
        if self.fit_method == 'Double_Exp':
            field_names = ['Amplitude [pm]', 'Time_Constant 1 [s]',
                           'Amplitude 2 [pm]', 'Time_Constant 2 [s]', 'Offset [pm]']
        if self.fit_method == 'Str_Exp':
            field_names = ['Amplitude [pm]', 'Beta', 'Offset [pm]']
        if self.fit_method == 'Logistic':
            field_names = ['A', 'K', 'B', 'v', 'Q', 'C']
        berelaxfit32 = np.dtype({'names': field_names,
                            'formats': [np.float32 for name in field_names]})
        # write and flush results
        results = usid.io.dtype_utils.stack_real_to_compound(self._results, compound_type=berelaxfit32)
        results = results.reshape(self.h5_results.shape[0], -1)
        pos_ind = slice(self._start_pos, self._end_pos)
        self.h5_results[pos_ind] = results[pos_ind]

        #if double, make amp1 < amp2:
        if self.fit_method == 'Double_Exp':
            import copy
            amp1_copy = copy.deepcopy(self.h5_results['Amplitude [pm]'])
            amp2_copy = copy.deepcopy(self.h5_results['Amplitude 2 [pm]'])
            tau1_copy = copy.deepcopy(self.h5_results['Time_Constant [s]'])
            tau2_copy = copy.deepcopy(self.h5_results['Time_Constant 2 [s]'])
            for i in range(self.h5_results['Amplitude [pm]'].shape[0]):
                if self.h5_results['Amplitude [pm]'][i] > self.h5_results['Amplitude 2 [pm]'][i]:
                    self.h5_results['Amplitude [pm]'][i] = amp2_copy[i]
                    self.h5_results['Amplitude 2 [pm]'][i] = amp1_copy[i]
                    self.h5_results['Time_Constant [s]'][i] = tau2_copy[i]
                    self.h5_results['Time_Constant 2 [s]'][i] = tau1_copy[i]

        # update last_pixel and start position
        self.h5_results_grp.attrs['last_pixel'] = self._end_pos
        self._start_pos = self._end_pos
        self.h5_main.file.flush()

    def _get_existing_datasets(self):
        return