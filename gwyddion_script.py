if __name__ == "__main__":
    import pycroscopy.io.translators.df_utils.gsf_read as gsf_read
    import pycroscopy.io.translators.gwyddion as gwy
    from pycroscopy.core.io.write_utils import write_dset_to_txt
    from pycroscopy.core.io.pycro_data import PycroDataset
    import h5py
    input_file_gsf = './data/chip.gsf'
    # input_file_gwy = './data/SingFreqPFM_0003.gwy'
    gsf = gwy.GwyddionTranslator()
    h5_from_gsf = gsf.translate(input_file_gsf)
    h5 = h5py.File(h5_from_gsf)
    pd = PycroDataset(h5['Measurement_000/Channel_000/Raw_Data'])
    w_u = write_dset_to_txt(pd)

    
    