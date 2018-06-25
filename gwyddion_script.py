if __name__ == "__main__":
    import pycroscopy.io.translators.df_utils.gsf_read as gsf_read
    import pycroscopy.io.translators.gwyddion as gwy
    from pycroscopy.core.io.write_utils import write_dset_to_txt
    from pycroscopy.core.io.pycro_data import PycroDataset
    import h5py
    import pycroscopy as px
    # input_file_gsf = './data/chip.gsf'
    input_file_gwy = './data/131017Spectroscopy002.gwy'
    gwy = gwy.GwyddionTranslator()
    h5_from_gwy = gwy.translate(input_file_gwy)
    h5_file = h5py.File(h5_from_gwy)
    for each in h5_file.attrs:
        print(each, h5_file.attrs[each])
    # px.hdf_utils.print_tree(h5_file)
    # pdRaw = px.PycroDataset(h5_file['Measurement_000/Channel_000/Raw_Data'])
    # write_dset_to_txt(pdRaw)
    
    