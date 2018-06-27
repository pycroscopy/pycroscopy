if __name__ == "__main__":
    import pycroscopy.io.translators.df_utils.gsf_read as gsf_read
    import pycroscopy.io.translators.gwyddion as gwy
    from pycroscopy.core.io.write_utils import write_dset_to_txt
    from pycroscopy.core.io.pycro_data import PycroDataset
    import h5py
    import pycroscopy as px
    import matplotlib.pyplot as plt
    import numpy as np
    input_file_gsf = './data/SingFreqPFM_0003.gwy'
    # input_file_gwy = './data/131017Spectroscopy002.gwy'
    gwy = gwy.GwyddionTranslator()
    h5_path = gwy.translate(input_file_gsf)
    # h5_file = h5py.File(h5_from_gwy)
    """
    with h5py.File(h5_path, 'r') as h5_file:
        px.hdf_utils.print_tree(h5_file)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        for axis, dset in zip(axes.flat, px.hdf_utils.get_all_main(h5_file)):
            print(dset.get_n_dim_form)
            axis.imshow(np.squeeze(dset.get_n_dim_form()))
            axis.set_title(dset.data_descriptor)
        plt.show()
    """
    with h5py.File(h5_path, 'r') as h5_file:
        px.hdf_utils.print_tree(h5_file)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        for axis, dset in zip(axes.flat, px.hdf_utils.get_all_main(h5_file)):
            axis.imshow(np.squeeze(dset.get_n_dim_form()))
            axis.set_title(dset.data_descriptor)
        fig.tight_layout() 
        plt.show()
    # for each in h5_file.attrs:
        # print(each, h5_file.attrs[each])
    # px.hdf_utils.print_tree(h5_file)
    # pdRaw = px.PycroDataset(h5_file['Measurement_000/Channel_000/Raw_Data'])
    # write_dset_to_txt(pdRaw)
