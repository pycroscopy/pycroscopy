# -*- coding: utf-8 -*-
"""
@author: Oleg Ovchinnikov
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import h5py as h5

def apply_select_channel(file_in_h5, img_num, channel_num):
    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_%02i" % (img_num, channel_num)
    image_path = "%s/Raw_Data" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img2)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]

    image_path = "/Frame_%04i/Channel_Current/Filter_Step_0000" % img_num

    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path] = img2
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "Original"
    h5_new_attrs["Number_Of_Variables"] = 0

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    image_path = "/Frame_%04i/Channel_Current" % img_num

    path_main = main_h5_handle.get(image_path)
    path_attrs = path_main

    channel_name = "Channel_%02i" % channel_num
    path_main = main_h5_handle.get(image_path)
    path_attrs = path_main.attrs
    path_attrs["Origin"] = current_ref
    path_attrs["Origin_Region"] = current_reg
    path_attrs["Origin_Name"] = channel_name

    main_h5_handle.close()
    return 1


def apply_wiener_filter(file_in_h5, img_num, filter_num):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img2)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]
    img = np.empty([max(posi_ind[:, 0]), max(posi_ind[:, 1])], dtype=h5_image.dtype)
    img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2), 0]

    # test shape doe wiener filter
    s = 1.1  # constant for now change latter

    size = img.shape
    h = np.empty(size)
    width = size[0]
    height = size[1]
    for x in range(0, width):
        for y in range(0, height):
            h[x, y] = (s / 2 * 3.141592) ** (-s * np.sqrt(x ** 2 + y ** 2))
    h = h / np.sum(np.sum(h))
    H = np.fft.fft2(h)
    K = np.linspace(.001, 1, 100)
    G = np.fft.fft2(img)
    errorV = np.empty([1, 100])
    errorV = errorV[0]
    for k1 in range(0, 100):
        W = np.conj(H) / (np.abs(H) ** 2 + K[k1])
        F = W * G
        R = np.abs(np.fft.ifft2(F))
        errorV[k1] = np.std(np.reshape(R, [1, width * height]))

    # use best option 
    minv = np.min(errorV)
    minl = np.where(errorV == minv)
    minl = minl[0]
    minl = minl[0]
    W = np.conj(H) / (np.abs(H) ** 2 + K[minl])
    F = W * G
    img = np.fft.ifft2(F)

    img = img.reshape(len(img2), 1)
    img = abs(img)
    img = np.real(img)

    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 2):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path] = img
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "Wiener"
    h5_new_attrs["Number_Of_Variables"] = 0

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return 1


def apply_gaussian_corr_filter(file_in_h5, img_num, filter_num, gauss_width, gauss_box_width):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img2)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]
    img = np.empty([max(posi_ind[:, 0]), max(posi_ind[:, 1])], dtype=h5_image.dtype)
    img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2), 0]

    gauss_cell = [1, 0, 0, gauss_width, gauss_width, 0]

    s1 = len(img[:, 0])
    s2 = len(img[0, :])
    new_deconv = np.zeros([s1, s2])

    for k1 in range(0, s1):
        for k2 in range(0, s2):
            x_min = max([-gauss_box_width, -k1])
            x_max = min([gauss_box_width, (s1 - k1 - 1)])
            y_min = max([-gauss_box_width, -k2])
            y_max = min([gauss_box_width, (s2 - k2 - 1)])
            [xx, yy] = np.meshgrid(range(y_min, y_max + 1), range(x_min, x_max + 1))
            gaus = fun_2d_gaussian(xx, yy, gauss_cell)
            temp = np.corrcoef(gaus.reshape([1, gaus.size]),
                               img[k1 + x_min:k1 + x_max + 1,
                               k2 + y_min:k2 + y_max + 1].reshape([1, gaus.size]))
            new_deconv[k1, k2] = temp[0, 1]

    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 2):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path] = new_deconv
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "2D Correlation"
    h5_new_attrs["Number_Of_Variables"] = 2
    h5_new_attrs["Variable_1_Name"] = "Filter width"
    h5_new_attrs["Variable_1_Value"] = gauss_box_width
    h5_new_attrs["Variable_2_Name"] = "Gassian width"
    h5_new_attrs["Variable_2_Value"] = gauss_width

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return 1


def fun_2d_gaussian(x, y, parm):
    import numpy as np

    amp = np.double(parm[0])

    x_cent = np.double(parm[1])
    y_cent = np.double(parm[2])

    x_wid = np.double(parm[3])
    y_wid = np.double(parm[4])

    ang = np.double(parm[5])

    a = ((np.cos(ang) ** 2) / (2 * x_wid ** 2)) + ((np.sin(ang) ** 2) / (2 * y_wid ** 2))
    b = -((np.sin(2 * ang)) / (4 * x_wid ** 2)) + ((np.sin(2 * ang)) / (4 * y_wid ** 2))
    c = ((np.sin(ang) ** 2) / (2 * x_wid ** 2)) + ((np.cos(ang) ** 2) / (2 * y_wid ** 2))

    gaussian = amp * (
        np.exp(-((a * (x - x_cent) ** 2) + (2 * b * (x - x_cent) * (y - y_cent)) + (c * (y - y_cent) ** 2))))

    return gaussian


def apply_invert_filter(file_in_h5, img_num, filter_num):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img)]

    m_img = img.mean()
    img = img - m_img
    img = -img
    img = img + m_img

    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 2):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    temp = 0
    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path] = img
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "invert z contrast"
    h5_new_attrs["Number_Of_Variables"] = 0

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return 1


def apply_find(file_path_h5, file_name_h5, file_path_png, file_name_png, filter_width, img_num, filter_num):
    image_path = "/Frame_%04i/Filtered_Data/Stack_0000" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    s = 1.1  # constant for now change latter

    size = img.shape
    s1 = size[0]
    s2 = size[1]

    mat_large = np.empty([s1 + 2 * filter_width + 1, s2 + 2 * filter_width + 1])
    mat_large[:, :] = np.inf

    for k1 in range(-filter_width, filter_width + 1):
        for k2 in range(-filter_width, filter_width + 1):
            mat_large[filter_width - k1:-(filter_width + k1) - 1,
                      filter_width - k2:-(filter_width + k2) - 1] = np.minimum(mat_large[filter_width - k1:
                                                                                         -filter_width - k1 - 1,
                                                                                         filter_width - k2:
                                                                                         -filter_width - k2 - 1],
                                                                                h5_image)

    deconv_mat_temp = mat_large[filter_width:len(mat_larg[1, :]) - filter_width,
                      filter_width:len(mat_larg[:, 1]) - filter_width]
    filtered_image = h5_image - deconv_mat_temp

    return 1


def apply_binarization_filter(file_in_h5, img_num, filter_num):
    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]

    i_max = img.max()
    i_min = img.min()
    i_diff = i_max - i_min

    max_r = 49
    filter_img = np.zeros([max_r + 1, len(img)])
    time_out = np.zeros([max_r + 1, 1])
    time_out_i = np.zeros([max_r + 1, 1])

    for x in range(0, max_r + 1):
        temp = np.zeros([len(img), 1])
        r = x / max_r
        temp[img > (i_min + (i_diff * r))] = 1
        filter_img[x, :] = temp[:, 0]
        time_out[x] = r
        time_out_i[x] = x + 1

    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 2):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path_f = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path_f] = img
    h5_image_new = main_h5_handle.get(image_path_f)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "Threshold_Reference_Image"
    h5_new_attrs["Number_Of_Variables"] = 0

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    image_path_sv = "%s/Spectroscopic_Values" % image_path
    main_h5_handle[image_path_sv] = time_out
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_sv_ref = h5_image_new.ref
    new_sv_reg = h5_image_new.regionref[0:len(time_out)]

    image_path_si = "%s/Spectroscopic_Indices" % image_path
    main_h5_handle[image_path_si] = time_out_i
    h5_image_new = main_h5_handle.get(image_path_si)
    new_si_ref = h5_image_new.ref
    new_si_reg = h5_image_new.regionref[0:len(time_out)]

    image_path_b = "%s/Binary_Matrix" % image_path
    main_h5_handle[image_path_b] = filter_img
    h5_image_new = main_h5_handle.get(image_path_b)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "Threshold_Matrix"
    h5_new_attrs["Number_Of_Variables"] = 0

    h5_new_attrs["Spectroscopic_Indices"] = new_si_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = new_si_reg
    h5_new_attrs["Spectroscopic_Values"] = new_sv_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = new_sv_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return 1


def apply_binarization_filter_select(file_in_h5, img_num, filter_num, threshold):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(img)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]

    i_max = img.max()
    i_min = img.min()
    i_diff = i_max - i_min

    max_r = 49.0
    filter_img = np.zeros([1, len(img)])
    temp = np.zeros([len(img), 1])
    r = threshold / max_r
    temp[img > (i_min + (i_diff * r))] = 1
    filter_img = temp[:, 0]

    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 2):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)

    try:
        main_h5_handle.__delitem__(image_path)
    except:
        temp = 1

    image_path = "%s/Filtered_Image" % image_path
    main_h5_handle[image_path] = filter_img
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "Threshold"
    h5_new_attrs["Number_Of_Variables"] = 1
    h5_new_attrs["Variable_1_Name"] = "Threshold"
    h5_new_attrs["Variable_1_Value"] = r

    h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
    h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
    h5_new_attrs["Position_Indices"] = pos_i_ref
    h5_new_attrs["Position_Indices_Region"] = pos_i_reg
    h5_new_attrs["Position_Values"] = pos_v_ref
    h5_new_attrs["Position_Values_Region"] = pos_v_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return


def cluster_into_atomic_columns(file_in_h5, img_num, filter_num, dist_val):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for ifilt in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, ifilt)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")
    current_ref_m = h5_image.ref
    current_reg_m = h5_image.regionref[0:len(img2)]

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]
    img = np.empty([max(posi_ind[:, 0]), max(posi_ind[:, 1])], dtype=h5_image.dtype)
    try:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2), 0]
    except:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2)]

    centers = cluster_2d_oleg_return_geo_center(img, dist_val)

    image_path_org = "/Frame_%04i/Channel_Current" % img_num
    image_path_new = "/Frame_%04i/Channel_Finished" % img_num

    try:
        main_h5_handle.__delitem__(image_path_new)
    except:
        temp = 1

    old_folder = main_h5_handle.get(image_path_org)
    main_h5_handle.create_group(image_path_new)
    new_folder = main_h5_handle.get(image_path_new)
    new_folder_attrs = new_folder.attrs

    parrent_ref = old_folder.attrs.get("Origin")
    parrent_reg = old_folder.attrs.get("Origin_Region")
    parrent_name = old_folder.attrs.get("Origin_Name")
    new_folder_attrs["Origin"] = parrent_ref
    new_folder_attrs["Origin_Region"] = parrent_reg
    new_folder_attrs["Origin_Name"] = parrent_name

    for ifilt in range(0, filter_num + 1):
        image_path_org = "%s/Filter_Step_%04i" % (image_path_org, ifilt)
        image_path_new = "%s/Filter_Step_%04i" % (image_path_new, ifilt)
        image_path_org_temp = "%s/Filtered_Image" % image_path_org
        image_path_new_temp = "%s/Filtered_Image" % image_path_new

        h5_image_old = main_h5_handle.get(image_path_org_temp)
        img_old = np.empty(h5_image_old.shape, dtype=h5_image_old.dtype)
        h5_image_old.read_direct(img_old)

        sec_i_ref = h5_image_old.attrs.get("Spectroscopic_Indices")
        sec_i_reg = h5_image_old.attrs.get("Spectroscopic_Indices_Region")
        sec_v_ref = h5_image_old.attrs.get("Spectroscopic_Values")
        sec_v_reg = h5_image_old.attrs.get("Spectroscopic_Values_Region")
        pos_i_ref = h5_image_old.attrs.get("Position_Indices")
        pos_i_reg = h5_image_old.attrs.get("Position_Indices_Region")
        pos_v_ref = h5_image_old.attrs.get("Position_Values")
        pos_v_rexg = h5_image_old.attrs.get("Position_Values_Region")
        filter_name = h5_image_old.attrs.get("Filter_Name")
        number_var = h5_image_old.attrs.get("Number_Of_Variables")

        main_h5_handle[image_path_new_temp] = img_old
        h5_image_new = main_h5_handle.get(image_path)
        h5_new_attrs = h5_image_new.attrs
        h5_new_attrs["Filter_Name"] = filter_name
        h5_new_attrs["Number_Of_Variables"] = number_var

        for ivar in range(1, number_var + 1):
            var_name = h5_image_old.attrs.get("Variable_%01i_Name" % ivar)
            var_value = h5_image_old.attrs.get("Variable_%01i_Value" % ivar)
            h5_new_attrs["Variable_%01i_Name" % ivar] = var_name
            h5_new_attrs["Variable_%01i_Value" % ivar] = var_value

        h5_new_attrs["Spectroscopic_Indices"] = sec_i_ref
        h5_new_attrs["Spectroscopic_Indices_Region"] = sec_i_reg
        h5_new_attrs["Spectroscopic_Values"] = sec_v_ref
        h5_new_attrs["Spectroscopic_Values_Region"] = sec_v_reg
        h5_new_attrs["Position_Indices"] = pos_i_ref
        h5_new_attrs["Position_Indices_Region"] = pos_i_reg
        h5_new_attrs["Position_Values"] = pos_v_ref
        h5_new_attrs["Position_Values_Region"] = pos_v_reg
        h5_new_attrs["Parent"] = parrent_ref
        h5_new_attrs["Parent_Region"] = parrent_reg

        parrent_ref = h5_image_new.ref
        parrent_reg = h5_image_new.regionref[0:len(img_old)]

    image_path = "%s/Lattice/Positions" % image_path_new
    main_h5_handle[image_path] = centers
    h5_image_new = main_h5_handle.get(image_path)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Method"] = "Density_Clustering"
    h5_new_attrs["Number_Of_Variables"] = 1
    h5_new_attrs["Variable_1_Name"] = "Connection_Distance"
    h5_new_attrs["Variable_1_Value"] = dist_val
    h5_new_attrs["Parent"] = current_ref_m
    h5_new_attrs["Parent_Region"] = current_reg_m

    main_h5_handle.close()


def cluster_2d_oleg(mat_in, dist_val):
    import numpy as np

    tt = [0, 0]
    clusters = []
    to_cluster = np.argwhere(mat_in)
    to_cluster = to_cluster.tolist()

    while len(to_cluster) > 0:
        clust = []
        final_clust = []
        clust.append(to_cluster[0])
        to_cluster.remove(to_cluster[0])
        while (len(clust) > 0) & (len(to_cluster) > 0):
            tt[0] = 5000.0 * dist_val
            tt[1] = len(to_cluster)
            t1 = min(tt)
            t1 = int(t1)
            to_cluster_t = to_cluster[0:t1]
            dem_diff = abs(to_cluster_t - np.tile(clust[0], [t1, 1]))
            diff_vec = np.argwhere(sum(dem_diff, 1) <= dist_val)
            for px in range(len(diff_vec), 0, -1):
                pt = diff_vec[px - 1]
                clust.append(to_cluster[pt])
                to_cluster.remove(to_cluster[pt])
            final_clust.append(clust[0])
            clust.remove(clust[0])

        if len(clust) > 0:
            while len(clust) < 0:
                final_clust.append(clust[0])
                clust.remove(clust[0])

        clusters.append(final_clust)

    return clusters


def cluster_2d_oleg_return_geo_center(mat_in, dist_val):
    import numpy as np

    tt = [0, 0]
    clusters = []
    to_cluster = np.argwhere(mat_in)
    to_cluster = to_cluster.tolist()

    while len(to_cluster) > 0:
        clust = []
        final_clust = []
        clust.append(to_cluster[0])
        to_cluster.remove(to_cluster[0])
        while (len(clust) > 0) & (len(to_cluster) > 0):
            tt[0] = 5000.0 * dist_val
            tt[1] = len(to_cluster)
            t1 = min(tt)
            t1 = int(t1)
            to_cluster_t = to_cluster[0:t1]
            dem_diff = abs(to_cluster_t - np.tile(clust[0], [t1, 1]))
            diff_vec = np.argwhere(np.sum(dem_diff, axis=1) <= dist_val)
            for px in range(len(diff_vec), 0, -1):
                pt = diff_vec[px - 1]
                clust.append(to_cluster[pt])
                to_cluster.remove(to_cluster[pt])
            final_clust.append(clust[0])
            clust.remove(clust[0])

        if len(clust) > 0:
            while len(clust) < 0:
                final_clust.append(clust[0])
                clust.remove(clust[0])

        if len(final_clust) > 2:
            clusters.append(np.mean(final_clust, 0))

    clusters = np.array(clusters)
    clusters = clusters.tolist()

    return clusters


def return_img(file_in_h5, img_num, filter_num):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Current" % img_num
    for x in range(0, filter_num + 1):
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    sec_v_ref = h5_image.attrs.get("Spectroscopic_Values")
    sec_v_reg = h5_image.attrs.get("Spectroscopic_Values_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")
    pos_v_ref = h5_image.attrs.get("Position_Values")
    pos_v_reg = h5_image.attrs.get("Position_Values_Region")

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]
    img = np.empty([max(posi_ind[:, 0]), max(posi_ind[:, 1])], dtype=h5_image.dtype)
    try:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2), 0]
    except:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2)]
    main_h5_handle.close()

    return img


def return_pos(file_in_h5, img_num):
    import numpy as np
    import h5py as h5

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Finished" % img_num
    type_ref = type(main_h5_handle.get(image_path))
    temp = 1
    x = -1
    while temp:
        x = x + 1
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
        image_temp = "%s/Lattice" % image_path
        temp_loc = main_h5_handle.get(image_temp)
        if type(temp_loc) == type_ref:
            temp = 0

    image_path = image_temp
    image_path = "%s/Positions" % image_path

    h5_image = main_h5_handle.get(image_path)
    pos = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(pos)

    return pos


def run_PCA_atoms(file_in_h5, img_num, box_width):
    import numpy as np
    import h5py as h5
    from scipy import linalg

    main_h5_handle = h5.File(file_in_h5, 'r+')
    image_path = "/Frame_%04i/Channel_Finished" % img_num
    type_ref = type(main_h5_handle.get(image_path))
    temp = 1
    x = -1
    while temp:
        x = x + 1
        image_path = "%s/Filter_Step_%04i" % (image_path, x)
        image_temp = "%s/Lattice" % image_path
        temp_loc = main_h5_handle.get(image_temp)
        if type(temp_loc) == type_ref:
            temp = 0

    pos_path = image_temp
    image_path = "%s/Positions" % pos_path

    h5_image = main_h5_handle.get(image_path)
    pos = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(pos)
    current_ref = h5_image.ref
    current_reg = h5_image.regionref[0:len(pos), 0:len(pos[0])]

    image_path = "/Frame_%04i/Channel_Current" % img_num
    image_path = "%s/Filter_Step_%04i" % (image_path, 0)
    image_path = "%s/Filtered_Image" % image_path

    h5_image = main_h5_handle.get(image_path)
    img2 = np.empty(h5_image.shape, dtype=h5_image.dtype)
    h5_image.read_direct(img2)

    sec_i_ref = h5_image.attrs.get("Spectroscopic_Indices")
    sec_i_reg = h5_image.attrs.get("Spectroscopic_Indices_Region")
    pos_i_ref = h5_image.attrs.get("Position_Indices")
    pos_i_reg = h5_image.attrs.get("Position_Indices_Region")

    spec_ind = main_h5_handle[sec_i_ref]
    spec_ind = spec_ind[sec_i_reg]

    posi_ind = main_h5_handle[pos_i_ref]
    posi_ind = posi_ind[pos_i_reg]
    img = np.empty([max(posi_ind[:, 0]), max(posi_ind[:, 1])], dtype=h5_image.dtype)
    try:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2), 0]
    except:
        img[posi_ind[:, 0] - 1, posi_ind[:, 1] - 1] = img2[0:len(img2)]

    sel_vec = np.zeros([len(pos[:, 0]), 1])
    kk = 0
    new_pos = []
    img_vectors = []
    for k1 in range(0, len(pos[:, 0])):
        if (box_width <= pos[k1, 0].round() <= len(img[:, 0]) - box_width and
                        box_width <= pos[k1, 1].round() <= len(img[0, :]) - box_width):
            sel_vec[k1] = 1
            new_pos.append(pos[k1, :])
            vector = img[pos[k1, 0] - box_width:pos[k1, 0] + box_width, pos[k1, 1] - box_width:pos[k1, 1] + box_width]
            img_vectors.append(vector.reshape([(box_width * 2) ** 2]))

    new_pos = array(new_pos)
    img_vectors = array(img_vectors)

    U, S, V = linalg.svd(img_vectors)
    V = V[0:len(S)]

    temp = 1
    x = -1

    while temp:
        x = x + 1
        image_temp = "%s/Analysis_%04i" % (pos_path, x)
        temp_loc = main_h5_handle.get(image_temp)
        if type(temp_loc) != type_ref:
            temp = 0

    image_path = image_temp

    [xx, yy] = np.meshgrid(range(1, len(U) + 1), range(1, len(U[0]) + 1))
    xx = np.reshape(xx, [1, len(U) * len(U[0])])
    yy = np.reshape(yy, [1, len(U) * len(U[0])])
    xy = [xx, yy]
    U = np.reshape(U, [1, len(U) * len(U[0])])

    image_path_sv = "%s/Spectroscopic_Values_01" % image_path
    main_h5_handle[image_path_sv] = [1]
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_sv_ref = h5_image_new.ref
    new_sv_reg = h5_image_new.regionref[0]

    image_path_si = "%s/Spectroscopic_Indices_01" % image_path
    main_h5_handle[image_path_si] = [1]
    h5_image_new = main_h5_handle.get(image_path_si)
    new_si_ref = h5_image_new.ref
    new_si_reg = h5_image_new.regionref[0]

    image_path_sv = "%s/Position_Values_01" % image_path
    main_h5_handle[image_path_sv] = xy
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_pv_ref = h5_image_new.ref
    new_pv_reg = h5_image_new.regionref[0:1, 0:len(xy[0])]

    image_path_si = "%s/Position_Indices_01" % image_path
    main_h5_handle[image_path_si] = xy
    h5_image_new = main_h5_handle.get(image_path_si)
    new_pi_ref = h5_image_new.ref
    new_pi_reg = h5_image_new.regionref[0:1, 0:len(xy[0])]

    image_path_b = "%s/Analysis_Data_01_U" % image_path
    main_h5_handle[image_path_b] = U
    h5_image_new = main_h5_handle.get(image_path_b)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "PCA_Atom_Shape"
    h5_new_attrs["Number_Of_Variables"] = 1
    h5_new_attrs["Variable_1_Name"] = "Box Width"
    h5_new_attrs["Variable_1_Value"] = box_width

    h5_new_attrs["Spectroscopic_Indices"] = new_si_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = new_si_reg
    h5_new_attrs["Spectroscopic_Values"] = new_sv_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = new_sv_reg
    h5_new_attrs["Position_Indices"] = new_pi_ref
    h5_new_attrs["Position_Indices_Region"] = new_pi_reg
    h5_new_attrs["Position_Values"] = new_pv_ref
    h5_new_attrs["Position_Values_Region"] = new_pv_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    [xy] = np.meshgrid(range(1, len(S) + 1))

    image_path_sv = "%s/Spectroscopic_Values_02" % image_path
    main_h5_handle[image_path_sv] = [1]
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_sv_ref = h5_image_new.ref
    new_sv_reg = h5_image_new.regionref[0]

    image_path_si = "%s/Spectroscopic_Indices_02" % image_path
    main_h5_handle[image_path_si] = [1]
    h5_image_new = main_h5_handle.get(image_path_si)
    new_si_ref = h5_image_new.ref
    new_si_reg = h5_image_new.regionref[0]

    image_path_sv = "%s/Position_Values_02" % image_path
    main_h5_handle[image_path_sv] = xy
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_pv_ref = h5_image_new.ref
    new_pv_reg = h5_image_new.regionref[0:len(xy)]

    image_path_si = "%s/Position_Indices_02" % image_path
    main_h5_handle[image_path_si] = xy
    h5_image_new = main_h5_handle.get(image_path_si)
    new_pi_ref = h5_image_new.ref
    new_pi_reg = h5_image_new.regionref[0:len(xy)]

    image_path_b = "%s/Analysis_Data_02_S" % image_path
    main_h5_handle[image_path_b] = S
    h5_image_new = main_h5_handle.get(image_path_b)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "PCA_Atom_Shape"
    h5_new_attrs["Number_Of_Variables"] = 1
    h5_new_attrs["Variable_1_Name"] = "Box Width"
    h5_new_attrs["Variable_1_Value"] = box_width

    h5_new_attrs["Spectroscopic_Indices"] = new_si_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = new_si_reg
    h5_new_attrs["Spectroscopic_Values"] = new_sv_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = new_sv_reg
    h5_new_attrs["Position_Indices"] = new_pi_ref
    h5_new_attrs["Position_Indices_Region"] = new_pi_reg
    h5_new_attrs["Position_Values"] = new_pv_ref
    h5_new_attrs["Position_Values_Region"] = new_pv_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    [xx, yy] = np.meshgrid(range(1, len(V) + 1), range(1, len(V[0]) + 1))
    xx = np.reshape(xx, [1, len(V) * len(V[0])])
    yy = np.reshape(yy, [1, len(V) * len(V[0])])
    xy = [xx, yy]
    V = np.reshape(U, [1, len(U) * len(U[0])])

    image_path_sv = "%s/Spectroscopic_Values_03" % image_path
    main_h5_handle[image_path_sv] = [1]
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_sv_ref = h5_image_new.ref
    new_sv_reg = h5_image_new.regionref[0]

    image_path_si = "%s/Spectroscopic_Indices_03" % image_path
    main_h5_handle[image_path_si] = [1]
    h5_image_new = main_h5_handle.get(image_path_si)
    new_si_ref = h5_image_new.ref
    new_si_reg = h5_image_new.regionref[0]

    image_path_sv = "%s/Position_Values_03" % image_path
    main_h5_handle[image_path_sv] = xy
    h5_image_new = main_h5_handle.get(image_path_sv)
    new_pv_ref = h5_image_new.ref
    new_pv_reg = h5_image_new.regionref[0:1, 0:len(xy[0])]

    image_path_si = "%s/Position_Indices_03" % image_path
    main_h5_handle[image_path_si] = xy
    h5_image_new = main_h5_handle.get(image_path_si)
    new_pi_ref = h5_image_new.ref
    new_pi_reg = h5_image_new.regionref[0:1, 0:len(xy[0])]

    image_path_b = "%s/Analysis_Data_03_V" % image_path
    main_h5_handle[image_path_b] = V1
    h5_image_new = main_h5_handle.get(image_path_b)
    h5_new_attrs = h5_image_new.attrs
    h5_new_attrs["Filter_Name"] = "PCA_Atom_Shape"
    h5_new_attrs["Number_Of_Variables"] = 1
    h5_new_attrs["Variable_1_Name"] = "Box Width"
    h5_new_attrs["Variable_1_Value"] = box_width

    h5_new_attrs["Spectroscopic_Indices"] = new_si_ref
    h5_new_attrs["Spectroscopic_Indices_Region"] = new_si_reg
    h5_new_attrs["Spectroscopic_Values"] = new_sv_ref
    h5_new_attrs["Spectroscopic_Values_Region"] = new_sv_reg
    h5_new_attrs["Position_Indices"] = new_pi_ref
    h5_new_attrs["Position_Indices_Region"] = new_pi_reg
    h5_new_attrs["Position_Values"] = new_pv_ref
    h5_new_attrs["Position_Values_Region"] = new_pv_reg
    h5_new_attrs["Parent"] = current_ref
    h5_new_attrs["Parent_Region"] = current_reg

    main_h5_handle.close()

    return
