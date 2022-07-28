#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:40:33 2020

@author: dzhigd
"""
import matplotlib.pyplot as plt
from bcdi.utils import validation as valid
import numpy as np
from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util

def roll_2d_frame(frame, horizontal_shift, vertical_shift):
    frame_roll = frame.copy()
    frame_roll = np.roll(frame_roll, -vertical_shift, axis = 0)    # Positive y rolls up
    frame_roll = np.roll(frame_roll, -horizontal_shift, axis = 1)     # Positive x rolls right
    return frame_roll

def normalize(data,axis = None):
    """
    normalize data 
    """
    if axis:
        return (data - np.mean(data, axis)) / (np.std(data, axis))
    else:
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0))

def calculate_projection(data,axis=None):
    if axis:
        output = np.sum(data,axis)
    else:
        output = np.sum(data,0)
        print("Projected along first dimension")
    return output

def cross_corr(data1, data2):
    """Calculates the cross correlation and lags without normalization.

    The definition of the discrete cross-correlation is in:
    https://www.mathworks.com/help/matlab/ref/xcorr.html

    Args:
    y1, y2: Should have the same length.

    Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
    """
    if len(data1) != len(data2):
        raise ValueError('The lengths of the inputs should be the same.')

    data1_auto_corr = np.dot(data1, data1) / len(data1)
    data2_auto_corr = np.dot(data2, data2) / len(data1)
    corr = np.correlate(data1, data2, mode='same')
    # The unbiased sample size is N - lag.
    unbiased_sample_size = np.correlate(np.ones(len(data1)), np.ones(len(data1)), mode='same')
    corr = corr / unbiased_sample_size / np.sqrt(data1_auto_corr * data2_auto_corr)
    shift = len(data1) // 2

    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)#
    return max_corr, argmax_corr - shift


def COM_voxels_reciproc(data, Qx, Qz, Qy ):

    # center of mass calculation in reciprocal space with the meshgrids
    COM_qx = np.sum(data* Qx)/np.sum(data)
    COM_qz = np.sum(data* Qz)/np.sum(data)
    COM_qy = np.sum(data* Qy)/np.sum(data)

    #print( 'coordinates in reciprocal space:')
    #print( COM_qx, COM_qz, COM_qy)
    return COM_qx, COM_qz, COM_qy

def grid_bcdi_labframe(
    data,
    mask,
    detector,
    setup,
    align_q=True,
    reference_axis=(1, 0, 0),
    debugging=False,
    **kwargs,
):
    """
    Interpolate BCDI reciprocal space data using a linearized transformation matrix.
    The resulting (qx, qy, qz) are in the laboratory frame (qx downstrean,
    qz vertical up, qy outboard).
    :param data: the 3D data, already binned in the detector frame
    :param mask: the corresponding 3D mask
    :param detector: an instance of the class Detector
    :param setup: instance of the Class experiment_utils.Setup()
    :param align_q: boolean, if True the data will be rotated such that q is along
     reference_axis, and q values will be calculated in the pseudo crystal frame.
    :param reference_axis: 3D vector along which q will be aligned, expressed in an
     orthonormal frame x y z
    :param debugging: set to True to see plots
    :param kwargs:
     - 'cmap': str, name of the colormap
     - 'fill_value': tuple of two real numbers, fill values to use for pixels outside
       of the interpolation range. The first value is for the data, the second for the
       mask. Default is (0, 0)
     - 'logger': an optional logger
    :return:
     - the data interpolated in the laboratory frame
     - the mask interpolated in the laboratory frame
     - a tuple of three 1D vectors of q values (qx, qz, qy)
     - a numpy array of shape (3, 3): transformation matrix from the detector
       frame to the laboratory/crystal frame
    """
    # logger = kwargs.get("logger", module_logger)
    # valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    # valid.valid_kwargs(
    #     kwargs=kwargs,
    #     allowed_kwargs={"cmap", "fill_value", "logger", "reference_axis"},
    #     name="kwargs",
    # )
    cmap = kwargs.get("cmap", "turbo")
    fill_value = kwargs.get("fill_value", (0, 0))
    # valid.valid_container(
    #     fill_value,
    #     container_types=(tuple, list, np.ndarray),
    #     length=2,
    #     item_types=Real,
    #     name="fill_value",
    # )

    # check some parameters
    if setup.rocking_angle == "energy":
        raise NotImplementedError(
            "Geometric transformation not yet implemented for energy scans"
        )
    # valid.valid_item(align_q, allowed_types=bool, name="align_q")
    # valid.valid_container(
    #     reference_axis,
    #     container_types=(tuple, list, np.ndarray),
    #     length=3,
    #     item_types=Real,
    #     name="reference_axis",
    # )
    reference_axis = np.array(reference_axis)

    # grid the data
    # logger.info(
    #     "Gridding the data using the linearized matrix, "
    #     "the result will be in the laboratory frame"
    # )
    string = "linmat_reciprocal_space_"
    (interp_data, interp_mask), q_values, transfer_matrix = setup.ortho_reciprocal(
        arrays=(data, mask),
        verbose=True,
        debugging=debugging,
        fill_value=fill_value,
        align_q=align_q,
        reference_axis=reference_axis,
        scale=("log", "linear"),
        title=("data", "mask"),
    )
    qx, qz, qy = q_values

    # check for Nan
    interp_mask[np.isnan(interp_data)] = 1
    interp_data[np.isnan(interp_data)] = 0
    interp_mask[np.isnan(interp_mask)] = 1
    # set the mask as an array of integers, 0 or 1
    interp_mask[np.nonzero(interp_mask)] = 1
    interp_mask = interp_mask.astype(int)

    # apply the mask to the data
    interp_data[np.nonzero(interp_mask)] = 0

    # save plots of the gridded data
    final_binning = (
        detector.preprocessing_binning[0] * detector.binning[0],
        detector.preprocessing_binning[1] * detector.binning[1],
        detector.preprocessing_binning[2] * detector.binning[2],
    )

    numz, numy, numx = interp_data.shape
    plot_comment = (
        f"_{numz}_{numy}_{numx}_"
        f"{final_binning[0]}_{final_binning[1]}_{final_binning[2]}.png"
    )

    max_z = interp_data.sum(axis=0).max()
    fig, _, _ = gu.contour_slices(
        interp_data,
       (qx, qz, qy),
       sum_frames=True,
       title="Regridded data",
       levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=True),
       plot_colorbar=True,
       scale="log",
       is_orthogonal=True,
       reciprocal_space=True,
       cmap=cmap,
    )
    # fig.savefig(detector.savedir + string + "sum" + plot_comment)
    # plt.close(fig)

    # fig, _, _ = gu.contour_slices(
    #     interp_data,
    #     (qx, qz, qy),
    #     sum_frames=False,
    #     title="Regridded data",
    #     levels=np.linspace(0, np.ceil(np.log10(interp_data.max())), 150, endpoint=True),
    #     plot_colorbar=True,
    #     scale="log",
    #     is_orthogonal=True,
    #     reciprocal_space=True,
    #     cmap=cmap,
    # )
#     fig.savefig(detector.savedir + string + "central" + plot_comment)
#     plt.close(fig)

#     fig, _, _ = gu.multislices_plot(
#         interp_data,
#         sum_frames=True,
#         scale="log",
#         plot_colorbar=True,
#         vmin=0,
#         title="Regridded data",
#         is_orthogonal=True,
#         reciprocal_space=True,
#         cmap=cmap,
#     )
#     fig.savefig(detector.savedir + string + "sum_pix" + plot_comment)
#     plt.close(fig)

#     fig, _, _ = gu.multislices_plot(
#         interp_data,
#         sum_frames=False,
#         scale="log",
#         plot_colorbar=True,
#         vmin=0,
#         title="Regridded data",
#         is_orthogonal=True,
#         reciprocal_space=True,
#         cmap=cmap,
#     )
#     fig.savefig(detector.savedir + string + "central_pix" + plot_comment)
#     plt.close(fig)
    if debugging:
        gu.multislices_plot(
            interp_mask,
            sum_frames=False,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            title="Regridded mask",
            is_orthogonal=True,
            reciprocal_space=True,
            cmap=cmap,
        )

    return interp_data, interp_mask, q_values, transfer_matrix
