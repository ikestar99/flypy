#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:00:35 2021

@author: katherinedelgado and Erin Barnhart
"""
import glob
import csv
import numpy
from pylab import *
import scipy.ndimage as ndimage
from PIL import Image, ImageSequence
from .responseclasssimple import Response


def get_file_names(parent_directory, file_type='all', label=''):
    if file_type == 'csv':
        file_names = glob.glob(parent_directory + '/*' + label + '*.csv')
    elif file_type == 'all':
        file_names = glob.glob(parent_directory + '/*' + label + '*')
    return file_names


def read_csv_file(filename, header=True):
    data = []
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            data.append(row)
    if header == True:
        out_header = data[0]
        out = data[1:]
        return out, out_header
    else:
        return out


def write_csv(data, header, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def read_tifs(filename):
    # Open image as PIL ImageSequence
    tiffs = Image.open(filename)
    # Convert each image page to a numpy array, then stack into 3D array
    return numpy.array(
        [numpy.array(page) for page in ImageSequence.Iterator(tiffs)],
        dtype=np.uint8)


def save_tif(image_array, filename):
    out = Image.fromarray(image_array)
    out.save(filename)


def count_frames(filename, threshold=1):
    """
    Reads in a stimulus output file and assigns an image frame number to each
    stimulus frame.
    """
    rows, header = read_csv_file(filename)  # import stim file
    R = numpy.asarray(rows,
                      dtype='float')  # convert stim file list to an array
    # set up the output array
    output_array = numpy.zeros((R.shape[0], R.shape[1] + 1))
    header.extend(['frames'])
    # calculate change in voltage signal
    vs = [0]
    vs.extend(R[1:, -1] - R[:-1, -1])
    # count image frames based on the change in voltage signal
    count_on = 0
    F_on = [0]
    count_off = 0
    F_off = [0]
    frame_labels = [0]
    n = 1
    while n < len(vs) - 1:
        if vs[n] > vs[n - 1] and vs[n] > vs[n + 1] and vs[n] > threshold:
            count_on = count_on + 1
            F_on.extend([count_on])
            F_off.extend([count_off])
        elif vs[n] < vs[n - 1] and vs[n] < vs[n + 1] and vs[
            n] < threshold * -1:
            count_off = count_off - 1
            F_off.extend([count_off])
            F_on.extend([F_on])
        else:
            F_on.extend([count_on])
            F_off.extend([count_off])
        frame_labels.extend([count_on * (count_on + count_off)])
        n = n + 1
    frame_labels.extend([0])
    output_array[:, :R.shape[1]] = R
    output_array[:, -1] = frame_labels
    OAS = output_array[output_array[:, -1].argsort()]
    i1 = numpy.searchsorted(OAS[:, -1], 1)
    OASc = OAS[i1:, :]  # removes rows before image series start
    output_list = []
    n = 0
    frame = 1
    same_frame = []
    while n < len(OASc):
        if int(OASc[n, -1]) == frame:
            same_frame.append(list(OASc[n, :]))
        else:
            output_list.append(same_frame[0])
            same_frame = []
            same_frame.append(list(OASc[n, :]))
            frame = frame + 1
        n = n + 1
    output_list.append(same_frame[0])
    return output_list, header


def parse_stim_file(stim_info_array, frame_index=-1, gt_index=1, rt_index=2,
                    mode='simple', stim_type_index=None):
    """
    Get frame numbers, global time, relative time per epoch, and stim_state
    (if it's in the stim_file)
    """
    frames = stim_info_array[:, frame_index]
    global_time = stim_info_array[:, gt_index]
    rel_time = stim_info_array[:, rt_index]
    if mode == 'simple':
        return frames, global_time, rel_time
    elif mode == 'multiple':
        stim_type = stim_info_array[:, stim_type_index]
        return frames, global_time, rel_time, stim_type


def define_stim_state(rel_time, on_time, off_time):
    """
    Define stimulus state (1 = ON; 0 = OFF) based on relative stimulus time.
    """
    stim_state = []
    for t in rel_time:
        if on_time < t < off_time:
            stim_state.extend([1])
        else:
            stim_state.extend([0])
    return stim_state


def segment_ROIs(mask_image):
    """convert binary mask to labeled image"""
    labels = ndimage.measurements.label(mask_image)
    return labels[0]


def generate_ROI_mask(labels_image, ROI_int):
    return labels_image == ROI_int


def measure_ROI_fluorescence(image, mask):
    """measure average fluorescence in an ROI"""
    masked_ROI = image * mask
    return numpy.sum(masked_ROI) / numpy.sum(mask)


def measure_ROI_ts(images, mask):
    out = []
    for image in images:
        out.append(measure_ROI_fluorescence(image, mask))
    return out


def measure_multiple_ROIs(images, mask_image):
    labels = segment_ROIs(mask_image)
    out = []
    num = []
    n = 1
    while n <= numpy.max(labels):
        mask = generate_ROI_mask(labels, n)
        out.append(measure_ROI_ts(images, mask))
        num.append([n])
        n = n + 1
    return out, num, labels


def extract_response_objects(
        image_file, mask_file, stim_file, on_time, off_time, sample_name=None,
        reporter_name=None, driver_name=None, stimulus_name=None):
    """
    inputs are file names for aligned images, binary mask, and unprocessed
    stimulus file outputs a list of response objects
    """
    # read files
    I = read_tifs(image_file)
    mask = read_tifs(mask_file)
    labels = segment_ROIs(mask)
    print('number of ROIs = ' + str(numpy.max(labels)))
    # process stimulus file
    stim_data, header = count_frames(stim_file)
    if (len(I)) != int(stim_data[-1][-1]):
        print("number of images does not match stimulus file")
        print('stimulus frames = ' + str(int(stim_data[-1][-1])))
        print('image frames = ' + str(len(I)))
    # get frames, global time, relative time, and stimulus state from stim data
    fr, gt, rt = parse_stim_file(numpy.asarray(stim_data))
    ss = define_stim_state(rt, on_time, off_time)
    # measure fluorscence intensities in each ROI
    responses, num, labels = measure_multiple_ROIs(I, mask)
    # load response objects
    response_objects = []
    for r, n in zip(responses, num):
        ro = Response(F=r, stim_time=rt, stim_state=ss, ROI_num=n[0])
        if sample_name:
            ro.sample_name = sample_name
        if reporter_name:
            ro.reporter_name = reporter_name
        if driver_name:
            ro.driver_name = driver_name
        if stimulus_name:
            ro.stimlus_name = stimulus_name
        response_objects.append(ro)
    return response_objects
