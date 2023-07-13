#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:00:35 2021

@author: katherinedelgado and Erin Barnhart
"""

import scipy.ndimage as scn

from pylab import *

import ResponseClassSimple
from flypy.pipeline import alignment
from flypy.utils.csvreader import CSVReader


def count_frames(
        filename, threshold=1, volCol="AIN4", fCol="frames",
        gtCol="global_time", dtCol="dt"):
    """
    Reads in a stimulus output file and assigns an image frame number to each
    stimulus frame
    """
    stim = CSVReader.fromFile(filename)
    # R = np.asarray(rows, dtype='float') # convert stim file list to an array
    # output_array=np.zeros((R.shape[0],R.shape[1]+2))
    # header.extend(['dt','frames'])

    vs = stim.getColumn(volCol)
    vs[1:] = (vs[1:] - vs[:-1])
    vs[0] = 0
    # count image frames based on the change in voltage signal
    count_on, count_off = 0, 0
    frame_labels = [0]
    # F_on = [0]; F_off = [0]
    for n in range(1, len(vs) - 1, 1):
        if all((
                vs[n] > vs[n - 1],
                vs[n] > vs[n + 1],
                vs[n] > threshold)):
            count_on += 1
        elif all((
                vs[n] < vs[n - 1],
                vs[n] < vs[n + 1],
                vs[n] < -threshold)):
            count_off -= 1

        # F_on.extend([count_on]); F_off.extend([count_off])
        frame_labels += [count_on * (count_on + count_off)]

    stim = stim.setColumn(fCol, frame_labels + [0])
    stim = stim.sortColumn(fCol)
    stim = stim.thresholdColumn(fCol, 1, ">=")
    stim = stim.dropDuplicates(fCol)
    gt = stim.getColumn(gtCol)
    gt[0] = 0
    stim = stim.setColumn(dtCol, gt)
    return stim


# parent_dir = '/Users/erin/Desktop/Mi1-ATPSnFR/'
#
# input_csv = parent_dir + 'inputs.csv'
# rows, header = ResponseTools.read_csv_file(input_csv)
#
# for row in rows[:1]:
#     input_dict = ResponseTools.get_input_dict(row, header)
#     # print(input_dict)
#     print('checking stim file for sample ' + input_dict['sample_name'] + ' ' +
#           input_dict['reporter_name'] + ' ' + input_dict['stimulus_name'])
#     sample_dir = parent_dir + input_dict['sample_name']
#     image_dir = sample_dir + '/images/'
#     stim_dir = sample_dir + '/stim_files/'
#     output_dir = stim_dir + 'parsed/'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     stim_file = ResponseTools.get_file_names(stim_dir, file_type='csv',
#                                              label=input_dict[
#                                                  'stimulus_name'])[0]
#     if input_dict['aligned'] == 'TRUE':
#         image_file = image_dir + input_dict['ch1_name'] + '-aligned.tif'
#     else:
#         image_file = image_dir + input_dict['ch1_name'] + '.tif'
#
#     I = ResponseTools.read_tifs(image_file)
#
#     frames = len(I)
#     time_interval = float(input_dict['time_interval'])
#     gt_index = int(input_dict['gt_index'])
#
#     stim_data, stim_data_OG, dataheader = ResponseTools.count_frames(stim_file,
#                                                                      gt_index=gt_index)
#     ResponseTools.find_dropped_frames(frames, time_interval, stim_data,
#                                       stim_data_OG, gt_index)
#
#     if input_dict['verbose'] == 'TRUE':
#         ResponseTools.write_csv(stim_data, dataheader,
#                                 stim_dir + 'parsed/parsed-' + input_dict[
#                                     'stimulus_name'] + '.csv')


def find_dropped_frames(
        frames, time_interval, oldStim, newStim, fCol="frames",
        gtCol="global_time", dtCol="dt", ):
    sFrames = newStim[-1][fCol]
    print("N image frames: {:d} \nN stim frames: {:d}".format(frames, sFrames))
    if sFrames == frames:
        print("looks fine!")
        return

    print("uh oh!")
    target_T = frames * time_interval
    stim_T = np.sum(newStim.getColumn(dtCol))
    print("total time should be {:.3f}s, got {:.3f}s ".format(
        target_T, stim_T))
    max_t_step = np.max(newStim.getColumn(dtCol))
    if np.round(max_t_step / time_interval) < 2:
        print(
            ("stim frames and image frames do not match, but no dropped "
             "frames found... double check the stim file ¯\_(ツ)_/¯"))
        return

    print("stimulus dropped at least one frame!")
    OUT = []
    num_df = 0
    for rowDict in newStim:
        if np.round(rowDict[dtCol] / time_interval) >= 2:
            num_df = num_df + 1
            gt_dropped = rowDict[gtCol] - time_interval
            stim_frame = np.searchsorted(
                oldStim.getColumn(gtCol), gt_dropped)
            print("check row {} of original stim file (maybe)".format(
                stim_frame))

        OUT.append(rowDict)

    print("found {} potential dropped frames".format(num_df))


def parse_stim_file(
        newStim, fCol="frames", rtCol="rel_time", stCol="stim_type"):
    """
    Get frame numbers, global time, relative time per epoch, and stim_state
    (if it's in the stim_file)
    """
    frames = newStim.getColumn(fCol)
    rel_time = newStim.getColumn(rtCol)
    stim_type = (
        newStim.getCol(stCol) if stCol in newStim else np.ones(frames.size))
    return frames, rel_time, stim_type


def define_stim_state(rel_time, on_time, off_time):
    """
    Define stimulus state (1 = ON; 0 = OFF) based on relative stimulus time
    """
    stim_state = ((rel_time > on_time) * (rel_time < off_time)).astype(int)
    return stim_state


def segment_ROIs(mask):
    """
    convert binary mask to labeled image
    Label all ROIs in a given mask with a unique integer value

    @param mask: integer mask array on which to perform labeling
    @type mask: numpy.ndarray

    @return: mask array with all unique non-zero ROIs labeled with a unique
        integer

        second returned value is number of ROIs labeled
    @rtype: numpy.ndarray, int
    """
    labeledMask, regions = scn.label(mask)
    return labeledMask, regions


def generate_ROI_mask(labels_image, ROI_int):
    return (labels_image == ROI_int).astype(int)


def _subtractBackground(rawstack, background):
    """
    Compute mean pixel intensity in background region of each image in a
    stack or hyperstack and subtract each image-specific value from all
    pixels in the corresponding image

    @param rawstack: input array of images on which to perform background
        correction, hyperstack or otherwise
    @type rawstack: numpy.ndarray
    @param background: 2D mask of region within image from which to compute
        background
    @type background: numpy.ndarray

    @return: array of same shape as input array with the average background
        pixel intensity of each 2D image subtracted from all other pixels
        within the image
    @rtype: numpy.ndarray
    """
    background = (background > 0).astype(int)
    background = Response._measureROIValues(rawstack, background)
    background = background[..., np.newaxis, np.newaxis]
    rawstack = rawstack - background
    return rawstack


def measure_ROI_fluorescence(image, mask):
    """
    measure average fluorescence in an ROI
    """
    masked_ROI = image * mask
    return (np.sum(masked_ROI) / np.sum(mask)).astype(float)


def measure_ROI_ts(images, mask):
    """
    Compute mean pixel intensity in masked region of each image in a stack
    or hyperstack

    @param hyperstack: input array of images on which to extract mean value
        of masked region, hyperstack or otherwise. Should correspond to a
        single ROI
    @type hyperstack: numpy.ndarray
    @param mask: 2D mask of region within image from which to compute
        mean
    @type mask: numpy.ndarray

    @return: array of same shape as input array with the omission of the
        last two axes corresponding to YX dimensions, which have been
        collapsed into a single integer
    @rtype: numpy.ndarray
    """
    for t in images.shape[0]:
        images[t] = measure_ROI_fluorescence(images[t], mask)

    return images


def measure_multiple_ROIs(images, mask_image):
    labeledMask, regions = segment_ROIs(mask_image)
    out = []
    for r in regions:
        mask = generate_ROI_mask(labeledMask, r)
        out += [measure_ROI_ts(images, mask)]

    return out, regions, labeledMask


def extract_response_objects(image_file, mask_file, stim_file, input_dict):
    """inputs are file names for aligned images, binary mask, and unprocessed stimulus file
	outputs a list of response objects"""
    # read files
    I = read_tifs(image_file)
    mask = read_tifs(mask_file)
    labels = segment_ROIs(mask)
    print('number of ROIs = ' + str(np.max(labels)))
    # process stimulus file
    stim_data, stim_data_OG, header = count_frames(stim_file)
    if (len(I)) != int(stim_data[-1][-1]):
        print("number of images does not match stimulus file")
        print('stimulus frames = ' + str(int(stim_data[-1][-1])))
        print('image frames = ' + str(len(I)))
    # stim_data = fix_dropped_frames(len(I),float(input_dict['time_interval']),stim_data,stim_data_OG,int(input_dict['gt_index']))
    # get frames, relative time, stimuulus type, and stimulus state from stim data
    fr, rt, st = parse_stim_file(stim_data,
                                 rt_index=int(input_dict['rt_index']),
                                 st_index=input_dict['st_index'])
    ss = define_stim_state(rt, float(input_dict['on_time']),
                           float(input_dict['off_time']))
    # measure fluorscence intensities in each ROI
    responses, num, labels = measure_multiple_ROIs(I, mask)
    # load response objects
    response_objects = []
    for r, n in zip(responses, num):
        ro = ResponseClassSimple.Response(F=r, stim_time=rt, stim_state=ss,
                                          ROI_num=n, stim_type=st)
        ro.sample_name = input_dict['sample_name']
        ro.reporter_name = input_dict['reporter_name']
        ro.driver_name = input_dict['driver_name']
        ro.stimulus_name = input_dict['stimulus_name']
        ro.time_interval = float(input_dict['time_interval'])
        response_objects.append(ro)
    return response_objects, stim_data, header, labels


def segment_individual_responses(response_objects, input_dict):
    for ro in response_objects:
        ro.segment_responses(int(input_dict['frames_before']),
                             int(input_dict['frames_after']))
        ro.measure_dff(int(input_dict['baseline_start']),
                       int(input_dict['baseline_stop']))


def measure_average_dff(response_objects, input_dict):
    for ro in response_objects:
        ro.measure_average_dff(int(input_dict['epoch_length']))


def save_raw_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        out = [ro.sample_name, ro.driver_name, ro.reporter_name,
               ro.stimulus_name, ro.ROI_num]
        out.extend(ro.F)
        OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(ro.F), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def save_individual_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        for st, dff in zip(ro.stim_type_ind, ro.dff):
            out = [ro.sample_name, ro.driver_name, ro.reporter_name,
                   ro.stimulus_name, ro.ROI_num, st]
            out.extend(dff)
            OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num', 'stim_type']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(dff), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def save_average_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        for st, a in zip(ro.stim_type_ind, ro.average_dff):
            out = [ro.sample_name, ro.driver_name, ro.reporter_name,
                   ro.stimulus_name, ro.ROI_num, st]
            out.extend(a)
            OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num', 'stim_type']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(a), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def plot_raw_responses(response_objects, filename):
    for ro in response_objects:
        plot(ro.F)
        savefig(filename + '-' + str(ro.ROI_num) + '-raw.png', dpi=300,
                bbox_inches='tight')
        clf()


def plot_average_responses(response_objects, filename):
    for ro in response_objects:
        for a in ro.average_dff:
            plot(a)
        savefig(filename + '-' + str(ro.ROI_num) + '-average.png', dpi=300,
                bbox_inches='tight')
        clf()


def alignMultiPageTiff(ref, img):
    tmat = []
    aligned_images = []
    for t in img:
        # print(t.shape)
        mat = alignment.registerImage(ref, t,
                                      mode="rigid")  # transformation matrix
        a = alignment.transformImage(t, mat)  # aligned image
        aligned_images.append(a)
        tmat.append(mat)
    A = np.asarray(aligned_images)
    return A, tmat


def alignFromMatrix(img, tmat):
    aligned_images = []
    for t, mat in zip(img, tmat):
        a = alignment.transformImage(t, mat)
        aligned_images.append(a)
    A = np.asarray(aligned_images)
    return A
