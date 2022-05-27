import numpy
import ResponseClassSimple
import ResponseTools
import os
from pylab import *

parent_dir = '/Users/erin/Desktop/Mi1-ATPSnFR/'

input_csv = parent_dir+'inputs.csv'
rows,header = ResponseTools.read_csv_file(input_csv)

for row in rows[16:]:
	input_dict = ResponseTools.get_input_dict(row,header)
	#print(input_dict)
	print('analyzing sample ' + input_dict['sample_name'] + ' ' + input_dict['reporter_name'] + ' ' + input_dict['stimulus_name'])

	sample_dir = parent_dir+input_dict['sample_name']
	image_dir = sample_dir+'/images/'
	mask_dir = sample_dir+'/masks/'
	stim_dir = sample_dir+'/stim_files/'
	plot_dir = sample_dir+'/plots/'
	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)
	output_dir = sample_dir+'/measurements/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if input_dict['aligned']=='TRUE':
		image_file = image_dir + input_dict['ch1_name']+'-aligned.tif'
	else:
		image_file = image_dir + input_dict['ch1_name']+'.tif'
	mask_file = mask_dir + input_dict['mask_name']+'.tif'
	stim_file = ResponseTools.get_file_names(stim_dir,file_type = 'csv',label = input_dict['stimulus_name'])[0]
	
	response_objects, stim_data, dataheader, labels = ResponseTools.extract_response_objects(image_file,mask_file,stim_file,input_dict)
	ResponseTools.save_raw_responses_csv(response_objects,output_dir+input_dict['output_name']+'-raw.csv')
	ResponseTools.plot_raw_responses(response_objects,plot_dir+input_dict['output_name'])
	print('extracted '+str(len(response_objects))+' response objects')
	
	ResponseTools.segment_individual_responses(response_objects,input_dict)
	ResponseTools.measure_average_dff(response_objects,input_dict)
	ResponseTools.save_individual_responses_csv(response_objects,output_dir+input_dict['output_name']+'-individual.csv')
	ResponseTools.save_average_responses_csv(response_objects,output_dir+input_dict['output_name']+'-average.csv')
	ResponseTools.plot_average_responses(response_objects,plot_dir+input_dict['output_name'])

	if input_dict['verbose']=='TRUE':
		ResponseTools.write_csv(stim_data,dataheader,stim_dir + 'parsed/parsed-'+input_dict['stimulus_name'] +'.csv')