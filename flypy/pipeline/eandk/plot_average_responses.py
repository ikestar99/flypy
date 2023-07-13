import numpy
import ResponseTools
from pylab import *

parent_dir = '/Users/erin/Desktop/Mi1-ATPSnFR/'
plot_dir = parent_dir+'plots/'
output_dir = parent_dir+'measurements/'

st_index = 5

input_csv = parent_dir+'inputs.csv'
rows,dict_header = ResponseTools.read_csv_file(input_csv)

ALL = []
for row in rows[:]:
	input_dict = ResponseTools.get_input_dict(row, dict_header)
	if input_dict['include']=='TRUE':
		print('including ' + input_dict['sample_name'] + ' ' + input_dict['reporter_name'] + ' ' + input_dict['stimulus_name'])
		sample_dir = parent_dir+input_dict['sample_name']
		input_file = sample_dir+'/measurements/'+input_dict['output_name']+'-average.csv'
		data,data_header = ResponseTools.read_csv_file(input_file)
		for d in data:
			ALL.append(d)
		out_name = input_dict['output_name']
ResponseTools.write_csv(ALL, data_header, output_dir + out_name + '-average-ALL.csv')

A = numpy.asarray(ALL)
A = numpy.asarray(A[:,st_index:],dtype = float)
A = A[A[:,0].argsort()]

t = float(input_dict['time_interval'])

num_st = numpy.max(A[:,0])
st = 1
min_index = 0
while st<=num_st:
	max_index = numpy.searchsorted(A[:,0],st+1)
	stALL = A[min_index:max_index,:]
	average = numpy.average(stALL[:,1:],axis = 0)
	stdev = numpy.std(stALL[:,1:],axis = 0)
	ste = stdev/numpy.sqrt(len(stALL))
	T = numpy.arange(t,t*(len(average)+1),t)
	plot(T,average)
	fill_between(T,average-ste,average+ste,alpha=0.2)
	st=st+1
	min_index=max_index
ymin = gca().get_ylim()[0]
ymax = gca().get_ylim()[1]
on = input_dict['on_time']
off = input_dict['off_time']
#plot([on,on],[ymin,ymax],color = 'yellow')
#fill_between([on,off],[ymin,ymin],[ymax,ymax],color = 'yellow',alpha=0.1)
xlabel(('seconds'))
ylabel(('\u0394 F/F'))
savefig(plot_dir+out_name+'-average-ALL.png',dpi=300,bbox_inches='tight')
clf()

