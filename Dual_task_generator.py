#This is the same as task_generator but it makes two even length lists of tasks spread over two spreadsheets randomlly chosen from the master list it creates

import numpy as np
from scipy import special
import math
from datetime import date
from datetime import timedelta
import random
import datetime
import pandas as pd
from pandas import DataFrame

def generate_Training_Batch(seed, length):
	"This function generates a batch of 250 randomized tasks and returns them as a dictionary, expects a random input between 1.0 and 1.1 for the seed and a FLOAT length (typically 250.)"
	#Initalize the length deep dictionary of output values, -1 is used to detect holes and errors
	TrainingOut = [dict(StartDate=-1, DueDate=-1, Status=-1, Value=-1) for _ in range(int(length))]
	
	#Generate a random distribution of durations following a zipfian function to replicate real world task duration density
	a = seed
	s = np.random.zipf(a, 100000)
	x = np.arange(1., length+1.) #Note because the sigmoid function breaks down on the orgin the input range is offset to by one
	y = x**(-a) / special.zetac(a)
	TempDuration = [i * 1000 for i in y] #Take the distribution and create a duration list in days
	Duration = [math.floor(j) for j in TempDuration]
	# print(Duration)	#This displays the duration ranges if you want to inspect what it looks like

	#Create date duration, these are our start date offsets and are evenly distributed
	DateOffset=[random.randrange(-100, 100, 1) for _ in range(len(TrainingOut))]
	FirstStartDate=datetime.date.today()

	# Making start date with help of duration generated
	StartDate = [FirstStartDate+timedelta(days=i) for i in DateOffset]

	# to calculate end date
	EndDate=[]
	for i in range(0, len(TrainingOut)):
		EndDate.append(StartDate[i]+timedelta(days=Duration[i]))
	
	active=[]
	done=[]
	new=[]
	status=[]
	x=[]

	#to calculate the offset value
	for i in range(0, len(TrainingOut)):
		x1 = EndDate[i] - FirstStartDate
		x2 = x1.days+100
		x.append(x2)

	# to calculate the value of Done status thresholds
	a =	89.84396
	b =	4.71499
	c =	88.40907
	d = -.7524174

	for i in range(0, len(TrainingOut)):
		y = d + (a-d) / (1 + (x[i]/c) ** b)
		done.append(y)

	# to calculate the value of New status thresholds
	d =	93.36659
	a =	5.222222
	b =	6.155488
	c =	112.899

	for i in range(0, len(TrainingOut)):
		y = d + (a-d) / (1 + (x[i]/c) ** b)
		new.append(y)

	#calculate value of Active status threshold
	active = list(100 - np.array(done) - np.array(new))

	#Generate random number from 0 to 100
	RanNum=[random.randrange(0, 100, 1) for _ in range(len(TrainingOut))]

	#Assign status of all tasks
	for i in range(0, len(TrainingOut)):
		if(RanNum[i] <= done[i]):
			status.append("Done")
		if (RanNum[i] < (done[i]+active[i])) :
			if((RanNum[i] >= done[i])):
			 status.append("Active")
		if (RanNum[i] <= (done[i]+active[i]+new[i])) :
			if((RanNum[i] >= active[i]+done[i])):
			 status.append("New")
		
	sort_order=[]

	#Assign values to each task
	for i in range(0, len(TrainingOut)):
		if (FirstStartDate < EndDate[i]):
			if(status[i] == "New"):
				sort_order.append(1)
			elif(status[i] == "Active"):
				sort_order.append(2)
			else:
				sort_order.append(0)
		elif (FirstStartDate == EndDate[i]):
			if(status[i] == "New"):
				sort_order.append(4)
			elif(status[i] == "Active"):
				sort_order.append(4.5)
			else:
				sort_order.append(0)
		elif (FirstStartDate > EndDate[i]):
			if(status[i] == "New"):
				sort_order.append(5)
			elif(status[i] == "Active"):
				sort_order.append(8)
			else:
				sort_order.append(0)

	#Cram these values into the Output array
	#NB should collapse the above to operate on TrainingOut directly
	for i in range(0, len(TrainingOut)):
		TrainingOut[i]['StartDate'] = StartDate[i]
		TrainingOut[i]['DueDate'] = EndDate[i]
		TrainingOut[i]['Status'] = status[i]
		TrainingOut[i]['Value'] = sort_order[i]

	return TrainingOut

#
#	Start of Main
#	Note you need a 'TrainingData' folder in the you are running this script
#
Number_Of_Batches = 1
Size_Of_Batch = 250.
#Size_Of_Batch = 500000.

for BatchNum in range(Number_Of_Batches):
	#Create a batch of training data
	Seed = (np.random.randint(0,999)/10000)+1.0
	print("Seed Chosen : " + str(Seed))
	TrainingBatch = generate_Training_Batch(Seed, Size_Of_Batch)
	
	#sort task according to values in descending order and if values are same, sort start date by ascending order
	TrainingBatch.sort(key=lambda x: (-x['Value'], x['StartDate']))

	#append position in all tasks
	for i in range(0,len(TrainingBatch)):
		TrainingBatch[i]['Position'] = i+1

	TrainingSheetName = './TrainingDataDual/PT1_Train_'
	if BatchNum < 10:
		TrainingSheetName = TrainingSheetName + '00' + str(BatchNum) + '.xlsx'
	elif BatchNum < 100:
		TrainingSheetName = TrainingSheetName + '0' + str(BatchNum) + '.xlsx'
	else:
		TrainingSheetName = TrainingSheetName + str(BatchNum) + '.xlsx'
	
	#Shuffle the array so when its broken in half its a random selection of elements that move
	np.random.shuffle(TrainingBatch)
	Both = np.array_split(TrainingBatch, 2)
	Tasks1=[]
	Tasks2=[]
	for i in Both[0]:
		Tasks1.append(i)
	for i in Both[1]:
		Tasks2.append(i)
	
	
	writer = pd.ExcelWriter(str(TrainingSheetName))
	#store all the data in excel worksheet
	df1 = DataFrame(Tasks1)
	df2 = DataFrame(Tasks2)
	df1.to_excel(writer, sheet_name='sheet1', index=False)
	df2.to_excel(writer, sheet_name='sheet2', index=False)
	writer.save()


