from keras.models import *
from keras.layers import *
from keras import optimizers
from sklearn.model_selection import train_test_split
import keras
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from graphics import *

#Create a random order to visit a pair of excel sheets
#Useful for complete random walks through training / test sets for dual task comparisons
def dual_scrambled_order(createDepth):
	sheet_visit_order = []
	vist_order1 = np.arange(createDepth)
	np.random.shuffle(vist_order1)
	vist_order2 = np.arange(createDepth)
	np.random.shuffle(vist_order2)
	for i in range(createDepth):
		sheet_visit_order.append([vist_order1[i]+1, vist_order2[i]+1]) #Spreadsheets are offset by 1 due to headers
	return sheet_visit_order

#Helper function that converts status strings into floats for pre-processing
def convert_status_to_float(inString) :
	if inString == "Done" :
		return 1.
	elif inString == "New" :
		return 1./3
	elif inString == "Active" :
		return 1./3*2


#Helper function that normalizes date values into floats for pre-processing
def normalize_date(dateIn, minimumDate, dateDifference):
	return (dateIn-minimumDate)/dateDifference


#Helper function that normalizes the values into floats for pre-processing
def normalize_value(valueIn, maxValue) :
	return valueIn / maxValue


#This function reads data from the given worksheet and puts it into a dictionary to be read as training data for the network
#
#The read data is then output in a list of dictionaries with the following values
# DueDate 			- Given due date from the training input spreadsheet [INT]
# DueFloat 			- Noramlized version of the due date, evenly distributed across the numberline [FLOAT]
# Status 			- Given status[STRING]
# StatusFloat 		- Noramlized version of status [FLOAT]
# Value 			- Given value [INT]
# ValueFloat 		- Normalized value evenly distributed across the numberline [FLOAT]
# GivenPos 			- Given position[INT]
# ExpectedPosition 	- Normalized post sort position, sorts are done during pre-processing to sidestep input errors [FLOAT]
def import_data_onetask(fname, verbosity):
	"Import data assumes the worksheet given by task_generator.py"
	#Read from our given workbook, assuming a the structure given from task_generator.py
	verbose = verbosity #Set to 0 to stop this function printing, 1 to see general values, 2 for the dictionary output & 3 for debug values
	
	list_Data = []
	workbook = xlrd.open_workbook(fname)
	sheet_names = workbook.sheet_names()
	sheet = workbook.sheet_by_name(sheet_names[0])
	
	#Read the values from the sheet and organise them into our dictionary
	j = 1
	while(j < sheet.nrows):
		list_row = dict(DueDate=-1, Status=-1, Value=-1, GivenPos=-1)
		list_row['DueDate'] = int(sheet.cell_value(j, 0))
		list_row['Status'] = sheet.cell_value(j, 3)
		list_row['Value'] = int(sheet.cell_value(j, 4))
		list_row['GivenPos'] = int(sheet.cell_value(j, 1))
		list_Data.append(list_row)
		j=j+1
	
	#Set our local constants to allow us to spread the read values across the numberline
	max_due = max(list_Data, key=lambda x:x['DueDate'])
	if verbose > 0 : print("Max DueDate : " + str(max_due['DueDate']))
	
	min_due = min(list_Data, key=lambda x:x['DueDate'])
	if verbose > 0 : print("Min DueDate : " + str(min_due['DueDate']))
	
	date_delta = max_due['DueDate']-min_due['DueDate']
	if verbose > 0 : print("DueDate Delta : " + str(date_delta))
	
	max_value = max(list_Data, key=lambda x:x['Value'])
	if verbose > 0 : print("Max Value : " + str(max_value['Value']))
	
	count = 0
	#Normalize the strings we care about into floats for the NN
	for locRow in list_Data :
		locRow['StatusFloat'] = convert_status_to_float(locRow['Status'])
		
		#Normalize the dates and values by decompressing them across the numberline
		locRow['DueFloat'] = normalize_date(locRow['DueDate'], min_due['DueDate'], date_delta)
		locRow['ValueFloat'] = normalize_value(locRow['Value'],max_value['Value'])
		locRow['ExpectedPosition'] = count/len(list_Data)
		count += 1

	if verbose > 1 : print(list_Data)
	return list_Data


#This function reads data from the given worksheet and puts it into a dictionary to be read as training data for the network
#This is designed to read in two different tasks
#The read data is then output in a list of dictionaries with the following values
# DueDate, DueDate2						- Given due date from the training input spreadsheet [INT]
# DueFloat, DueFloat2					- Noramlized version of the due date, evenly distributed across the numberline [FLOAT]
# Status, Status2						- Given status[STRING]
# StatusFloat, StatusFloat2				- Noramlized version of status [FLOAT]
# Value, Value2	 						- Given value [INT]
# ValueFloat, ValueFloat2 				- Normalized value evenly distributed across the numberline [FLOAT]
# GivenPos, GivenPos2 					- Given position[INT]
# ExpectedPosition, ExpectedPosition2	- Normalized post sort position, sorts are done during pre-processing to sidestep input errors [FLOAT]
def import_data_twotask(fname, verbosity):
	"Import data assumes the worksheet given by task_generator_dual.py"
	#Read from our given workbook, assuming a the structure given from task_generator_dual.py
	verbose = verbosity #Set to 0 to stop this function printing, 1 to see general values, 2 for the dictionary output & 3 for debug values
	
	list_Data = []
	workbook = xlrd.open_workbook(fname)
	sheet_names = workbook.sheet_names()
	sheet1 = workbook.sheet_by_name(sheet_names[0])
	sheet2 = workbook.sheet_by_name(sheet_names[1])
	
	
	visit_order = dual_scrambled_order(250)
	
	#Read the values from the sheet and organise them into our dictionary
	for j in range(len(visit_order)):
		list_row = dict(DueDate1=-1, Status1=-1, Value1=-1, GivenPos1=-1, DueDate2=-1, Status2=-1, Value2=-1, GivenPos2=-1)
		print(visit_order[j][0])
		list_row['DueDate'] = int(sheet1.cell_value(visit_order[j][0], 0))
		list_row['DueDate2'] = int(sheet2.cell_value(visit_order[j][1], 0))
		list_row['Status'] = sheet1.cell_value(visit_order[j][0], 3)
		list_row['Status2'] = sheet2.cell_value(visit_order[j][1], 3)
		list_row['Value'] = int(sheet1.cell_value(visit_order[j][0], 4))
		list_row['Value2'] = int(sheet2.cell_value(visit_order[j][1], 4))
		list_row['GivenPos'] = int(sheet1.cell_value(visit_order[j][0], 1))
		list_row['GivenPos2'] = int(sheet2.cell_value(visit_order[j][1], 1))
		list_Data.append(list_row)
	
	#Set our local constants to allow us to spread the read values across the numberline
	max_due1 = max(list_Data, key=lambda x:x['DueDate'])
	max_due2 = max(list_Data, key=lambda x:x['DueDate2'])
	if verbose > 0 : print("Max DueDate 1 : " + str(max_due1['DueDate']) + " | Max DueDate 2 : " + str(max_due2['DueDate2']))
	
	min_due1 = min(list_Data, key=lambda x:x['DueDate'])
	min_due2 = min(list_Data, key=lambda x:x['DueDate2'])
	if verbose > 0 : print("Min DueDate 1 : " + str(min_due1['DueDate']) + " | Min DueDate 2 : " + str(min_due2['DueDate2']))
	
	date_delta1 = max_due1['DueDate1']-min_due1['DueDate']
	date_delta2 = max_due2['DueDate2']-min_due2['DueDate2']
	if verbose > 0 : print("DueDate Delta 1 : " + str(date_delta1) + " | DueDate Delta 2 : " + str(date_delta2))
	
	max_value1 = max(list_Data, key=lambda x:x['Value'])
	max_value2 = max(list_Data, key=lambda x:x['Value2'])
	if verbose > 0 : print("Max Value 1 : " + str(max_value1['Value']) + " | Max Value 2 : " + str(max_value2['Value2']))
	
	count = 0
	#Normalize the strings we care about into floats for the NN
	for locRow in list_Data :
		locRow['StatusFloat'] = convert_status_to_float(locRow['Status'])
		locRow['StatusFloat2'] = convert_status_to_float(locRow['Status2'])
		
		#Normalize the dates and values by decompressing them across the numberline
		locRow['DueFloat'] = normalize_date(locRow['DueDate'], min_due1['DueDate'], date_delta1)
		locRow['DueFloat2'] = normalize_date(locRow['DueDate2'], min_due2['DueDate2'], date_delta2)
		locRow['ValueFloat'] = normalize_value(locRow['Value'],max_value1['Value'])
		locRow['ValueFloat2'] = normalize_value(locRow['Value2'],max_value2['Value2'])
		locRow['ExpectedPosition'] = count/len(list_Data)
		locRow['ExpectedPosition2'] = count/len(list_Data)
		count += 1

	if verbose > 1 : print(list_Data)
	return list_Data




#Generate different models, used to retain previous test models
def generateModel (givenModel, choice):
	#Prototype 1.0.1
	#First attempt at changing the model, based around trying to classify the 3d input space
	if choice == 1 :
		givenModel.add(Dense(3,input_dim=3,activation='sigmoid')) #input layer
		givenModel.add(Dense(8,activation='sigmoid')) #1 hidden layer - 3d Lines
		givenModel.add(Dense(7,activation='sigmoid')) #1 hidden layer - 3d turns
		givenModel.add(Dense(1,activation='sigmoid')) #output layer
	#6 input model comparing two tasks
	elif choice == 2:
		givenModel.add(Dense(6, input_dim=6, activation='tanh')) #input layer for Due/Due/Status/Status/Value/Value
		givenModel.add(Dense(3, activation='tanh')) #comparators for Due/Status/Value
		givenModel.add(Dense(1, activation='tanh')) #output layer, postive means first task first, negative means first task second
	#First model we started with, Prototype 1.0
	else :
		givenModel.add(Dense(3,input_dim=3,activation='sigmoid')) #input layer
		givenModel.add(Dense(8,activation='sigmoid')) #1 hidden layer
		givenModel.add(Dense(12,activation='sigmoid')) #1 hidden layer
		givenModel.add(Dense(3,activation='sigmoid')) #1 hidden layer
		givenModel.add(Dense(1,activation='sigmoid')) #output layer
		
def plotTrainingData(givenDictionary):
	#Plot that shit
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import cm
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	X_Values = []
	Y_Values = []
	Z_Values = []
	Colours = []
	
	for i in givenDictionary:
		X_Values.append(i['DueFloat'])
		Y_Values.append(i['StatusFloat'])
		Z_Values.append(i['ValueFloat'])
		Colours.append(i['ExpectedPosition'])

	#Plot everything
	ax.scatter(	X_Values,
				Y_Values,
				Z_Values,
				zdir='z', #When plotted 2d use Values as depth
				c=Colours,
				cmap=cm.coolwarm) 
				
	ax.set_xlabel('Due Date')
	ax.set_ylabel('Status')
	ax.set_zlabel('Value')
	
	fig.savefig('foo.pdf', bbox_inches='tight')
	plt.clf()
	#No more Plots

#Outputs the values of single task prediction models
def output_singleTask(prediction, testValues, targetValues) :
	#Start of Graphics engine

	windowWidth = 800
	windowHeight = 600
	windowTitle = "AI Mind"

	win = GraphWin(windowTitle, windowWidth, windowHeight)
	win.setBackground(color_rgb(236, 254, 233))

	#Create the array of text location points
	box_count = 10
	txt_points = [Point(windowWidth/2, 40+(j*20)) for j in range(box_count)]
	text_boxes = [Text(txt_points[j], "Starter!") for j in range(box_count)]

	for j in range(box_count):
		text_boxes[j].draw(win)		 # draw the object to the window

	#Create the heading label above the strings
	heading_text = Text(Point(windowWidth/2, 20), "Due Date, Status, Value | Target Pos | Predicted Pos")
	heading_text.draw(win)				# draw the heading to the window

	Count = box_count

	#Close window button
	close_Window_Box = Rectangle(Point(windowWidth-45, windowHeight-20), Point(windowWidth-1, windowHeight-1))
	close_Window_Text = Text(close_Window_Box.getCenter(), "Close")
	close_Window_Box.draw(win)
	close_Window_Text.draw(win)

	for i in range(0,len(prediction)):
		#print("actual",training_target1[i]*249,"predicted data",pred[i]*249)
		#pDate = m[0,i]
		#pStatus = m[1,i]
		#pValue = m[2,i]
		pGuff = testValues[i]
		pGuff[0] = round(pGuff[0], 4)
		pGuff[1] = round(pGuff[1], 4)
		pPos = targetValues[i]
		pPredict = np.around(prediction[i], 4)
		print(pGuff,'|',pPos,'|',pPredict)
		text_boxes[i%10].setText(str(pGuff) + "|" + str(pPos) + "|" + str(pPredict))
		Count = Count - 1
		if Count == 0:
			Count = box_count - 1
			mout = win.getMouse()	  # Pause to view result until the user clicks the mouse
			p1 = close_Window_Box.getP1()
			p2 = close_Window_Box.getP2()
			if mout.getX() > p1.getX() and mout.getY() > p1.getY():
				win.close()


#Output the values of daul task predictions
def output_dualTask(prediction, testValues, targetValues) :
	count = 0
	for i in prediction:
		print(np.around(i, 4), end=' | ')
		print(testValues[count], end=' | ')
		print(targetValues[count])
		count += 1
		

#
# START OF MAIN
#

imported_data = []
ChosenModel = 2 #0 for original model, 1 for prototype model, 2 for dual task model
No_training_batches = 4

#First models
if ChosenModel < 2:
	for i in range(No_training_batches):
		fileName = './TrainingData/PT1_Train_00' + str(i+1) + '.xlsx'
		imported_data.extend(import_data_onetask(fileName, 1)) #Add the entire array, Append will just add the first element
#Dual task model
if ChosenModel == 2:
	for i in range(No_training_batches):
		fileName = './TrainingDataDual/PT1_Train_00' + str(i+1) + '.xlsx'
		imported_data.extend(import_data_twotask(fileName, 1)) #Add the entire array, Append will just add the first element

#Plot the training data to visualize it
plotTrainingData(imported_data)

if ChosenModel < 2 : #Single Task Models
	training_data = np.zeros((len(imported_data), 3))
	training_target = np.zeros((len(imported_data), 1))
	#Pull out the training data and target from the dictionary
	count = 0
	for locRow in imported_data:
		training_data[count] = np.array([locRow['DueFloat'], 
										locRow['StatusFloat'], 
										locRow['ValueFloat']])
		training_target[count] = np.array([locRow['ExpectedPosition']])
		count += 1
else : #Dual Task Models
	training_data = np.zeros((len(imported_data), 6))
	training_target = np.zeros((len(imported_data), 1))
	#Pull out the training data and target from the dictionary
	count = 0
	for locRow in imported_data:
		training_data[count] = np.array([locRow['DueFloat'], 
										locRow['DueFloat2'], 
										locRow['StatusFloat'], 
										locRow['StatusFloat2'], 
										locRow['ValueFloat'], 
										locRow['ValueFloat2']])
		#The goal is to get a number in the range of -1 to 1 that is the proportional distance two tasks are apart in the set with negative meaning the first task is AFTER the second
		ttarget = (locRow['ExpectedPosition2']-locRow['ExpectedPosition']) / len(imported_data)
		training_target[count] = np.array(ttarget)
		count += 1
		

#split the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(training_data, training_target, test_size=0.2)

#Generate the architecture for the model
model = Sequential()
generateModel(model, ChosenModel)
model.summary()

model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=["accuracy"])

localEpochs = 1000
hist=model.fit(	np.array(training_data, dtype=float),
				np.array(training_target, dtype=float),
				epochs=localEpochs, 
				validation_split=0.2)

#hist=model.fit(	x_train, 
				#y_train, 
				#epochs=localEpochs, 
				#validation_data=(x_test , y_test))
				#callbacks=[keras.callbacks.TensorBoard(log_dir="logs/final/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)])
#score = model.evaluate(x_test, y_test, batch_size=2)
#print('Test accuracy:', score[1])

if 1==0 : 
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(len(train_loss))
	xc1=range(len(val_loss))

	plt.plot(xc,train_loss)
	plt.plot(xc1,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.legend(['train','val'])
	plt.show()

print("Finished showing the plot")

# for the prediction part
TestInputs=[]
TestTargets=[]

#fname = 'data_scramble.xlsx'
if ChosenModel < 2 : #Single task prediction
	fname = 'data.xlsx'
	workbook = xlrd.open_workbook(fname)
	sheet_names = workbook.sheet_names()
	sheet = workbook.sheet_by_name(sheet_names[5])
	list_data = []
	j = 1
	while(j < sheet.nrows):
		Vals = [sheet.cell_value(j, 0), sheet.cell_value(j, 1), sheet.cell_value(j, 2)]
		Targets = sheet.cell_value(j, 3)
		j = j + 1
		TestInputs.append(Vals)
		TestTargets.append(Targets)
	
elif ChosenModel == 2 : #Dual task prediction
	print("Started loading the PT1 Test file")
	fname = '.\TrainingDataDual\PT1_Test.xlsx'
	print("Loaded the PT1 Test file")
	workbook = xlrd.open_workbook(fname)
	sheet_names = workbook.sheet_names()
	sheet1 = workbook.sheet_by_name(sheet_names[0])
	sheet2 = workbook.sheet_by_name(sheet_names[1])
	list_data = []
	visit_order = dual_scrambled_order(250)
	print("Started iterating through the sheet")
	for j in range(len(visit_order)):
		Vals = [sheet1.cell_value(visit_order[j][0], 0), #Due 1
			sheet1.cell_value(visit_order[j][0], 1),  #Status 1
			sheet1.cell_value(visit_order[j][0], 2),  #Value 1
			sheet2.cell_value(visit_order[j][1], 0),  #Due 2
			sheet2.cell_value(visit_order[j][1], 1),  #Status 2
			sheet2.cell_value(visit_order[j][1], 2),] #Value 2
		Targets = [sheet1.cell_value(visit_order[j][0],3), sheet2.cell_value(visit_order[j][1],3)]
		TestInputs.append(Vals)
		TestTargets.append(Targets)

testing_array = np.array(TestInputs, dtype=float)
pred = model.predict(testing_array)
print(testing_array)
print(pred)  

if ChosenModel < 2 : #Single task prediction
	output_singleTask(pred, testing_array, TestTargets)
else : 
	output_dualTask(pred, testing_array, TestTargets)
















#for saving the model to json format and weights in hd5 format
# # serialize model to JSON
# model_json = model.to_json()
#
# with open("model.json", "w") as json_file:
#	 json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


# later for loading the model and weights...
# # # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
#
# hist=loaded_model.fit(x_train, y_train, batch_size=2,epochs=150, validation_data=(x_test , y_test))
#
#
# # score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))