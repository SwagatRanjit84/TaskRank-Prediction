from keras.models import *
from keras.layers import *
from keras import optimizers
from sklearn.model_selection import train_test_split
import keras
import xlrd
import numpy as np
import matplotlib.pyplot as plt

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
def import_data(fname):
	"Import data assumes the worksheet given by task_generator.py"
	#Read from our given workbook, assuming a the structure given from task_generator.py
	verbose = 1 #Set to 0 to stop this function printing, 1 to see general values, 2 for the dictionary output & 3 for debug values
	
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
		if locRow['Status'] == "Done" :
			locRow['StatusFloat'] = 1.
		elif locRow['Status'] == "New" :
			locRow['StatusFloat'] = 1./3*2
		elif locRow['Status'] == "Active" :
			locRow['StatusFloat'] = 1./3
		
		#Normalize the dates and values by decompressing them across the numberline
		locRow['DueFloat'] = (locRow['DueDate']-min_due['DueDate'])/date_delta
		locRow['ValueFloat'] = locRow['Value']/max_value['Value']
		locRow['ExpectedPosition'] = count/len(list_Data)
		count += 1

	if verbose > 1 : print(list_Data)
	return list_Data

#Generate different models, used to retain previous test models
def generateModel (givenModel, choice):
	if choice == 1 :
		givenModel.add(Dense(3,input_dim=3,activation='sigmoid')) #input layer
		givenModel.add(Dense(8,activation='sigmoid')) #1 hidden layer - 3d Lines
		givenModel.add(Dense(7,activation='sigmoid')) #1 hidden layer - 3d turns
		givenModel.add(Dense(1,activation='sigmoid')) #output layer
		
	else :  #Given model
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

#
# START OF MAIN
#

imported_data = []

for i in range(5):
	fileName = './TrainingData/PT1_Train_00' + str(i+1) + '.xlsx'
	imported_data.extend(import_data(fileName)) #Add the entire array, Append will just add the first element
#print(imported_data)

#Plot the training data to visualize it
plotTrainingData(imported_data)

training_data = np.zeros((len(imported_data), 3))
training_target = np.zeros((len(imported_data), 1))
#Pull out the training data and target from the dictionary
count = 0
for locRow in imported_data:
	training_data[count] = np.array([locRow['DueFloat'], locRow['StatusFloat'], locRow['ValueFloat']])
	training_target[count] = np.array([locRow['ExpectedPosition']])
	count += 1
	

#split the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(training_data, training_target, test_size=0.2)

#Generate the architecture for the model
model = Sequential()
generateModel(model, 1)
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


# for the prediction part
target1=[]

#fname = 'data_scramble.xlsx'
fname = 'data.xlsx'
workbook = xlrd.open_workbook(fname)
sheet_names = workbook.sheet_names()
sheet = workbook.sheet_by_name(sheet_names[5])
list_data = []
j = 1
while(j < sheet.nrows):
	m = [sheet.cell_value(j, 0), sheet.cell_value(j, 1), sheet.cell_value(j, 2)]
	temp = sheet.cell_value(j, 3)
	j = j + 1
	list_data.append(m)
	target1.append(temp)

training_data1 = np.array(list_data, dtype=float)
training_target1 = np.array(target1, dtype=float)

pred = model.predict(training_data1) 

#Start of Graphics engine
from graphics import *
import numpy

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

for i in range(0,len(pred)):
	#print("actual",training_target1[i]*249,"predicted data",pred[i]*249)
	#pDate = m[0,i]
	#pStatus = m[1,i]
	#pValue = m[2,i]
	pGuff = list_data[i]
	pGuff[0] = round(pGuff[0], 4)
	pGuff[1] = round(pGuff[1], 4)
	pPos = training_target1[i]
	pPredict = numpy.around(pred[i], 4)
	print(pGuff,'|',pPos,'|',pPredict)
	text_boxes[i%10].setText(str(pGuff) + "|" + str(pPos) + "|" + str(pPredict))
	Count = Count - 1
	if Count == 0:
		Count = box_count - 1
		#mout = win.getMouse()	  # Pause to view result until the user clicks the mouse
		p1 = close_Window_Box.getP1()
		p2 = close_Window_Box.getP2()
		if mout.getX() > p1.getX() and mout.getY() > p1.getY():
			win.close()
















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