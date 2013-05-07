#Pybrain Implementation of Air Travel Delay Neural Network.

from __future__ import division
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import FeedForwardNetwork,LinearLayer,SigmoidLayer,FullConnection
from pybrain.supervised.trainers import BackpropTrainer
import sys
from math import fabs

#Scaling factors For Data
d = 4000 #Distance
op = 2700 #Operations/day
vis = 10 #Visibility
t = 40 #Temperature
p = 30 #Pressure
ws = 44 #Wind Speed
dl = 100 #Delays

trainingSet = SupervisedDataSet(6,1)
pathToTrainingFile = raw_input("Please Enter The Path To The Training Data:  \n")
try:
	trainingFile = open(pathToTrainingFile)
except IOError:
	print "You fool! Training File Not Found. Try typing the path in correctly next time."
	sys.exit()
testYN = raw_input("Do You Wish To Test The Final Network? [Y/N] \n")
pathToTestFile = raw_input("Please Enter The Path To The Test File: \n")
try:
	testFile = open(pathToTestFile)
except IOError:
	print "You fool! Test File Not Found. Try typing the path in correctly next time."
	sys.exit()

for line in trainingFile.readlines():
	data = [float(x) for x in line.strip().split(',') if x != '']
	distance = data[0]/d #Scaling Distance Data
	operations = data[1]/op #Scaling number of Operations
	visibility = data[2]/vis # Scaling visibility
	temp = data[3]/t #Scaling temperature
	pressure = data[4]/p #Scaling pressure
	windspeed = data[5]/ws #Scaling wind speed
	inputs = distance,operations,visibility,temp,pressure,windspeed
	delay = data[6]/dl
	trainingSet.addSample(inputs,delay)

n = FeedForwardNetwork()

#Build Network
hiddenN = int(input("Enter Number of Hidden Neurons. Recommended: 6 \n"))
print "Building Network"
inLayer = LinearLayer(6)
hiddenLayer = SigmoidLayer(hiddenN)
outLayer = LinearLayer(1)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer , hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer , outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()

#Train The Network
lr = float(input("Enter Learning Rate. High accuracy: 0.01 , Mixed Performance: 0.3 , Just get it over with: 0.5 \n"))
mmtum = float(input("Enter backpropagation momentum. High Accuracy: 0.05 , Mixed Performance: 0.1, Just get it over with: 0.5 \n"))
maxepoch = int(input("Enter Number of Training Iterations (Integer > 10): \n"))
trainer = BackpropTrainer(n, learningrate=lr, momentum=mmtum, verbose=True)
trainer.trainOnDataset(trainingSet, maxepoch)
trainer.testOnData(verbose=True)

if(testYN) == "N":
	sys.exit()

print("\n \n \n Testing Network...")
errors = 0
testCounts = 0
for line in testFile.readlines():
	data = [float(x) for x in line.strip().split(',') if x != '']
	distance = data[0]/d #Scaling Distance Data
	operations = data[1]/op #Scaling number of Operations
	visibility = data[2]/vis # Scaling visibility
	temp = data[3]/t #Scaling temperature
	pressure = data[4]/p #Scaling pressure
	windspeed = data[5]/ws #Scaling wind speed
	inputs = distance,operations,visibility,temp,pressure,windspeed
	delay = data[6]/dl
	estDelay = n.activate(inputs)[0]*dl
	errors += fabs(estDelay-delay*dl)
	testCounts += 1

avgError = errors/testCounts
print "\n \n \n \n"
print "Average Error in Minutes:"
print avgError