
#
# Neural network example
#
# XOR gate simulation
#

import random
import pdb
from math import sqrt, exp
	
def sigmoid_derivative(x):
	return x * (1-x)

#
# Class representation of a synapse that connects the output of one neuron
#   to the input of another, with some weighting factor applied
#
class synapse:
	def __init__(self, n):
		self.weight = random.gauss(0.5, 0.33)
		self.neuron = n
		
	def get_value(self):
		return self.neuron.output * self.weight

#
# Class representation of a neuron
#
class neuron:
	FUNCTION_LINEAR = 0
	FUNCTION_SIGMOID = 1
	
	def __init__(self, f):
		self.input = 0
		self.output = 0
		self.function = f
		self.synapses = []
	
	def __str__(self):
		s = ""
		# Activation function
		if self.function == self.FUNCTION_LINEAR:
			s += "LINEAR  "
		elif self.function == self.FUNCTION_SIGMOID:
			s += "SIGMOID "
		else:
			s += "UNDEF   "
		# Input and output values, use to check activation function applied correctly
		s += '{:6.4f}'.format(self.input) + "  -->  "
		s += '{:6.4f}'.format(self.output) + "  "
		# Synapse inputs, use to check number and weighting
		s += "("
		for i in self.synapses:
			s += '{:6.4f}'.format(i.weight) + " "
		s += ")"
		return s
		
	def compute(self):
		# If no incoming synapses, simply propagate input to output
		if (len(self.synapses) == 0):
			self.output = self.input
		else:
			# Calculate new input from incoming synapses
			self.input = 0
			for s in self.synapses:
				self.input += s.get_value()
			# Apply new input to activation function
			if (self.function == self.FUNCTION_LINEAR):
				self.output = self.input
			elif (self.function == self.FUNCTION_SIGMOID):
				self.output = 1 / (1 + exp(-self.input))
			else:
				# Unknown activation function
				self.output = 0
		return self.output
			
	def add_synapse(self, s):
		self.synapses.append(synapse(s))


class neuralnet:
	def __init__(self, nInput, nHidden, nOutput):
		self.learningRate = 0.5
		self.input = []
		for i in range(0, nInput):
			self.input.append(neuron(neuron.FUNCTION_LINEAR))
		self.hidden = []
		for i in range(0, nHidden):
			self.hidden.append(neuron(neuron.FUNCTION_SIGMOID))
		self.output = []
		for i in range(0, nOutput):
			self.output.append(neuron(neuron.FUNCTION_SIGMOID))
		
		# Connect input to hidden layer
		for h in self.hidden:
			for i in self.input:
				h.add_synapse(i)
		
		# Connect output to hidden layer
		for o in self.output:
			for h in self.hidden:
				o.add_synapse(h)
				
		# Open .csv file for debugging / algorithm profiling
		self.csv = open("xor.csv", "w")
		
	def __del__(self):
		self.csv.close()
	
	def __str__(self):
		s = "Inputs:\n"
		for i in self.input:
			s += "  " + str(i) + "\n"
		s += "Hidden:\n"
		for h in self.hidden:
			s += "  " + str(h) + "\n"
		s += "Output:\n"
		for o in self.output:
			s += "  " + str(o) + "\n"
		return s
		
	def update (self, inValues):
		# Apply values to input neurons
		for i in range(0, len(inValues)):
			self.input[i].input = inValues[i]

		# Update input layer
		for i in self.input:
			i.compute()
			
		# Update hidden layer
		for h in self.hidden:
			h.compute()
		
		# Update output layer
		for o in self.output:
			o.compute()

	def train_single_output(self, inValues, outValue, nCycles):
		for i in range(nCycles):
			# Apply input values and forward-propagate neural net
			self.update(inValues)
			
			# Calculate
			outputNeuron = self.output[0]
			outputErrorSum = outValue - outputNeuron.output
			derivative = sigmoid_derivative(outputNeuron.output)
			deltaOutputSum = derivative * outputErrorSum
			
			# Calculate delta weights to output layer synapses, do not apply yet
			deltaOutputWeight = []
			for s in outputNeuron.synapses:
				dw = deltaOutputSum / s.neuron.output
				deltaOutputWeight.append(dw)
							
			# Calculate hidden sum deltas
			deltaHiddenFactor = []
			for j in range(len(outputNeuron.synapses)):
				dhf = deltaOutputSum / outputNeuron.synapses[j].weight
				deltaHiddenFactor.append(dhf)

			deltaHiddenSum = []
			for k in range(len(self.hidden)):
				dhs = deltaHiddenFactor[k] * sigmoid_derivative(self.hidden[k].output)
				deltaHiddenSum.append(dhs)

				for m in range(len(self.hidden[k].synapses)):
					deltaInputWeight = dhs
					self.hidden[k].synapses[m].weight += deltaInputWeight * self.learningRate
				
			# Apply weights to output layer synapses
			for j in range(len(outputNeuron.synapses)):
				outputNeuron.synapses[j].weight += deltaOutputWeight[j] * self.learningRate
				self.csv.write('{:8.6f}'.format(outputNeuron.synapses[j].weight) + ', ')
			self.csv.write("\n")
						
	def train(self, inValues, outValues, nCycles):
		for i in range(nCycles):
			self.update(inValues)

			# Calculate total squared error over all output values
			eSum = 0.0
			for o in range(0, len(outValues)):
				e = outValues[o] - self.output[o].output
				eSum += 0.5 * e * e
			dOutputSum = eSum * sigmoid_derivative(eSum)

			# Calculate weight adjustments to output layer synapses
			nOutput = len(self.output)
			oDelta = []
			for o in range(nOutput):
				oCurrent = self.output[o].output
				delta = (outValues[0] - oCurrent) * (oCurrent * (1 - oCurrent))
				oDelta.append(delta * self.learningRate)
			
			# Calculate weight adjustments to hidden layer synapses
			nHidden = len(self.hidden)
			hDelta = []
			for h in range(nHidden):
				hCurrent = self.hidden[h].output
				nSynapses = len(self.hidden[h].synapses)
				sDelta = []
				for s in range(nSynapses):
					temp = 0
					for o in range(len(self.output)):
						dOut = outValues[o] - self.output[o].output
						temp += dOut * self.hidden[h].synapses[s].weight
					delta = temp * hCurrent * (1 - hCurrent)
					sDelta.append(delta * self.learningRate)
				hDelta.append(sDelta)
				
			# Apply new weights to output and hidden layers
			for o in range(nOutput):
				for s in range(len(self.output[o].synapses)):
					self.output[o].synapses[s].weight += oDelta[o]
			for h in range(nHidden):
				for s in range(len(self.hidden[h].synapses)):
					self.hidden[h].synapses[s].weight += hDelta[h][s]
				
# Local function to iterate a number of samples and collect success/fail metrics
def sample(inValues, outValues, nSamples, epsilon):
	nSuccess = 0
	nFailure = 0
	for i in range(nSamples):
		nn.update (inValues)
		for j in range(len(outValues)):
			# If output neuron value is within error margin, count it as a success
			if (abs(outValues[j] - nn.output[j].output) <= epsilon):
				nSuccess += 1
			else:
				nFailure += 1
	return (nSuccess, nFailure)

# Create neural network
nn = neuralnet(2, 3, 1)

# Override synapse weights to match example
#nn.output[0].synapses[0].weight = 0.3
#nn.output[0].synapses[1].weight = 0.5
#nn.output[0].synapses[2].weight = 0.9

#nn.hidden[0].synapses[0].weight = 0.8
#nn.hidden[0].synapses[1].weight = 0.2

#nn.hidden[1].synapses[0].weight = 0.4
#nn.hidden[1].synapses[1].weight = 0.9

#nn.hidden[2].synapses[0].weight = 0.3
#nn.hidden[2].synapses[1].weight = 0.5
print(nn)

for i in range(1000):
	nn.train_single_output([0,0], 0, 1)
	nn.train_single_output([0,1], 1, 1)
	nn.train_single_output([1,0], 1, 1)
	nn.train_single_output([1,1], 0, 1)

nn.update([0,0])
print(nn)
nn.update([0,1])
print(nn)
nn.update([1,0])
print(nn)
nn.update([1,1])
print(nn)

