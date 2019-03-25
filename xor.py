
#
# Neural network example
#
# XOR gate simulation
#

import random
from math import sqrt, exp

def dtanh(self, y):
	return 1.0-y*y
	
def sigmoid_derivative(x):
	return x * (1-x)

#
# Class representation of a synapse that connects the output of one neuron
#   to the input of another, with some weighting factor applied
#
class synapse:
	def __init__(self, n):
		#self.weight = random.random()
		self.weight = 0.5
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
		self.learningRate = 0.1
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

	def train(self, inValues, outValues, nCycles):
		for i in range(0, nCycles):
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
	
#[nSuccess, nFailure] = sample([0,1], 1000)
#print ("nSuccess = " + '{:5d}'.format(nSuccess) + "    nFailure = " + '{:5d}'.format(nFailure))

nn.train([0,1], [1], 10000)
print(nn)

print (sample([0,1],[1],100,0.05))

#for i in range(10000):
#	nn.train([0,1], [1], 1)
#	nn.train([1,1], [0], 1)


#print (nn.output[0].value)

#(nSuccess, nFailure) = sample([0,1], 1000)
#print ("nSuccess = " + '{:5d}'.format(nSuccess) + "    nFailure = " + '{:5d}'.format(nFailure))

#print(nn)

