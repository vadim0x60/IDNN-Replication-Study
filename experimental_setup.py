import information_process
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class ZivInformationPlane():
    # The code by Ravid Schwartz-Ziv is very hard to understand 
    # (the reader is encouraged to try it themselves)
    # Solution: wrap it into this class and don't touch it with a 10-meter pole
    
    def __init__(self, X, Y, bins = np.linspace(-1, 1, 30)):
        """
        Inititalize information plane (set X and Y and get ready to calculate I(T;X), I(T;Y))
        X and Y have to be discrete
        """
        
        plane_params = dict(zip(['pys', 'pys1', 'p_YgX', 'b1', 'b', 
                                 'unique_a', 'unique_inverse_x', 'unique_inverse_y', 'pxs'], 
                                information_process.extract_probs(np.array(Y).astype(np.float), X)))
        
        plane_params['bins'] = bins
        plane_params['label'] = Y
        plane_params['len_unique_a'] = len(plane_params['unique_a'])
        del plane_params['unique_a']
        del plane_params['pys']
        
        self.X = X
        self.Y = Y
        self.plane_params = plane_params
        
    def mutual_information(self, layer_output):
        """ 
        Given the outputs T of one layer of an NN, calculate MI(X;T) and MI(T;Y)
        
        params:
            layer_output - a 3d numpy array, where 1st dimension is training objects, second - neurons
        
        returns:
            IXT, ITY - mutual information
        """
            
        information = information_process.calc_information_for_layer_with_other(layer_output, **self.plane_params)
        return information['local_IXT'], information['local_ITY']

class BufferedSequential(nn.Module):
    def __init__(self, layers, buffer_or_not):
        super(BufferedSequential, self).__init__()
        self.layers = layers
        self.buffer_or_not = buffer_or_not
        self.n_buffers = np.sum(buffer_or_not)
        
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
        
    def forward(self, x):
        if not isinstance(x, Variable):
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            x = Variable(x)

        self.buffer = []
        
        for layer, is_buffered in zip(self.layers, self.buffer_or_not):
            x = layer(x)
            if is_buffered:
                self.buffer.append(x)
                
        return x

class ReshapeLayer(nn.Module):
    def __init__(self, new_shape):
        super(ReshapeLayer, self).__init__()
        self.new_shape = new_shape
    
    def parameters(self):
        return []
    
    def forward(self, x):
        return x.view(self.new_shape)

def mutual_information_for_network_family(infoplane, buffered_networks):
    # Make the mutual information model more 'stochastic' by splitting 
    # the dataset among a family of similar neural networks
    layer_outputs = [[] for layer in range(np.sum(buffered_networks[0].buffer_or_not))]
    for x in infoplane.X:
        network = np.random.choice(buffered_networks)
        network(x)
        for i, variable in enumerate(network.buffer):
            layer_outputs[i].append(variable.data.numpy())
            
    return [infoplane.mutual_information(np.array(otp)) for otp in layer_outputs]