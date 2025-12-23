import math

class Scalar:

    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._backward = None
        self.prev = children

    def __repr__(self):
        return f"Scalar={self.data}"

    def __mul__(self, other):
        output = Scalar(self.data * other.data, (self,other))
        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = _backward
        return output
    
    def __add__(self, other):            
        output = Scalar(self.data + other.data, (self,other))
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward
        return output
    
    def __sub__(self,other):
        output = Scalar(self.data + (-other.data), (self,other))
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward
        return output

    def __neg__(self):
        output = Scalar(-self.data, (self,))
        def _backward():
            self.grad += -output.grad
        output._backward = _backward
        return output
    
    def __div__(self, other):
        output = Scalar(self.data * other.data**-1, (self,other))
        def _backward():
            self.grad += output.grad / other.data
            other.grad += -output.grad * self.data / (other.data * other.data)
        output._backward = _backward
        return output

    def exp(self):
        output = Scalar(math.exp(self.data), (self,))
        def _backward():
            self.grad += output.grad * output.data
        output._backward = _backward
        return output

    def __pow__(self, exponent):
        output = Scalar(self.data ** exponent, (self,))
        def _backward():
            self.grad += output.grad * (exponent * (self.data ** (exponent - 1)))
        output._backward = _backward
        return output
    
    def relu(self):
        output = Scalar(max(0,self.data), (self,))
        def _backward():
            self.grad += output.grad * (1.0 if max(0,self.data) != 0 else 0.0)
        output._backward = _backward
        return output

    def sigmoid(self):
        output = Scalar( (1 + math.exp(self.data))**-1)
        def _backward():
            self.grad += output.grad * (output.data * (1.0 - output.data))
        output._backward = _backward
        return output
    
    def backward(self):
        self.grad = 1.0

        topo = []
        visited = set()
        stack = [(self, False)]  #(node, expanded?)

        while len(stack) > 0:
            node, expanded = stack.pop()

            if expanded:
                topo.append(node)
                continue

            if node in visited:
                continue

            visited.add(node)

            #postorder
            stack.append((node, True))
            for child in reversed(node.prev):
                if child not in visited:
                    stack.append((child, False))

        #run backprop
        for node in reversed(topo):
            if node._backward is not None:
                node._backward()