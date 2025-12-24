import numpy as np
from util import utils

class Tensor:

    def __init__(self, data, children=()):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.shape = self.data.shape
        self._backward = lambda: None 
        self._prev = children

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self,other))
        def _backward():
            dself = out.grad * other.data
            dother = out.grad * self.data
            self.grad  += utils.unbroadcast(dself,  self.data.shape)
            other.grad += utils.unbroadcast(dother, other.data.shape)
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            A = self.data
            B = other.data
            G = out.grad

            A_was_1d = (A.ndim == 1)
            B_was_1d = (B.ndim == 1)

            A2 = A[None, :] if A_was_1d else A          # (..., 1, n) if needed
            B2 = B[:, None] if B_was_1d else B          # (..., n, 1) if needed
            G2 = G

            if A_was_1d and B_was_1d:
                # scalar output
                G2 = np.array([[G]])
            elif A_was_1d and not B_was_1d:
                # output (..., p) treat as (..., 1, p)
                G2 = G[..., None, :]
            elif not A_was_1d and B_was_1d:
                # output (..., m) treat as (..., m, 1)
                G2 = G[..., :, None]

            dA2 = G2 @ np.swapaxes(B2, -1, -2)
            dB2 = np.swapaxes(A2, -1, -2) @ G2

            if A_was_1d:
                dA2 = dA2[..., 0, :]   # (..., n)
            if B_was_1d:
                dB2 = dB2[..., :, 0]   # (..., n)

            self.grad  += utils.unbroadcast_to(dA2, self.data.shape)
            other.grad += utils.unbroadcast_to(dB2, other.data.shape)

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
         
        out = Tensor(self.data + other.data, (self,other))
        def _backward():
            self.grad += utils.unbroadcast(out.grad, self.data.shape)
            other.grad += utils.unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data - other.data, (self, other))
        def _backward():
            self.grad  += utils.unbroadcast(out.grad, self.data.shape)
            other.grad += utils.unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out
    
    #shape operations
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
    def transpose(self, axis1=-2, axis2=-1):
        out = Tensor(np.swapaxes(self.data, axis1, axis2), (self,))

        def _backward():
            self.grad += np.swapaxes(out.grad, axis1, axis2)

        out._backward = _backward
        return out

    @property
    def T(self):  # convenience for 2D
        return self.transpose(-2, -1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            # If sum removed dims, put them back so broadcast works
            if axis is not None and not keepdims:
                ax = axis if isinstance(axis, tuple) else (axis,)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)
            self.grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else np.prod(np.array(self.data.shape)[axis])
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                ax = axis if isinstance(axis, tuple) else (axis,)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)
            self.grad += np.broadcast_to(g, self.data.shape) / denom

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,))

        def _backward():
            # scatter-add grad back into the sliced positions
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other))

        def _backward():
            # out = self / other
            # dself  = out.grad * (1/other)
            # dother = out.grad * (-self/other^2)
            dself = out.grad / other.data
            dother = out.grad * (-self.data / (other.data ** 2))

            self.grad  += utils.unbroadcast(dself,  self.data.shape)
            other.grad += utils.unbroadcast(dother, other.data.shape)

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, (self,))

        def _backward():
            # d/dx exp(x) = exp(x)
            self.grad += out.grad * out_data

        out._backward = _backward
        return out

    def log(self):
        out_data = np.log(self.data)
        out = Tensor(out_data, (self,))

        def _backward():
            # d/dx log(x) = 1/x
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, (self,))

        def _backward():
            g = out.grad

            if axis is not None and not keepdims:
                ax = axis if isinstance(axis, tuple) else (axis,)
                for a in sorted(ax):
                    g = np.expand_dims(g, a)

            out_b = np.broadcast_to(out_data if keepdims else np.broadcast_to(out_data, g.shape), g.shape)
            x_b = self.data
            if axis is not None and not keepdims:
                out_b = np.broadcast_to(g * 0 + out_b, x_b.shape)

            max_b = self.data.max(axis=axis, keepdims=True)
            mask = (self.data == max_b)

            if axis is None:
                count = mask.sum()
            else:
                count = mask.sum(axis=axis, keepdims=True)

            g_keep = out.grad
            if axis is not None and not keepdims:
                ax = axis if isinstance(axis, tuple) else (axis,)
                for a in sorted(ax):
                    g_keep = np.expand_dims(g_keep, a)
            g_keep = np.broadcast_to(g_keep, self.data.shape)

            self.grad += mask * (g_keep / count)

        out._backward = _backward
        return out
    
    def relu(self):
        out_data = np.maximum(0.0, self.data)
        out = Tensor(out_data, (self,))

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out
    
    #=====#
    def zero_grad(self):
        """Recursively zero grads in the whole graph rooted at this tensor."""
        visited = set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            v.grad = np.zeros_like(v.data)
            for p in v._prev:
                build(p)

        build(self)

    def backward(self, grad=None):
        topo = []
        visited = set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        # Seed gradient
        if grad is None:
            if self.data.shape != ():
                raise ValueError(
                    "backward() on non-scalar requires grad argument with same shape as tensor."
                )
            self.grad = np.ones_like(self.data)
        else:
            g = np.array(grad, dtype=float)
            if g.shape != self.data.shape:
                raise ValueError(f"grad shape {g.shape} must match tensor shape {self.data.shape}")
            self.grad = g

        # Run reverse topo
        for v in reversed(topo):
            v._backward()