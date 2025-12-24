from tensor import Tensor
import numpy as np

class utils:

    @staticmethod
    def unbroadcast(grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    @staticmethod
    def unbroadcast_to(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for axis, size in enumerate(shape):
            if size == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad.reshape(shape)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        X = x.data
        k = np.sqrt(2.0 / np.pi)
        t = k * (X + 0.044715 * (X ** 3))
        tanh_t = np.tanh(t)

        out_data = 0.5 * X * (1.0 + tanh_t)
        out = Tensor(out_data, (x,))

        def _backward():
            dt_dx = k * (1.0 + 3.0 * 0.044715 * (X ** 2))
            dy_dx = 0.5 * (1.0 + tanh_t) + 0.5 * X * (1.0 - tanh_t**2) * dt_dx
            x.grad += out.grad * dy_dx

        out._backward = _backward
        return out

    @staticmethod
    def gather(x: Tensor, axis: int, indices) -> Tensor:
        idx = np.array(indices, dtype=np.int64)
        out_data = np.take_along_axis(x.data, idx, axis=axis)
        out = Tensor(out_data, (x,))

        def _backward():
            grids = list(np.indices(out_data.shape))
            ax = axis if axis >= 0 else x.data.ndim + axis
            grids[ax] = idx
            np.add.at(x.grad, tuple(grids), out.grad)

        out._backward = _backward
        return out

    @staticmethod
    def embedding(W: Tensor, ids) -> Tensor:
        ids = np.array(ids, dtype=np.int64)
        out_data = W.data[ids]
        out = Tensor(out_data, (W,))

        def _backward():
            grad2d = out.grad.reshape(-1, W.data.shape[1])
            ids1d = ids.reshape(-1)
            np.add.at(W.grad, ids1d, grad2d)

        out._backward = _backward
        return out

    @staticmethod
    def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
        X = x.data
        x_max = np.max(X, axis=axis, keepdims=True)
        x_shift = X - x_max
        exp_shift = np.exp(x_shift)
        sum_exp = np.sum(exp_shift, axis=axis, keepdims=True)
        out_data = x_shift - np.log(sum_exp)

        out = Tensor(out_data, (x,))

        def _backward():
            dy = out.grad
            softmax = np.exp(out_data)
            dy_sum = np.sum(dy, axis=axis, keepdims=True)
            x.grad += dy - softmax * dy_sum

        out._backward = _backward
        return out

    @staticmethod
    def cross_entropy(logits: Tensor, targets, axis: int = -1, reduction: str = "mean") -> Tensor:
        t = np.array(targets, dtype=np.int64)
        lp = utils.log_softmax(logits, axis=axis)

        ax = axis if axis >= 0 else logits.data.ndim + axis
        idx = np.expand_dims(t, axis=ax)
        picked = utils.gather(lp, ax, idx)
        loss_per = -np.squeeze(picked.data, axis=ax)

        if reduction == "mean":
            out_data = loss_per.mean()
        elif reduction == "sum":
            out_data = loss_per.sum()
        elif reduction == "none":
            out_data = loss_per
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        out = Tensor(out_data, (logits,))

        def _backward():
            soft = np.exp(lp.data)      # (..., C)
            grad = soft.copy()

            grids = list(np.indices(t.shape))
            grids.insert(ax, t)
            grad[tuple(grids)] -= 1.0

            upstream = out.grad
            if reduction == "mean":
                grad *= upstream / t.size
            elif reduction == "sum":
                grad *= upstream
            else:
                grad *= np.expand_dims(upstream, axis=ax)

            logits.grad += grad

        out._backward = _backward
        return out
