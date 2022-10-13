"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1, input_2 = node.inputs
        return out_grad / input_2, -1 * out_grad * input_1 / input_2 ** 2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        return array_api.swapaxes(a, len(a.shape) - 2, len(a.shape) - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1 = node.inputs[0]
        s_input_1, s_out_grad = list(input_1.shape), list(out_grad.shape)
        axes = [i for i in range(len(s_input_1)) if s_input_1[i] != s_out_grad[i]]
        assert len(axes) == 2
        print('axes = {}'.format(axes))
        print('outgrad, \n', out_grad.shape)
        return transpose(out_grad, tuple(axes))
        ## END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1 = node.inputs[0]
        return reshape(out_grad.numpy(), input_1.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1 = node.inputs[0]
        s_input_1, s_out_grad = list(input_1.shape), list(out_grad.shape)
        if len(s_input_1) != len(s_out_grad):
            s_input_1 = [1] * (len(s_out_grad) - len(s_input_1)) + s_input_1[:]        
        axes = [i for i in range(len(s_input_1)) if s_input_1[i] != s_out_grad[i]]

        return summation(out_grad, tuple(axes)).reshape(input_1.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     input_1 = node.inputs[0]
    #     return Tensor(array_api.ones(input_1.shape))
    #     ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Tensors are not subscriptable in needle.
        # Call `reshape` alternatively to add axes.
        axes_shape = list(node.inputs[0].shape)
        if self.axes:
            for i in self.axes:
                axes_shape[i] = 1
        else:
            axes_shape = [1] * len(axes_shape)
        return (broadcast_to(reshape(out_grad, axes_shape),
                             node.inputs[0].shape),) # a deliberate tuple
        ## END YOUR SOLUTION

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
        input_1, input_2 = node.inputs
        len1, len2 = len(input_1.shape), len(input_2.shape)
        out1 = out_grad@array_api.transpose(input_2)
        out2 = array_api.transpose(input_1)@out_grad
        len_out_grad = len(out_grad.shape)
        if len2 != len_out_grad:
            out2 = summation(out2, tuple([i for i in range(len_out_grad - len2)]))
        if len1 != len_out_grad:
            out1 = summation(out1, tuple([i for i in range(len_out_grad - len1)]))
        return out1, out2
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1 = node.inputs[0]
        return out_grad * -1 * array_api.ones(input_1.shape)
        ### END YOUR SOLUTION
    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     return (negate(out_grad),) # a deliberate tuple
        ### END YOUR SOLUTION

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_1 = node.inputs[0]
        return out_grad / input_1
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     input_1 = node.inputs[0]
    #     return out_grad * input_1
    #     ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (multiply(out_grad,
                         exp(node.inputs[0])),) # a deliberate tuple

def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

