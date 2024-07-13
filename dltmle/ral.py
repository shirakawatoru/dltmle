import numpy as np

class RALEstimate:
    def __init__(self, est, ic):
        self.est = est
        self.ic = ic
        self.n = len(ic)
        
    @property
    def se(self):
        return np.sqrt((self.ic ** 2).mean() / self.n)
    
    # operations with the delta method

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est + other, self.ic)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est + other.est, self.ic + other.ic)
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and '{}'".format(type(self).__name__, type(other).__name__))
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est - other, self.ic)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est - other.est, self.ic - other.ic)
        else:
            raise TypeError("Unsupported operand type(s) for -: '{}' and '{}'".format(type(self).__name__, type(other).__name__))
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est * other, self.ic * other)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est * other.est, self.ic * other.est + self.est * other.ic)
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(self).__name__, type(other).__name__))        
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est / other, self.ic / other)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est / other.est, self.ic / other.est - self.est * other.ic / other.est ** 2)
        else:
            raise TypeError("Unsupported operand type(s) for /: '{}' and '{}'".format(type(self).__name__, type(other).__name__))
    
    def __neg__(self):
        return RALEstimate(-self.est, -self.ic)        

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est + other, self.ic)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est + other.est, self.ic + other.ic)
        else:
            raise TypeError("Unsupported operand type(s) for +: '{}' and '{}'".format(type(other).__name__, type(self).__name__))
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(other - self.est, self.ic)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(other.est - self.est, other.ic - self.ic)
        else:
            raise TypeError("Unsupported operand type(s) for -: '{}' and '{}'".format(type(other).__name__, type(self).__name__))
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(self.est * other, self.ic * other)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(self.est * other.est, self.ic * other.est + self.est * other.ic)
        else:
            raise TypeError("Unsupported operand type(s) for *: '{}' and '{}'".format(type(other).__name__, type(self).__name__))
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return RALEstimate(other / self.est, - other * self.ic / self.est ** 2)
        elif isinstance(other, RALEstimate):
            assert self.n == other.n
            return RALEstimate(other.est / self.est, other.ic / self.est - other.est * self.ic / self.est ** 2)
        else:
            raise TypeError("Unsupported operand type(s) for /: '{}' and '{}'".format(type(other).__name__, type(self).__name__))

    def __repr__(self):
        est, se = self.est, self.se
        return "estimate: {:.3f}, SE: {:.3f}, 95% CI: ({:.3f}, {:.3f})".format(est, se, est - 1.96 * se, est + 1.96 * se)
