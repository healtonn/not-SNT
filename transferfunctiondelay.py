from numpy import linspace, dot, transpose, nan_to_num, squeeze, zeros, atleast_1d
from scipy import integrate, interpolate
from scipy.signal import lti, dlti, TransferFunction


class TransferFunctionDelay(TransferFunction):
    def __init__(self, inputdelay, *args, **kwargs):
        super(TransferFunctionDelay, self).__init__(*args, **kwargs)
        self.d = inputdelay


def lsim3(system, U=None, T=None, X0=None, **kwargs):
    """
    Metoda pro simulaci LTI s casovym spozdenim.
    Pouze jsem zkopiroval metodu lsim2 z oficialni implementace
    a udelal na dvou mistech upravy aby bylo brano v potaz spozdeni
    https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/ltisys.py#L1745-L1853
    kopirovano dne: 26.05.2018
    """
    if isinstance(system, lti):
        sys = system._as_ss()
    elif isinstance(system, dlti):
        raise AttributeError('lsim2 can only be used with continuous-time '
                             'systems.')
    else:
        sys = lti(*system)._as_ss()
    # delay = 50
    if X0 is None:
        X0 = zeros(sys.B.shape[0], sys.A.dtype)

    if T is None:
        T = linspace(0, 10.0, 101)

    T = atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    if U is not None:
        U = atleast_1d(U)
        if len(U.shape) == 1:
            U = U.reshape(-1, 1)
        sU = U.shape
        if sU[0] != len(T):
            raise ValueError("U must have the same number of rows "
                             "as elements in T.")

        if sU[1] != sys.inputs:
            raise ValueError("The number of inputs in U (%d) is not "
                             "compatible with the number of system "
                             "inputs (%d)" % (sU[1], sys.inputs))
        # Create a callable that uses linear interpolation to
        # calculate the input at any time.
        ufunc = interpolate.interp1d(T, U, kind='linear',
                                     axis=0, bounds_error=False)

        def fprime(x, t, sys, ufunc):
            """The vector field of the linear system."""
            return dot(sys.A, x) + squeeze(dot(sys.B, nan_to_num(ufunc([t]))))

        xout = integrate.odeint(fprime, X0, T, args=(sys, ufunc), **kwargs)
        yout = dot(sys.C, transpose(xout)) + dot(sys.D, transpose(U))
    else:
        def fprime(x, t, sys):
            """The vector field of the linear system."""
            return dot(sys.A, x)

        xout = integrate.odeint(fprime, X0, T, args=(sys,), **kwargs)
        yout = dot(sys.C, transpose(xout))

    return T, squeeze(transpose(yout)), xout
