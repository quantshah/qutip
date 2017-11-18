"""
Wigner function of a Kerr resonator
"""
from scipy.special import hyp2f1, factorial, gamma
from scipy.misc import comb
import numpy as np
import decimal


def hyper2f1m(m, c1, c1c2, eta):
    """
    The Gauss hyper geometric function summing upto m terms. It is given by:

    $$\_2F_1(-m, c_1; c_1 + c_2; \eta) = \sum_{k=0}^{k=m} (-\eta)^k {m \choose k} \frac{\tau(k + c_1) \tau(c1 + c2)}
                                                                                       {\tau(k+c_1+c_2)\tau(c_1)}$$

    Parameters
    ----------
    c1, c2: float
        Real valued arguments

    eta: complex
        Argument, complex valued

    -m: int
        The number of terms in the summation
    """
    hyp = 0

    for k in range(m+1):
        c = comb(m, k)
        gamma_num = gamma(k + c1)*gamma(c1c2)
        gamma_den = gamma(k+c1c2)*gamma(c1)

        hyp += np.power(-eta, k)* c * (gamma_num/gamma_den)

    return hyp


class Kerr(object):
    """
    A class to calculate necessary function for simulating a Kerr resonator.
    """
    def __init__(self, f, g, c):
        """
        All the system parameters go here
        
        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The index upto convergence calculated by the normalization function.
        """
        self.f = f
        self.g = g
        self.c = c

        self.m = self._convergence()
        self.N = self.normalization(self.m)

    def fm(self, m):
        """
        The function given by 

        $$ F_m(f, g, c) = (i\sqrt{g})^m \bigg[. _2F_1(-m, -c -i\frac{f}{\sqrt{g}};-2c;2)\bigg]$$

        Parameters
        ----------
        m: int
            The series index for the particular function

        Returns
        -------
        func: complex
            The value of the function used in the normalization composed of a hypergeometric function
        """
        f, g, c = self.f, self.g, self.c
        
        coeff = np.power(1j * np.sqrt(g), m)
        
        c1 = - c -(1j*f)/(np.sqrt(g))
        c1c2 = -2*c

        hyp = hyper2f1m(m, c1, c1c2, 2)

        return (coeff * hyp)


    def _convergence(self, func = None, closed_form=False, tol=1e-6):
        """
        Calculate the value of m for convergence of the given function

        Parameters
        ----------
        func: function
            A function for which the convergence needs to be determined
        closed_form: bool
            If F = 0, use the closed form expression

        Returns
        -------
        m: int
            The value of m for convergence
        """
        if func == None:
            func = self.normalization
        i = 0
        M_current = func(i)
        M_next = func(i+1)

        while(np.absolute(M_next - M_current) > tol):
            M_current = M_next
            M_next = func(i+1)
            i += 1

        return i


    def normalization(self, m = 1):
        """
        The normalization constant calculated using the formula

        $$N = \sum_{m=0}^\inf \frac{2^m}{m!} |F_m(f, g,c)|^2$$

        Returns
        -------
        N: float
            The normalization constant calculated till convergence upto m.
        """
        f, g, c = self.f, self.g, self.c

        coefficient = lambda x: np.power(2, x)/factorial(x)

        N = 0.

        for k in range(0, m+1):
            N += coefficient(k) * np.absolute(self.fm(k))

        return N


    def correlation(i, j):
        """
        The correlation function defined using the normalization and fm

        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The truncation of the index upto which the factor should be calculated

        i, j: int
            The index (i, j) of the matrix to calculate the correlation function

        Returns
        -------
        corr: float
            The value of the correlation.
        """
        f, g, c = self.f, self.g, self.c

        m = self.m
        N = self.N

        corr = 0

        coefficient = lambda x: np.power(2, x)/factorial(x)
        hyp = lambda x: self.fm(x + j)*(self.fm(x + i).conjugate())

        generator = (coefficient(idx) * hyp(idx) for idx in range(m))

        for term in generator:
            corr += term
        
        return corr/N

    def wigner(self, z):
        """
        The Wigner function

        Parameters
        ----------
        f, g, c: complex
            Dimensionless quantities formed using the system parameters

        m: int
            The truncation of the index upto which the factor should be calculated

        z: complex
            The value of z

        Returns
        -------
        W: float
            The real valued Wigner function.
        """        
        f, g, c = self.f, self.g, self.c
        m = self.m
        N = self.N

        _wigner = 0

        coefficient = lambda x: np.power(2*z.conjugate(), m)/factorial(x)
        generator = (coefficient(idx) * self.fm(idx) for idx in range(m))

        for term in generator:
            _wigner += term

        _wigner = np.power(np.absolute(_wigner), 2)

        wigner = (2./(N * np.pi)) * _wigner * np.exp(-2*np.power(abs(z),2))

        return wigner

