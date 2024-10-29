# @ The code in the present module was copied, modified, merged and distributed 
# for educational purposes under the MIT License reported below. The original 
# code can be found at https://github.com/amanchokshi/non-normal?tab=MIT-1-ov-file.

# MIT License

# Copyright (c) 2023 Aman Chokshi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Class to generate a non-normal field with known mean, variance, skewness and (excess) kurtosis.
"""

import numpy as np
from scipy import optimize
import warnings


class Fleishman:
    """
    Generate a non-normal field with known mean, variance, skewness and (excess) kurtosis.

    This is based on Fleishman method, which creates a cubic polynomial
    with a normal seed field, which is non-normal.

    Y = a + bX + cX^2 + dX^3

    The trick is to tune the four polynomial coefficient (a, b, c, d) such that
    the resulting non-normal field (Y) has the desired mean, var, skew & ekurt.

    Parameters
    ----------
    size: int
        size of output field
        default: 2**20

    mean: float
        mean
        default: 0

    var: float
        varience
        default: 1

    skew: float
        skewness
        default: 0

    ekurt: float
        excess kurtosis
        default: 0

    seed: int
        random number generator seed value
        default: 42

    max_iter: int
        maximum number of iterations to iterations for
        the newton method to converge

    converge: float
        newton method convergence threshold
        default: 1e-30

    References
    ----------
        - https://link.springer.com/article/10.1007/BF02293811
        - https://support.sas.com/content/dam/SAS/support/en/books/simulating-data-with-sas/65378_Appendix_D_Functions_for_Simulating_Data_by_Using_Fleishmans_Transformation.pdf
        - https://www.diva-portal.org/smash/get/diva2:407995/FULLTEXT01.pd
        - https://pubmed.ncbi.nlm.nih.gov/34779511/
        - https://gist.github.com/zeimusu/7432603b85dc6406c6ea
        - https://link.springer.com/article/10.1007/BF02293687
    """
    def __init__(
        self
      , size=2**20
      , mean=0.
      , var=1.
      , skew=0.
      , ekurt=0.
      , seed=42
      , max_iter=128
      , converge=1e-10
      , verbose=0
      , coeff=None
    ):
        self.size = size
        self.mean = mean
        self.var = var
        self.skew = skew
        self.ekurt = ekurt
        self.seed = seed
        self.max_iter = max_iter
        self.converge = converge
        self.verbose = verbose
        self.coeff = coeff

        # Seed random number generator
        self.rng = np.random.default_rng(self.seed)
    

    def fl_func(self, x):
        """
        Define a function which will have roots.

        iff: the coeffs give the desired skew and ekurtosis

        Parameters
        ----------
        b, c, d: float
            Fleishman Polynomial Coefficients

        Returns
        -------
        function: float
        """
        b, c, d = x

        f1 = (b**2) + 6 * (b * d) + 2 * (c**2) + 15 * (d**2) - 1
        f2 = 2 * c * ((b**2) + 24 * (b * d) + 105 * (d**2) + 2) - self.skew
        f3 = 24 * (
            (b * d)
            + (c**2) * (1 + (b**2) + 28 * (b * d))
            + (d**2) * (12 + 48 * (b * d) + 141 * (c**2) + 225 * (d**2))
        ) - self.ekurt

        return f1**2 + f2**2 + f3**2


    def fl_ic(self):
        """Initial condition estimate of the Fleishman coefficients."""

        b0 = (
            0.95357
            - 0.05679 * self.ekurt
            + 0.03520 * self.skew**2
            + 0.00133 * self.ekurt**2
        )
        c0 = 0.10007 * self.skew + 0.00844 * self.skew**3
        d0 = 0.30978 - 0.31655 * b0

        return (b0, c0, d0)
    

    def fl_coeff(self):
        """Compute Fleishman coefficients."""

        # Feasibility condition for the existence of solutions
        # TO-DO: check correctness of this expression!
        ekurt_thresh = -1.13168 + 1.58837 * self.skew**2

        if self.ekurt < ekurt_thresh:
            warnings.warn(
                f"For the Fleishman method to work with:\n\tmean: {self.mean:.2f}\n\tvari: {self.var:.2f}\n\tskew: {self.skew:.2f}\nThe value of [ekurt] must be >= [{ekurt_thresh:.4f}]\nUsing [ekurt] threshold value."
            )

            self.ekurt = ekurt_thresh

        # Initial condition
        x0 = self.fl_ic() # self.rng.normal(size=3)

        # Optimization
        res = optimize.minimize(
            lambda x: self.fl_func(x)
          , x0
          , method='nelder-mead' # 'nelder-mead' 'BFGS'
          , options={'xatol': self.converge, 'disp': self.verbose} # 'xatol': self.converge
        )

        b, c, d = res.x

        self.coeff = {"a": -c, "b": b, "c": c, "d": d}


    def gen_field(self, size=None):
        """Generate the non-normal Fleishman field."""

        if size is None:
            size = self.size

        # Feasibility condition for the existence of solutions
        # TO-DO: check correctness of this expression!
        ekurt_thresh = -1.13168 + 1.58837 * self.skew**2

        if self.ekurt < ekurt_thresh:
            warnings.warn(
                f"For the Fleishman method to work with:\n\tmean: {self.mean:.2f}\n\tvari: {self.var:.2f}\n\tskew: {self.skew:.2f}\nThe value of [ekurt] must be >= [{ekurt_thresh:.4f}]\nUsing [ekurt] threshold value."
            )

            self.ekurt = ekurt_thresh

        # Find Fleishman coefficients
        self.fl_coeff()
        b, c, d = self.coeff['b'], self.coeff['c'], self.coeff['d']

        # Normal sample for Fleishman method
        X = self.rng.normal(size=int(size))

        # Generate the field from the Fleishman polynomial
        # Then scale it by the std and mean
        self.field = (-1 * c + X * (b + X * (c + X * d))) * np.sqrt(
            self.var
        ) + self.mean

            