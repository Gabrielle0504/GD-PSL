import torch
import numpy as np


class BaseProblem:
    """Shared tensor/device utilities for all benchmark definitions."""

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Keep bounds on the same device as the evaluated decision tensor."""
        self.lbound = self.lbound.to(device=x.device)
        self.ubound = self.ubound.to(device=x.device)
        return x

    @staticmethod
    def _constraint_violation(*constraints: torch.Tensor) -> torch.Tensor:
        """Aggregate positive constraint violations with one consistent dtype."""
        values = torch.stack(constraints).to(dtype=torch.float64)
        return torch.clamp(-values, min=0.0).sum(dim=0).to(dtype=torch.float32)

def get_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        're21': RE21,
        're24': RE24,
        're31': RE31,
        're32': RE32,
        're34': RE34,
        're35': RE35,
        're37': RE37,
        'dtlz2': DTLZ2,
        'dtlz7': DTLZ7,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)


def closest_value(arr, val):
    '''
    Get closest value to val in arr
    '''
    return arr[torch.argmin(torch.abs(arr[:, None] - val), axis=0)]

def div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0

    '''
    results = x1 * 0.0
    results[x2 != 0.0] = x1[x2 != 0.0] / x2[x2 != 0.0]

    return results

class RE21(BaseProblem):
    def __init__(self, n_dim = 4):

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.ideal_point = np.array([1237.8414230005742, 0.002761423749158419])
        self.nadir_point = np.array([2886.3695604236013, 0.039999999999998245])


    def evaluate(self, x):

        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        x = self._prepare_input(x)
        f1 =  L * ((2 * x[:,0]) + np.sqrt(2.0) * x[:,1] + torch.sqrt(x[:,2]) + x[:,3])
        f2 =  ((F * L) / E) * ((2.0 / x[:,0]) + (2.0 * np.sqrt(2.0) / x[:,1]) - (2.0 * np.sqrt(2.0) / x[:,2]) + (2.0 /  x[:,3]))

        f1 = f1
        f2 = f2

        # f = torch.stack([f1, f2], dim = 1)
        objs = torch.stack([f1,f2]).T

        return objs

class RE24(BaseProblem):
    def __init__(self, n_dim = 2):

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.ones(n_dim).float() * 0.5
        self.ubound = torch.tensor([4, 50]).float()
        self.nadir_point = [481.608088535, 44.2819047619]

    def evaluate(self, x):

        x = self._prepare_input(x)
        x1 = x[:,0]
        x2 = x[:,1]

        #First original objective function
        f1 = x1 + (120 * x2)

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5
        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g1 = 1 - (sigma_b / sigma_b_max)
        g2 = 1 - (tau / tau_max)
        g3 = 1 - (delta / delta_max)
        g4 = 1 - (sigma_b / sigma_k)


        f2 = self._constraint_violation(g1,g2,g3,g4)

        objs = torch.stack([f1,f2]).T

        return objs

class RE31(BaseProblem):
    def __init__(self, n_dim = 3):


        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([1e-5, 1e-5, 1]).float()
        self.ubound = torch.tensor([100, 100, 3]).float()
        self.nadir_point = [500.002668442, 8246211.25124, 19359919.7502]

    def evaluate(self, x):

        x = self._prepare_input(x)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]

        # First original objective function
        f1 = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        # Second original objective function
        #f2 = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x3 * x1)
        f2 = div((20.0 * torch.sqrt(16.0 + (x3 * x3))),(x3 * x1))

        # Constraint functions
        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000 - div( (80.0 * torch.sqrt(1.0 + x3 * x3)) , (x3 * x2))

        f3 = self._constraint_violation(g1,g2,g3)


        objs = torch.stack([f1,f2,f3]).T

        return objs

class RE32(BaseProblem):
    def __init__(self, n_dim = 4):


        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0.125, 0.1, 0.1, 0.125]).float()
        self.ubound = torch.tensor([5, 10, 10, 5]).float()
        self.nadir_point = [37.7831517014, 17561.6, 425062976.628]

    def evaluate(self, x):

        x = self._prepare_input(x)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f2 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g1 = tauMax - tau
        g2 = sigmaMax - sigma
        g3 = x4 - x1
        g4 = PC - P

        f3 = self._constraint_violation(g1,g2,g3,g4)

        objs = torch.stack([f1,f2,f3]).T

        return objs

class RE34(BaseProblem):
    def __init__(self, n_dim = 5):


        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([1, 1, 1, 1, 1]).float()
        self.ubound = torch.tensor([3, 3, 3, 3, 3]).float()
        self.nadir_point = [1695.2002035, 10.7454, 0.26399999965]

    def evaluate(self, x):

        x = self._prepare_input(x)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]

        f1 = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f2 = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (0.3628 * x2 * x4)  - (0.1106 * x1 * x1)  - (0.3437 * x3 * x3) + (0.1764 * x4 * x4)
        f3 = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)


        objs = torch.stack([f1,f2,f3]).T

        return objs


class RE35(BaseProblem):
    def __init__(self, n_dim = 7):


        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0 ]).float()
        self.ubound = torch.tensor([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]).float()
        self.nadir_point = [6634.56208, 1695.96387746, 397.358927317]

    def evaluate(self, x):

        x = self._prepare_input(x)
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = torch.round(x[:,2])#x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]


        # First original objective function (weight)
        f1 = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

        # Second original objective function (stress)
        tmpVar = torch.pow((745.0 * x4) / (x2 * x3), 2.0)  + 1.69 * 1e7
        f2 =  torch.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions
        g1 = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g2 = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g3 = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g4 = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g5 = -(x2 * x3) + 40.0
        g6 = -(x1 / x2) + 12.0
        g7 = -5.0 + (x1 / x2)
        g8 = -1.9 + x4 - 1.5 * x6
        g9 = -1.9 + x5 - 1.1 * x7
        g10 =  -f2 + 1300.0
        tmpVar = torch.pow((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g11 = -torch.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0

        #g = np.where(g < 0, -g, 0)
        #f3 = g[0] + g[1] + g[2] + g[3] + g[4] + g[5] + g[6] + g[7] + g[8] + g[9] + g[10]

        f3 = self._constraint_violation(g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11)


        objs = torch.stack([f1,f2,f3]).T

        return objs


class RE37(BaseProblem):
    def __init__(self, n_dim = 4):


        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):

        x = self._prepare_input(x)
        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]

        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)


        objs = torch.stack([f1,f2,f3]).T

        return objs

def _default_dtlz_dimension(n_obj, three_objective_dimension):
    """Keep the existing three-objective defaults while scaling M objectives."""
    if n_obj < 2:
        raise ValueError("DTLZ problems require at least two objectives.")
    return three_objective_dimension + (n_obj - 3)


def _dtlz_spherical_objectives(x, g, n_obj, alpha=1.0):
    """Build the shared DTLZ2/3/4 hyperspherical objective mapping."""
    theta = x[:, :n_obj - 1] ** alpha * (np.pi / 2)
    values = []
    scale = 1 + g
    for objective in range(n_obj):
        value = scale
        for angle in range(n_obj - objective - 1):
            value = value * torch.cos(theta[:, angle])
        if objective:
            value = value * torch.sin(theta[:, n_obj - objective - 1])
        values.append(value)
    return torch.stack(values, dim=1)


class DTLZ2(BaseProblem):
    def __init__(self, n_obj=3, n_dim=None):
        self.n_obj = n_obj
        self.n_dim = _default_dtlz_dimension(n_obj, 12) if n_dim is None else n_dim
        if self.n_dim < self.n_obj:
            raise ValueError("n_dim must be at least n_obj for DTLZ2.")
        self.lbound = torch.zeros(self.n_dim).float()
        self.ubound = torch.ones(self.n_dim).float()
        self.nadir_point = [1] * self.n_obj

    def evaluate(self, x):
        x = self._prepare_input(x)
        g = torch.sum((x[:, self.n_obj - 1:] - 0.5) ** 2, dim=1)
        return _dtlz_spherical_objectives(x, g, self.n_obj)

class DTLZ7(BaseProblem):
    """Benchmark MOP proposed by Deb, Thiele, Laumanns, and Zitzler."""

    def __init__(self, n_obj=3, n_dim=None):
        self.n_obj = n_obj
        # PlatEMO default setting: D = M + 19
        self.n_dim = n_obj + 19 if n_dim is None else n_dim
        self.lbound = torch.zeros(self.n_dim).float()
        self.ubound = torch.ones(self.n_dim).float()
        # Component-wise maximum over the Pareto front:
        # x_i <= 0.859401 (i = 1, ..., M-1) and f_M <= 2M (obtained for g = 0)
        self.nadir_point = [0.859401] * (n_obj - 1) + [2.0 * n_obj]

    def evaluate(self, x):
        x = self._prepare_input(x)
        M = self.n_obj

        # g = 1 + 9 * mean(x_M, ..., x_D)
        g = 1 + 9 * torch.mean(x[:, M - 1:], dim=1)

        # f_1, ..., f_{M-1} = x_1, ..., x_{M-1}
        xp = x[:, :M - 1]

        # f_M = (1 + g) * (M - sum(x_i / (1 + g) * (1 + sin(3 * pi * x_i))))
        h = torch.sum(xp / (1 + g).unsqueeze(1) * (1 + torch.sin(3 * np.pi * xp)), dim=1)
        fM = (1 + g) * (M - h)

        objs = torch.cat([xp, fM.unsqueeze(1)], dim=1)

        return objs
