import numpy as np
from dataclasses import dataclass, field
from collections.abc import Callable
from scipy.optimize import fsolve

from .dtw import gen_cost_matrix, backtracking

v_e = 3e8
