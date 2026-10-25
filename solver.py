from typing import Callable, Tuple
import numpy as np


def profile_step(x: np.ndarray, a: float = 3.0, b: float = 7.0) -> np.ndarray:
    return np.where((x >= a) & (x <= b), 1.0, 0.0)


def profile_gaussian(x: np.ndarray, xs: float = 5.0, delta: float = 0.8) -> np.ndarray:
    return np.exp(-((x - xs) ** 2) / (2.0 * delta ** 2))


def profile_parabola(x: np.ndarray, a: float = 3.0, b: float = 7.0) -> np.ndarray:
    cond = (x >= a) & (x <= b)
    res = np.zeros_like(x)
    res[cond] = (x[cond] - a) * (b - x[cond])
    max_val = np.max(res)
    return res / max_val if max_val > 0 else res


class AdvectionSolver:
    def __init__(self, L: float = 20.0, N: int = 200, u: float = 1.0):
        self.L = L
        self.N = N
        self.u = u
        self.dx = L / N
        self.x = np.linspace(0.0, L - self.dx, N)

    def exact_solution(self, initial_func: Callable[[np.ndarray], np.ndarray], t: float) -> np.ndarray:
        x_shifted = np.mod(self.x - self.u * t, self.L)
        return initial_func(x_shifted)

    def step_upwind(self, s: np.ndarray, Co: float) -> np.ndarray:
        s_left = np.roll(s, 1)
        return s - Co * (s - s_left)

    def step_ftcs(self, s: np.ndarray, Co: float) -> np.ndarray:
        s_right = np.roll(s, -1)
        s_left = np.roll(s, 1)
        return s - 0.5 * Co * (s_right - s_left)

    def step_lax(self, s: np.ndarray, Co: float) -> np.ndarray:
        s_right = np.roll(s, -1)
        s_left = np.roll(s, 1)
        return 0.5 * (s_right + s_left) - 0.5 * Co * (s_right - s_left)

    def step_lax_wendroff(self, s: np.ndarray, Co: float) -> np.ndarray:
        s_right = np.roll(s, -1)
        s_left = np.roll(s, 1)
        conv = 0.5 * Co * (s_right - s_left)
        diff = 0.5 * (Co ** 2) * (s_right - 2.0 * s + s_left)
        return s - conv + diff

    def solve(
        self,
        scheme_name: str,
        initial_func: Callable[[np.ndarray], np.ndarray],
        Co: float,
        t_final: float
    ) -> Tuple[np.ndarray, float]:
        dt = Co * self.dx / self.u
        num_steps = int(np.round(t_final / dt))
        s = initial_func(self.x).copy()

        schemes = {
            "upwind": self.step_upwind,
            "ftcs": self.step_ftcs,
            "lax": self.step_lax,
            "lax_wendroff": self.step_lax_wendroff,
        }
        if scheme_name not in schemes:
            raise ValueError(f"Unknown scheme: {scheme_name}")

        step_fn = schemes[scheme_name]

        for _ in range(num_steps):
            s = step_fn(s, Co)
            if np.any(np.isnan(s)) or np.any(np.abs(s) > 1e10):
                break

        return s, num_steps * dt