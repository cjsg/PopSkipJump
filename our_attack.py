import logging
import numpy as np
from hopskip import HopSkipJumpAttack


class OurAttack(HopSkipJumpAttack):

    def geometric_progression_for_stepsize(
            self, x, update, dist, decision_function, current_iteration, original=None
    ):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        xx = self.search_for_boundary(x, original, decision_function, 1 / (28*28*28))
        epsilon = dist / np.sqrt(current_iteration)
        count = 1
        while True:
            if count % 200 == 0:
                logging.warning("Decreased epsilon {} times".format(count))
            updated = np.clip(xx + epsilon * update, self.clip_min, self.clip_max)
            success = (decision_function(updated[None], self.sampling_freq, remember=self.remember))[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover
            count += 1
        return epsilon

    def search_for_boundary(self, x_t, x_star, decision_function, theta_det):
        is_adv = (decision_function(x_t[None], 1, remember=self.remember))[0]
        if not is_adv:
            x_tt = 2 * x_t - x_star
            while True:
                res = self.binary_search(x_t, x_tt, theta_det, decision_function)
                if np.any(res != x_tt):
                    x_tt = res
                    break
                x_tt = res
        else:
            x_tt = self.binary_search(x_star, x_t, theta_det, decision_function)
        return x_tt

    def binary_search(self, x_star, x_t, theta_det, decision_function):
        high, low = 1, 0
        while high - low > theta_det:
            mid = (high + low) / 2.0
            x_mid = (1 - mid) * x_star + mid * x_t
            is_adv = (decision_function(x_mid[None], 1, remember=self.remember))[0]
            if is_adv:
                high = mid
            else:
                low = mid
        out = (1 - high) * x_star + high * x_t
        return out