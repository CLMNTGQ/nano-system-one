"""Check a compiled host scorer against the trained checkpoint on its entire domain."""
import argparse
import ctypes as ct
import itertools
from pathlib import Path

import numpy as np
import torch

from episodes import FEATURES, HISTORY
from train import TemporalScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    native = ct.CDLL(str(args.library.resolve())).nso_select
    native.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint, ct.c_int, ct.POINTER(ct.c_float)]
    native.restype = ct.c_int
    model = TemporalScorer(0)
    with np.load(args.checkpoint) as weights:
        model.load_state_dict({key: torch.tensor(weights[key]) for key in weights.files})
    domain = np.array(list(itertools.product(range(3), repeat=HISTORY)))
    with torch.no_grad():
        expected = model(torch.tensor(FEATURES[domain])).numpy()
    for window, logits in zip(domain, expected):
        history = (ct.c_uint8 * HISTORY)(*window)
        output = (ct.c_float * 3)()
        assert native(history, 7, 0, output) == -1
        for mask in range(8):
            choice = native(history, mask, 1, output)
            if not mask or window[-1] == 0:
                assert choice == -1
                continue
            np.testing.assert_allclose(output, logits, atol=2e-5)
            admissible = [i for i in range(3) if mask & (1 << i)]
            assert choice == max(admissible, key=lambda i: logits[i])
    print("Verified 729 windows, all admissibility masks, and ownership deferral.")


if __name__ == "__main__":
    main()
