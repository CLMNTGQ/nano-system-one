"""Compare native masked selection and serialized scores to the complete table."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "edge"))
from choice import ChoiceScorer, NAMES


def main(library):
    scorer = ChoiceScorer(library)
    source = (ROOT / "edge/scores.inc").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    assert scorer.model == "choice-v1/" + blob
    rows = [[float(x.strip().removesuffix("f")) for x in line.strip("{},").split(",")]
            for line in source.decode().splitlines()]
    for index, history in enumerate(itertools.product(range(3), repeat=6)):
        for mask in range(8):
            result = json.loads(json.dumps(scorer.decide(history, mask), allow_nan=False))
            allowed = [n for i,n in enumerate(NAMES) if mask & (1 << i)]
            assert result["allowed"] == allowed
            if not allowed or not history[-1]:
                assert result["status"] == "defer" and "reason" in result
                assert "logits" not in result and "selected" not in result
            else:
                assert result["status"] == "choice"
                assert result["selected"] == max(allowed, key=lambda n: rows[index][NAMES.index(n)])
                assert all(abs(result["logits"][n]-rows[index][i]) < 1e-5 for i,n in enumerate(NAMES))
    print("729 histories x 8 masks verified")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
