"""Serialize the native model's bounded choice at the telemetry boundary."""
import ctypes as ct

NAMES = ("track", "search", "wait")


class ChoiceScorer:
    def __init__(self, path):
        self.library = ct.CDLL(str(path.resolve()))
        self.library.nso_model_id.argtypes = []
        self.library.nso_model_id.restype = ct.c_char_p
        self.model = self.library.nso_model_id().decode("ascii")
        self.select = self.library.nso_select
        self.select.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint, ct.c_int, ct.POINTER(ct.c_float)]
        self.select.restype = ct.c_int

    def decide(self, history, mask, blocked=None):
        if type(mask) is not int or not 0 <= mask <= 7:
            raise ValueError("Invalid behavior mask")
        record = dict(model=self.model, allowed=[n for i,n in enumerate(NAMES) if mask & (1 << i)])
        reason = blocked
        if reason is None:
            if not mask:
                reason = "empty_mask"
            elif len(history) < 6:
                reason = "warmup"
            elif not history[-1]:
                reason = "missing_observation"
        if reason:
            return dict(record, status="defer", reason=reason)
        if len(history) != 6 or any(x not in (0, 1, 2) for x in history):
            raise ValueError("Expected six ternary observations")
        logits = (ct.c_float * 3)()
        choice = self.select((ct.c_uint8 * 6)(*history), mask, 1, logits)
        if choice == -1:
            return dict(record, status="defer", reason="native_defer")
        return dict(record, status="choice", selected=NAMES[choice], logits=dict(zip(NAMES, logits)))
