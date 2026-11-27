"""One-block temporal Transformer. Only the final query is needed for Choice."""
import json
import numpy as np

from .policy import ACTIONS, FEATURES, VERSION, WINDOW


class Network:
    def __init__(self, kind, seed=7):
        rng = np.random.default_rng(seed)
        d = 12
        n = len(FEATURES)
        shapes = {"embed": (n, d), "position": (WINDOW, d), "q": (d, d), "k": (d, d),
                  "v": (d, d), "ff": (d, 2*d), "project": (2*d, d), "out": (d, 3)}
        if kind == "mlp":
            shapes = {"ff": (WINDOW*n, 2*d), "out": (2*d, 3)}
        elif kind != "transformer":
            raise ValueError(kind)
        self.kind = kind
        self.p = {k: rng.normal(0, 1/np.sqrt(a), (a,b)) for k,(a,b) in shapes.items()}
        self.p["bias"] = np.zeros(3)

    def forward(self, x):
        p = self.p
        if self.kind == "mlp":
            flat = x.reshape(len(x), -1)
            hidden = np.tanh(flat @ p["ff"])
            return hidden @ p["out"] + p["bias"], (flat, hidden)
        e = x @ p["embed"] + p["position"]
        q, k, v = e[:, -1] @ p["q"], e @ p["k"], e @ p["v"]
        a = (k*q[:, None]).sum(-1) / np.sqrt(q.shape[-1])
        a = np.exp(a-a.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        u = e[:, -1] + (a[:,:,None]*v).sum(1)
        h = np.tanh(u @ p["ff"])
        z = u + h @ p["project"]
        return z @ p["out"] + p["bias"], (x,e,q,k,v,a,u,h,z)

    def __call__(self, x):
        return self.forward(x)[0]

    def gradients(self, cache, dy):
        p = self.p
        g = {"bias": dy.sum(0)}
        if self.kind == "mlp":
            x,h = cache
            g["out"] = h.T @ dy
            g["ff"] = x.T @ ((dy @ p["out"].T)*(1-h*h))
            return g
        x,e,q,k,v,a,u,h,z = cache
        g["out"] = z.T @ dy
        dz = dy @ p["out"].T
        g["project"] = h.T @ dz
        dh = (dz @ p["project"].T)*(1-h*h)
        g["ff"] = u.T @ dh
        du = dz + dh @ p["ff"].T
        dv = a[:,:,None]*du[:,None]
        da = (v*du[:,None]).sum(-1)
        ds = a*(da-(da*a).sum(1, keepdims=True))/np.sqrt(q.shape[-1])
        dq = (ds[:,:,None]*k).sum(1)
        dk = ds[:,:,None]*q[:,None]
        g["q"] = e[:,-1].T @ dq
        g["k"] = np.einsum("btd,bte->de", e, dk)
        g["v"] = np.einsum("btd,bte->de", e, dv)
        de = dk @ p["k"].T + dv @ p["v"].T
        de[:,-1] += du + dq @ p["q"].T
        g["position"] = de.sum(0)
        g["embed"] = np.einsum("btf,btd->fd", x, de)
        return g

    def save(self, path):
        path.write_text(json.dumps(dict(version=VERSION, actions=ACTIONS, features=FEATURES,
                                        kind=self.kind, weights={k:v.tolist() for k,v in self.p.items()}))+"\n")

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text())
        if (data["version"], data["actions"], data["features"]) != (VERSION,list(ACTIONS),list(FEATURES)):
            raise ValueError("Incompatible checkpoint")
        net = cls(data["kind"])
        for k, v in net.p.items():
            v[:] = np.asarray(data["weights"][k]).reshape(v.shape)
        return net
