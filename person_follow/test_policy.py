import json
import io
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np
from .capture import raw_frame
from .model import Network
from .policy import VERSION, replay
from .train import synthetic


class PersonFollowTests(unittest.TestCase):
    def test_mask_override_loss_and_recovery(self):
        base = dict(version=VERSION,target=[.2,.1,.4,.7],quality=.9,age_s=0)
        rows = [dict(base,t=i*.2) for i in range(16)]
        rows[6]["override"] = True
        rows[7]["target"] = None
        rows[14]["age_s"] = 1
        decisions = [d for d,_ in replay(rows,lambda x:np.array([[100.,2.,1.]]))]
        self.assertEqual(decisions[5]["selected"],"hold_distance")
        self.assertEqual(decisions[6]["status"],"defer")
        self.assertEqual(decisions[7]["selected"],"stop")
        self.assertEqual(decisions[8]["reason"],"warming_up")
        self.assertEqual(decisions[14]["selected"],"stop")
        with self.assertRaises(ValueError):
            list(replay([rows[0],rows[0]]))

    def test_export_and_gradients(self):
        x = np.random.default_rng(5).normal(size=(2,6,6))
        for kind in ("mlp","transformer"):
            net = Network(kind)
            result,cache = net.forward(x)
            grad = net.gradients(cache,np.ones_like(result))
            for k,p in net.p.items():
                idx = np.unravel_index(np.argmax(abs(grad[k])),p.shape)
                p[idx] += 1e-6
                plus = net(x).sum()
                p[idx] -= 2e-6
                minus = net(x).sum()
                p[idx] += 1e-6
                self.assertAlmostEqual((plus-minus)/2e-6,grad[k][idx],places=5)
            path = Path(__file__).parent/"checkpoints"/(kind+".json")
            self.assertEqual(len(list(replay(synthetic(33),Network.load(path)))),50)

    def test_camera_socket(self):
        header = dict(width=2, height=1, format="UYVY", bytes=4, rotate=0,
                      captured_at_unix_us=1000000)
        wire = json.dumps(dict(result=header)).encode()+b"\n"+bytes([128,16,128,235])
        with patch("person_follow.capture.socket.socket") as factory:
            client = factory.return_value.__enter__.return_value
            client.makefile.return_value = io.BytesIO(wire)
            image,t = raw_frame("/test/media.sock")
            request = json.loads(client.sendall.call_args.args[0])
            self.assertEqual(request["method"], "media.frame")
        self.assertEqual(image.tolist(), [[16,235]])
        self.assertEqual(t, 1.)
