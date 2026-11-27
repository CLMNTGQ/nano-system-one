import io
import json
import os
from pathlib import Path
import threading
import time
import unittest

from .active import Controller, ZERO, control_loop
from .model import Network
from .policy import VERSION


def state(t, source=None):
    return dict(t_ns=round(t*1e9), policy="walk", safety=dict(fallen=False, limp=False),
                head=[0]*4, move=dict(source=source, limited_by=[] if source else ["deadman"]))


def depth(t):
    return dict(t_ns=round(t*1e9), rows=8, cols=8, distance_mm=[900]*64, status=[5]*64)


def observation(t, height=.30, error=.18):
    return dict(version=VERSION, t=t, age_s=.01, quality=.98,
                target=[(1+error)/2-.05,.1,.1,height])


def network():
    return Network.load(Path(__file__).parent/"checkpoints"/"transformer.json")


class ActiveTests(unittest.TestCase):
    def test_feedback_changes_next_measurement(self):
        controller = Controller(network(), "test")
        height, error = .27, .35
        moved = turned = False
        for i in range(130):
            t = i*.2
            controller.robot(state(t, "test"))
            controller.depth(depth(t))
            controller.observe(observation(t, height, error), t+.01, t)
            (vx, _, yaw), _ = controller.command(t)
            moved |= vx > 0
            turned |= yaw < 0
            self.assertLessEqual(vx,.10)
            self.assertLessEqual(abs(yaw),.35)
            # Idealized plant: approaching enlarges the box; right yaw centers it.
            height += vx*.04
            error += yaw*.18
        self.assertTrue(moved and turned)
        self.assertGreater(height,.42)
        self.assertLess(abs(error),.08)
        self.assertEqual(controller.command(t)[0],ZERO)
        controller.observe(dict(observation(t+.2),target=None), t+.2,t+.2)
        self.assertEqual(controller.command(t+.2)[0],ZERO)

    def test_inhibits_and_handover(self):
        c = Controller(None,"test")
        for i in range(6):
            t = i*.2
            c.observe(observation(t),t,t)
        c.robot(state(t,"test"))
        c.depth(depth(t))
        self.assertGreater(c.command(t)[0][0],0)
        self.assertIsNone(c.command(t+.26)[0])
        c.robot(state(t+.36,"test"))
        self.assertEqual(c.command(t+.36)[1],"depth_blocked_or_stale")
        blocked = depth(t+.36)
        blocked["distance_mm"][27] = 260
        c.depth(blocked)
        self.assertEqual(c.command(t+.36)[0],ZERO)
        c.robot(state(t+.51,"test"))
        c.depth(depth(t+.51))
        self.assertEqual(c.command(t+.51)[1],"expired_observation")
        c.robot(state(t+.52,"padd"))
        c.robot(state(t+.53,"test"))
        self.assertIsNone(c.command(t+.53)[0])

    def test_loss_while_walking(self):
        c = Controller(network(), "test")
        for i in range(6):
            t = i*.2
            c.observe(observation(t),t,t)
        c.robot(state(t,"test"))
        c.depth(depth(t))
        self.assertGreater(c.command(t)[0][0],0)
        t += .1
        c.observe(dict(observation(t), target=None),t,t)
        self.assertEqual(c.command(t),(ZERO, "unreliable_target"))

    def test_wire_commands_and_camera_silence(self):
        # Real selectable pipes carry the exact notification framing; no hardware moves.
        pipes = [os.pipe() for _ in range(3)]
        sent = []
        errors = []
        class Robot:
            def fileno(self):
                return pipes[0][0]
            def sendall(self, data):
                sent.append(json.loads(data))
        c = Controller(network(),"test")
        def run():
            try:
                control_loop(Robot(), pipes[1][0], pipes[2][0], c, io.StringIO())
            except ConnectionError:
                pass
            except Exception as exc:
                errors.append(exc)
        def write(i, msg):
            os.write(pipes[i][1], (json.dumps(msg)+"\n").encode())
        thread = threading.Thread(target=run)
        thread.start()
        try:
            for step in range(22):
                now = time.monotonic()
                write(0, dict(method="robot.state",params=state(now,"test" if sent else None)))
                write(1, dict(method="tof.frame",params=depth(now)))
                if step < 7:
                    write(2, observation(time.time()))
                time.sleep(.05)
        finally:
            os.close(pipes[2][1])
            thread.join(2)
            for fd in [pipes[2][0], *pipes[0], *pipes[1]]:
                os.close(fd)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors,[])
        self.assertTrue(any(m["params"]["vx"] > 0 for m in sent))
        self.assertEqual(sent[-1], dict(jsonrpc="2.0",method="robot.move",
                                       params=dict(vx=0.,vy=0.,vyaw=0.)))
