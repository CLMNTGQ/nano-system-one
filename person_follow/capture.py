"""Capture normalized single-person observations from video or MicroDuck mediad."""
import argparse
import json
import math
import socket
import time

import numpy as np
from .policy import VERSION


def raw_frame(path):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(3)
        client.connect(path)
        client.sendall(b'{"jsonrpc":"2.0","id":1,"method":"media.frame","params":{}}\n')
        with client.makefile("rb") as stream:
            reply = json.loads(stream.readline(65536))
            if "error" in reply:
                raise RuntimeError(reply["error"])
            h = reply["result"]
            w, n, size = h["width"], h["height"], h["bytes"]
            if not (w > 0 and n > 0 and w % 2 == 0 and size == 2*w*n <= 16*1024**2
                    and h["format"] == "UYVY" and h["rotate"] in (0, 90, 180, 270)):
                raise ValueError("Invalid mediad frame geometry")
            data = stream.read(size)
            if len(data) != size:
                raise EOFError("Truncated mediad frame")
    # HOG only needs luminance. Undo the clockwise camera mount.
    grey = np.frombuffer(data, dtype=np.uint8)[1::2].reshape(n, w)
    return np.rot90(grey, h["rotate"] // 90).copy(), h["captured_at_unix_us"]/1e6


def frames(args, cv):
    if args.socket:
        previous = None
        while True:
            frame, t = raw_frame(args.socket)
            if previous is None or t > previous:
                previous = t
                yield frame, t
            time.sleep(1/args.hz)
    else:
        cap = cv.VideoCapture(args.video)
        try:
            fps = cap.get(cv.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Expected a readable constant-frame-rate video")
            index, next_t = 0, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = index/fps
                index += 1
                if t >= next_t:
                    yield frame, t
                    next_t += 1/args.hz
        finally:
            cap.release()


def main():
    import cv2 as cv
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--socket")
    parser.add_argument("--hz", type=float, default=5)
    parser.add_argument("--show", action="store_true", help="Preview selected box on a desktop")
    args = parser.parse_args()
    if not 0 < args.hz <= 30:
        parser.error("Expected 0 < hz <= 30")
    detector = cv.HOGDescriptor()
    detector.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    previous_box = None
    previous_seen = None
    try:
        for frame, t in frames(args, cv):
            height, width = frame.shape[:2]
            boxes, weights = (detector.detectMultiScale(frame, winStride=(8, 8))
                              if width >= 64 and height >= 128 else ([], []))
            target, quality = None, 0.
            if len(boxes) == 1:
                x, y, w, h = map(int, boxes[0])
                target = [x/width, y/height, w/width, h/height]
                margin = float(np.asarray(weights).ravel()[0])
                quality = 1/(1+math.exp(-max(-50., min(50., margin))))
            # One-person trial. Interrupt the track on an implausible center jump.
            if target is not None and previous_box is not None and t-previous_seen <= .5:
                cx = target[0]+target[2]/2
                last_cx = previous_box[0]+previous_box[2]/2
                if abs(cx-last_cx) > .25:
                    target, quality = None, 0.
            if target is not None:
                previous_box, previous_seen = target, t
            if args.show:
                preview = frame.copy()
                if target is not None:
                    cv.rectangle(preview, (x,y), (x+w,y+h), (255,255,255), 2)
                cv.imshow("Person follow / shadow", preview)
                if cv.waitKey(1) == 27:
                    break
            age = max(0., time.time()-t) if args.socket else 0.
            print(json.dumps(dict(version=VERSION, t=t, age_s=age, target=target,
                                  quality=quality, detections=len(boxes))), flush=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
