"""Exercise the real socket adapter against deterministic Unix-socket fixtures."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


async def main(library):
    stimulus = dict(symbol=2, owner='autod')
    async def serve(reader, writer, media):
        try:
            req = json.loads(await reader.readline())
            assert req["method"] == ("media.ball" if media else "robot.subscribe")
            writer.write(b'{"jsonrpc":"2.0","id":1,"result":null}\n')
            while True:
                state = {"autonomous": True, "soccer": {"autonomous": True, "phase": "approaching"},
                         "policy": "walk", "safety": {"fallen": False}, "move": {"source": stimulus["owner"]},
                         "loop": {"missed": 0}}
                msg = {"jsonrpc":"2.0","method":"media.ballSeen" if media else "robot.state",
                       "params":{"seen":stimulus["symbol"] == 2} if media else state}
                if media and stimulus["symbol"] == 0:
                    await asyncio.sleep(0.04)
                    continue
                writer.write((json.dumps(msg)+"\n").encode())
                await writer.drain()
                await asyncio.sleep(0.04)
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        detector = path / "detector.json"
        detector.write_text("{}")
        async def robot_client(r, w):
            await serve(r, w, False)
        async def media_client(r, w):
            await serve(r, w, True)
        robot = await asyncio.start_unix_server(robot_client, str(path/"robot"))
        media = await asyncio.start_unix_server(media_client, str(path/"media"))
        cmd = [sys.executable, str(ROOT/"edge/robot_trace.py"), "--library", library]
        async with robot, media:
            proc = await asyncio.create_subprocess_exec(*cmd,"capture","--robot-socket",str(path/"robot"),
                "--media-socket",str(path/"media"),"--detector",str(detector),stdout=asyncio.subprocess.PIPE)
            try:
                records = []
                async def stage(symbol, owner, count):
                    stimulus.update(symbol=symbol, owner=owner)
                    group = [await asyncio.wait_for(proc.stdout.readline(), 5) for _ in range(count)]
                    records.extend(group)
                    return [json.loads(line) for line in group]
                stable = await stage(2, "autod", 9)
                assert stable[-1]["proposal"] == "track"
                misses = await stage(1, "autod", 5)
                assert all(row["symbol"] == 1 for row in misses[-3:])
                outage = await stage(0, "autod", 5)
                assert all(row["symbol"] == 0 and row["proposal"] == "defer" for row in outage[-3:])
                assert outage[-1]["decision"]["reason"] == "missing_observation"
                assert "logits" not in outage[-1]["decision"]
                assert outage[-1]["valid_age_ms"] > misses[-1]["valid_age_ms"]
                returned = await stage(2, "autod", 7)
                assert returned[-1]["proposal"] == "track"
                pad = await stage(2, "padd", 3)
                assert pad[-1]["decision"]["reason"] == "operator_override"
                assert pad[-1]["proposal"] == "defer" and not pad[-1]["eligible"]
                resumed = await stage(2, "autod", 8)
                eligible = [r for r in resumed if r["eligible"]]
                assert len(eligible) >= 6 and all(r["proposal"] == "defer" for r in eligible[:5])
                assert eligible[-1]["proposal"] == "track"
            finally:
                if proc.returncode is None:
                    proc.terminate()
                await proc.wait()
        rows = [json.loads(x) for x in records]
        assert all(row["proposal"] == "defer" for row in rows[:5])
        assert any(row["proposal"] == "track" for row in rows[5:])
        trace = path/"trace.jsonl"
        trace.write_bytes(b"".join(records))
        process = await asyncio.create_subprocess_exec(*cmd,"replay",str(trace))
        assert await process.wait() == 0
        chosen = next(r for r in rows if r["decision"]["status"] == "choice")
        chosen["decision"]["logits"]["wait"] += 1
        trace.write_text("".join(json.dumps(r)+"\n" for r in rows))
        process = await asyncio.create_subprocess_exec(*cmd,"replay",str(trace))
        assert await process.wait() == 1


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
