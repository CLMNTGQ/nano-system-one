"""Exercise the real socket adapter against deterministic Unix-socket fixtures."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


async def main(library):
    async def serve(reader, writer, media):
        try:
            req = json.loads(await reader.readline())
            assert req["method"] == ("media.ball" if media else "robot.subscribe")
            writer.write(b'{"jsonrpc":"2.0","id":1,"result":null}\n')
            while True:
                state = {"autonomous": True, "soccer": {"autonomous": True, "phase": "approaching"},
                         "policy": "walk", "safety": {"fallen": False}, "move": {"source": "autod"},
                         "loop": {"missed": 0}}
                msg = {"jsonrpc":"2.0","method":"media.ballSeen" if media else "robot.state",
                       "params":{"seen":True} if media else state}
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
                records = [await asyncio.wait_for(proc.stdout.readline(),5) for _ in range(9)]
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


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
