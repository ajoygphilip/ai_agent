import json
import time
import uuid
from contextlib import asynccontextmanager


def emit(record: dict) -> None:
    # print("emitting", json.dumps(record, default=str))
    pass


@asynccontextmanager
async def span(name: str, **attrs):
    rec = {"id": uuid.uuid4().hex[:4], "name": name, **attrs}
    start = time.perf_counter()

    try:
        yield rec
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = type(e).__name__
        raise
    finally:
        rec["duration_s"] = round(time.perf_counter() - start, 3)
        emit(rec)


__all__ = ["span"]
