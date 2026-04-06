"""
Initialize the Redis task queue for distributed rendering.

Idempotent: can be re-run safely to recover from failures.
- Skips already-completed chunks
- Re-queues failed chunks (configurable via REQUEUE_FAILED)
- Clears stale processing entries

Redis key layout:
  {prefix}:queue         (List)  chunks waiting to be processed
  {prefix}:processing    (Hash)  chunk_id -> worker info (in-flight)
  {prefix}:completed     (Set)   chunk_ids finished successfully
  {prefix}:failed        (Set)   chunk_ids that failed
  {prefix}:progress:completed  (String)  counter
  {prefix}:progress:failed     (String)  counter
  {prefix}:meta:*        (String) metadata
"""

import redis
import json
import os
import sys


def main():
    host = os.environ.get("REDIS_HOST", "lumitex-redis")
    port = int(os.environ.get("REDIS_PORT", "6666"))
    prefix = os.environ.get("REDIS_KEY_PREFIX", "lumitex")
    total = int(os.environ["TOTAL_UIDS"])
    chunk_size = int(os.environ["CHUNK_SIZE"])
    requeue_failed = os.environ.get("REQUEUE_FAILED", "true").lower() == "true"

    r = redis.Redis(host=host, port=port, decode_responses=True)
    r.ping()
    print(f"Connected to Redis at {host}:{port}")

    queue_key = f"{prefix}:queue"
    processing_key = f"{prefix}:processing"
    completed_key = f"{prefix}:completed"
    failed_key = f"{prefix}:failed"

    # Already completed chunks
    completed = r.smembers(completed_key)
    completed_ids = set(completed)

    # Clear stale processing entries (from crashed workers)
    stale = r.hgetall(processing_key)
    if stale:
        print(f"Clearing {len(stale)} stale processing entries: {list(stale.keys())}")
        r.delete(processing_key)

    # Optionally re-queue failed chunks
    failed = r.smembers(failed_key)
    if requeue_failed and failed:
        print(f"Re-queuing {len(failed)} failed chunks: {sorted(failed, key=int)}")
        r.delete(failed_key)
        # Reset failed counter
        r.set(f"{prefix}:progress:failed", 0)
    else:
        # If not re-queuing, treat failed as "done" (skip them)
        completed_ids = completed_ids | failed

    # Rebuild the queue
    r.delete(queue_key)

    total_chunks = (total + chunk_size - 1) // chunk_size
    chunks = []
    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, total)
        if str(chunk_id) not in completed_ids:
            chunks.append(json.dumps({
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
            }))

    if chunks:
        r.rpush(queue_key, *chunks)

    # Store metadata
    r.set(f"{prefix}:meta:total_uids", total)
    r.set(f"{prefix}:meta:chunk_size", chunk_size)
    r.set(f"{prefix}:meta:total_chunks", total_chunks)

    print(f"Queue initialized: {len(chunks)} pending, {len(completed_ids)} already done, {total_chunks} total chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
