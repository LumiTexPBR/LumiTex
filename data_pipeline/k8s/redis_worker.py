"""
Redis-based rendering worker.

Pulls chunks from a Redis queue, dispatches render_objaverse.py for each,
and tracks completion/failure in Redis. Exits when the queue is empty.

Usage: called by entrypoint-redis.sh inside a K8s pod.
"""

import redis
import json
import subprocess
import os
import sys
import time


def build_render_command(chunk, worker_id):
    """Build the render_objaverse.py command for a given chunk."""
    cmd = [
        "python3", "render_objaverse.py",
        "--download_dir",       os.environ.get("DOWNLOAD_DIR", "/data/objaverse"),
        "--envmap_dir",         os.environ.get("ENVMAP_DIR", "/data/envmaps"),
        "--data_uids",          os.environ.get("DATA_UIDS_PATH", "/data/uids/data_uids.json"),
        "--rendered_uids_txt",  f"/data/rendered_uids/rendered_uids_{worker_id}_{chunk['chunk_id']}.txt",
        "--output_dir",         os.environ.get("OUTPUT_DIR", "/data/output"),
        "--start_index",        str(chunk["start"]),
        "--end_index",          str(chunk["end"]),
        "--processes",          os.environ.get("NUM_PROCESSES", "4"),
        "--gpu_id",             "0",
        "--s3_config",          os.environ.get("S3_CONFIG_PATH", "/etc/s3/s3_config.yaml"),
    ]
    if os.environ.get("RENDER_MATERIAL", "true").lower() == "true":
        cmd.append("--render_material")
    if os.environ.get("LOG_TO_WANDB", "false").lower() == "true":
        cmd.append("--log_to_wandb")
    return cmd


def main():
    host = os.environ.get("REDIS_HOST", "lumitex-redis")
    port = int(os.environ.get("REDIS_PORT", "6666"))
    prefix = os.environ.get("REDIS_KEY_PREFIX", "lumitex")
    worker_id = os.environ.get("HOSTNAME", f"worker-{os.getpid()}")
    max_empty_retries = int(os.environ.get("MAX_EMPTY_RETRIES", "3"))

    r = redis.Redis(host=host, port=port, decode_responses=True)
    r.ping()

    queue_key = f"{prefix}:queue"
    processing_key = f"{prefix}:processing"
    completed_key = f"{prefix}:completed"
    failed_key = f"{prefix}:failed"

    completed_count = 0
    failed_count = 0
    empty_retries = 0

    print(f"[{worker_id}] Worker started. Redis={host}:{port}, prefix={prefix}")

    while True:
        raw = r.lpop(queue_key)

        if raw is None:
            empty_retries += 1
            remaining = r.hlen(processing_key)
            print(f"[{worker_id}] Queue empty (attempt {empty_retries}/{max_empty_retries}), "
                  f"{remaining} chunks still processing by other workers.")
            if empty_retries >= max_empty_retries:
                break
            time.sleep(10)
            continue

        # Got a chunk, reset empty counter
        empty_retries = 0
        chunk = json.loads(raw)
        chunk_id = str(chunk["chunk_id"])

        # Skip if already completed (idempotency on re-queue)
        if r.sismember(completed_key, chunk_id):
            print(f"[{worker_id}] Chunk {chunk_id} already completed, skipping.")
            continue

        # Mark as processing
        r.hset(processing_key, chunk_id, json.dumps({
            "worker": worker_id,
            "started_at": time.time(),
        }))

        start, end = chunk["start"], chunk["end"]
        print(f"[{worker_id}] === Chunk {chunk_id}: UIDs [{start}, {end}) ===")

        cmd = build_render_command(chunk, worker_id)
        t0 = time.time()

        try:
            subprocess.run(cmd, check=True)
            r.sadd(completed_key, chunk_id)
            r.incr(f"{prefix}:progress:completed")
            completed_count += 1
            elapsed = time.time() - t0
            print(f"[{worker_id}] Chunk {chunk_id} done in {elapsed:.0f}s.")
        except subprocess.CalledProcessError as e:
            r.sadd(failed_key, chunk_id)
            r.incr(f"{prefix}:progress:failed")
            failed_count += 1
            print(f"[{worker_id}] Chunk {chunk_id} FAILED (exit code {e.returncode}).")
        finally:
            r.hdel(processing_key, chunk_id)

    queue_remaining = r.llen(queue_key)
    total_completed = r.scard(completed_key)
    total_failed = r.scard(failed_key)
    print(f"[{worker_id}] Exiting. "
          f"This worker: completed={completed_count}, failed={failed_count}. "
          f"Global: completed={total_completed}, failed={total_failed}, queue={queue_remaining}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
