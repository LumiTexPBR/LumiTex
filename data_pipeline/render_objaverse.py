"""
Render objaverse dataset using blender.

Example usage:

python render_objaverse.py --download_dir data/objaverse \
                        --envmap_dir envmaps \
                        --data_uids data/LumiTex_pbr_uids.json \
                        --rendered_uids_txt rendered_uids.txt \
                        --output_dir test_render \
                        --start_index 30000 \
                        --end_index 32501 \
                        --processes 8 \
                        --gpu_id 0 \
                        --s3_config s3/s3_config.yaml \
                        --log_to_wandb
"""

import os
import sys
import json
import argparse
import numpy as np
import objaverse
import multiprocessing
from utils import * 
import time
import threading
import subprocess
import shlex
import shutil
from s3.s3_utils import init_s3, download_from_s3, upload_to_s3, clean_local_files
from parse_exr_depth import process_depth_exr_folder
from multiprocessing.sharedctypes import Synchronized
from typing import Optional

import wandb

file_lock = threading.Lock()

# remove log files
if os.path.exists("logs"):
    shutil.rmtree("logs")
os.makedirs("logs", exist_ok=True)
logger = get_logger("logger", log_file="logs/render_objaverse.log")
checker_logger = get_logger("checker", log_file="logs/checker.log")
s3_logger = get_logger("s3", log_file="logs/s3.log")

cmd = (
    "blender -b -P blender_script.py -- "
        "--object_path {object_path} "
        "--envmap_dir {envmap_dir} "
        "--output_dir {output_dir} "
        "--resolution {resolution} "
        "--unit_sphere {unit_sphere} "
        "--camera_type {camera_type} "
        "--num_lightenvs {num_lightenvs} "
        "--num_images {num_images} "
        "--start_azimuth {start_azimuth} "
        "--start_elevation {start_elevation} "
        "--gpu_id {gpu_id} "
)


def run_bash(bash_args, uid):
    process = subprocess.Popen(
        bash_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )

    while True:
        if process.poll() is not None:
            break

        output = process.stdout.readline().decode("utf-8")
        print(output, end="")
    
    with file_lock:
        with open(args.rendered_uids_txt, "a") as f:
            f.write(uid + "\n")
    time.sleep(0.03)


def log_progress(
    valid_counter: Optional[Synchronized],
    invalid_counter: Optional[Synchronized],
    total_items: int,
    stop_event: threading.Event,
    interval: int = 5,
):
    """Periodically logs progress to wandb."""
    while not stop_event.is_set():
        if valid_counter is not None and invalid_counter is not None:
            with valid_counter.get_lock():
                current_valid = valid_counter.value
            with invalid_counter.get_lock():
                current_invalid = invalid_counter.value
            
            processed_count = current_valid + current_invalid
            progress = (processed_count / total_items) * 100 if total_items > 0 else 0
            
            wandb.log(
                {
                    "valid_count": current_valid,
                    "invalid_count": current_invalid,
                    "processed_count": processed_count,
                    "progress_percent": progress,
                },
                # Use step to track progress over time if desired
                # step=processed_count 
            )
        stop_event.wait(interval)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--data_uids", type=str, default="data/train_uids.json")
    parser.add_argument("--rendered_uids_txt", type=str, default="./rendered_uids.txt")
    parser.add_argument("--download_dir", type=str, default="data/objaverse")
    parser.add_argument("--output_dir", type=str, default="test_render")
    parser.add_argument("--envmap_dir", type=str, required=True)
    parser.add_argument("--render_material", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--online_download", action="store_true")
    parser.add_argument("--s3_config", type=str, default=None, help="s3 config file path")
    parser.add_argument("--log_to_wandb", action="store_true", help="log to wandb")
    return parser.parse_args()


def render_once(
    cmd_str: str,
    uid: str,
    logger: logging.Logger,
    rendered_uids_txt: str,
    silent: bool = True,
) -> bool:
    def run_cmd(cmd_str: str, silent: bool = False):
        """
        Run a shell command using subprocess
        :param cmd_str: shell command
        :return: True if success, False otherwise
        """
        cmd_elements = shlex.split(cmd_str)
        try:
            if silent:
                subprocess.run(cmd_elements, check=True, text=True, stdout=subprocess.DEVNULL)
            else:
                subprocess.run(cmd_elements, check=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error("Error in running cmd %s" % cmd_str)
            logger.error("Error code %s" % e.returncode)
            logger.error("Error msg %s" % e.stderr)
            return False
        return True
    
    start_time = time.time()

    if not run_cmd(cmd_str, silent):
        logger.error(f"Command in {uid} raise an error....")
        logger.error(f"Command: {cmd_str}")
        return False

    end_time = time.time()
    logger.info(f"Render {uid} done in {end_time - start_time} seconds")
    
    result_dir = os.path.join(args.output_dir, uid)
    # process_depth_exr_folder(os.path.join(result_dir, "hdr", "Z"))
    if check_directory_images(result_dir, checker_logger):
        with open(rendered_uids_txt, 'a', encoding='UTF-8') as f:
            f.write('{}\n'.format(uid))
        return True
    else:
        return False


if __name__ == "__main__":
    args = parse_args()
    uids = json.load(open(args.data_uids))
    cpu_cnt = multiprocessing.cpu_count()
    num_proc = min(args.processes, cpu_cnt)
    logger.info(f"Config: {args}")
    os.makedirs(args.download_dir, exist_ok=True)
    start_time = time.time()
    
    rendered_uid = []
    if os.path.exists(args.rendered_uids_txt):
        with open(args.rendered_uids_txt, 'r') as f:
            rendered_uid = f.read().split('\n')[:-1]
    
    uids = set(uids) - set(rendered_uid)
    
    # setup uid_to_path
    if args.online_download:
        objaverse.BASE_PATH = args.download_dir
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")
        uid_to_path = objaverse.load_objects(uids, download_processes=num_proc)
    elif args.s3_config:
        s3_client, s3_config = init_s3(args.s3_config, s3_logger)
        cos_paths = json.load(open("./data/LumiTex_pbr.json"))[args.start_index:args.end_index] # split here
        uid_to_path = {uid: path for path in cos_paths if (uid := path.split("/")[-1][:-4]) in uids}
        os.makedirs(os.path.join(args.download_dir, "hf-objaverse-v1"), exist_ok=True)
    else: # use local paths
        uid_to_path_all = json.load(open("./data/train_local_paths.json"))
        uid_to_path = {uid: path for uid, path in uid_to_path_all.items() if uid in uids}
    
    
    pool = multiprocessing.Pool(num_proc)
    logger.info(f"Rendering {len(uid_to_path)} objects, {cpu_cnt} CPUs, using {num_proc} processes")
    # blender rendering mode
    envmap_dir = args.envmap_dir
    material_cmd = "--material_type PBR" if args.render_material else ""
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging_thread = None
    stop_logging_event = None
    if args.log_to_wandb:
        wandb.init(
            project="ObjaverseRenderer-LumiTex", 
            entity="xxxxxxxxxxxxxx",
            name=f"{args.start_index:05d}_{args.end_index:05d}",
        )
        valid_counter = multiprocessing.Value('i', 0)
        invalid_counter = multiprocessing.Value('i', 0)
        total_items = len(uids)
        
        def update_counters_callback(success: bool):
            """Callback function to update counters based on task result."""
            if success:
                with valid_counter.get_lock():
                    valid_counter.value += 1
            else:
                with invalid_counter.get_lock():
                    invalid_counter.value += 1

        def handle_error_callback(error):
            """Callback function to handle errors from tasks."""
            logger.error(f"Error in worker process: {error}")
            with invalid_counter.get_lock():
                invalid_counter.value += 1

        stop_logging_event = threading.Event()
        logging_thread = threading.Thread(
            target=log_progress,
            args=(valid_counter, invalid_counter, total_items, stop_logging_event, 60),
            daemon=True # Set as daemon so it exits if main thread exits unexpectedly
        )
        logging_thread.start()
    else:
        valid_counter = None
        invalid_counter = None
        def update_counters_callback(success: bool): pass
        def handle_error_callback(error): pass
    
    gpu_id_offset = 0
    for uid, path in uid_to_path.items():
        if s3_client:
            object_path = os.path.join(args.download_dir, "hf-objaverse-v1", path)
            # download from s3
            cos_path = os.path.join(s3_config["cos_paths"]["download_dir"], path)
            if not download_from_s3(s3_client, s3_config['bucket'], cos_path, object_path, s3_logger):
                logger.error(f"Failed to download {path} from S3")
                if args.log_to_wandb:
                    with invalid_counter.get_lock():
                        invalid_counter.value += 1
                continue
        else:
            object_path = path

        for env in [envmap_dir]: # default hdr rendering, "None" for photometric rendering
            bledner_bash_args = cmd.format(
                object_path=object_path,
                envmap_dir=env,
                output_dir=f"{args.output_dir}",
                resolution=1024,
                unit_sphere=True,
                camera_type="ORTHO",
                num_lightenvs=2,
                num_images=6,
                start_azimuth=0,
                start_elevation=0,
                gpu_id=args.gpu_id + gpu_id_offset,
            ) + material_cmd
            gpu_id_offset = (gpu_id_offset + 1) % 1
            # # single process
            # run_bash(bledner_bash_args, uid)
            pool.apply_async(
                func=render_once,
                args=(bledner_bash_args, uid, logger, args.rendered_uids_txt),
                callback=update_counters_callback,
                error_callback=handle_error_callback
            )
        
    pool.close()
    pool.join()
    
    # --- Stop the logging thread ---
    if args.log_to_wandb and logging_thread is not None:
        stop_logging_event.set() # Signal the thread to stop
        logging_thread.join() # Wait for the logging thread
        
        # Log one final time to ensure the absolute final count is captured
        final_valid = valid_counter.value
        final_invalid = invalid_counter.value
        final_processed = final_valid + final_invalid
        final_progress = (final_processed / total_items) * 100 if total_items > 0 else 0
        wandb.log({
            "final_valid_count": final_valid,
            "final_invalid_count": final_invalid,
            "final_processed_count": final_processed,
            "final_progress_percent": final_progress,
        })
        logger.info(f"Final counts - Valid: {final_valid}, Invalid: {final_invalid}")

    time_cost = time.time() - start_time
    logger.info(f"All rendering tasks DONE in {time_cost:.2f} seconds")
    
    if s3_client and time_cost > 60:
        # zip files and logs
        index_range = f"{args.start_index:05d}_{args.end_index:05d}"
        zip_file_name = f"LumiTex_{index_range}.zip"
        log_file_name = f"logs_{index_range}.zip"
        upload_local_dir = os.path.join(os.path.dirname(args.output_dir), ".cos_upload")
        zip_file_path = os.path.join(upload_local_dir, zip_file_name)
        log_file_path = os.path.join(upload_local_dir, log_file_name)
        os.makedirs(upload_local_dir, exist_ok=True)
        if not os.path.exists(zip_file_path):
            zip_directory(args.output_dir, zip_file_path)
        if not os.path.exists(log_file_path):
            zip_directory("logs", log_file_path)
        # upload to s3 and clean local files
        upload_dir = os.path.join(s3_config["cos_paths"]["upload_dir"], s3_config["taskid"])
        upload_to_s3(s3_client, s3_config['bucket'], zip_file_path, os.path.join(upload_dir, zip_file_name), s3_logger)
        upload_to_s3(s3_client, s3_config['bucket'], log_file_path, os.path.join(upload_dir, log_file_name), s3_logger)
        
        clean_local_files(args.output_dir, s3_logger)
        clean_local_files(os.path.join(args.download_dir, "hf-objaverse-v1"), s3_logger)
        # clean_local_files(upload_local_dir, s3_logger)
        logger.success(f"Uploaded {zip_file_name} and {log_file_name} to S3")

