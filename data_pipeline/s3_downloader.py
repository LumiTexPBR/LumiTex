"""Usage:

python file_upload.py \
    --src_path /root/autodl-tmp/.cos_upload/LumiTex_16000_17000.zip \
    --dest_path LumiTex_16000_17000.zip \
    --s3_config s3/s3_config.yaml
    
python s3_downloader.py --s3_config s3/s3_config.yaml

"""

from s3.s3_utils import init_s3, download_from_s3, upload_to_s3, clean_local_files
import argparse
from utils import *
import os
import subprocess

logger = get_logger("logger")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_config", type=str, required=True)
    args = parser.parse_args()
    
    s3_client, s3_config = init_s3(args.s3_config, logger)
    # * upload to s3
    # upload_dir = os.path.join(s3_config["cos_paths"]["upload_dir"], s3_config["taskid"])
    # upload_to_s3(s3_client, s3_config['bucket'], args.src_path, os.path.join(upload_dir, args.dest_path), logger)
    # * download from s3
    download_cos_dir = os.path.join(s3_config["cos_paths"]["upload_dir"], s3_config["taskid"])
    tmp_dir = "../.cos_download"
    unzipped_list = []
    new_unzipped_list = []
    with open(os.path.join(tmp_dir, "LumiTex_unzipped_list.txt"), "r") as f:
        for line in f:
            unzipped_list.append(line.strip())
    
    # unzip processes
    unzip_processes = []
    
    for index in range(0, 63):
        index_name = f"{index:02d}000_{index+1:02d}000"
        zip_name = f"LumiTex_{index_name}.zip"
        zip_path = os.path.join(tmp_dir, zip_name)
        log_name = f"logs_{index_name}.zip"
        logger.info(f"Downloading {zip_name}")
        if not os.path.exists(zip_path) and zip_name not in unzipped_list:
            try:
                if download_from_s3(
                    s3_client, 
                    s3_config['bucket'], 
                    os.path.join(download_cos_dir, zip_name), 
                    zip_path, 
                    logger, 
                    VERBOSE=False,
                    show_progress=True
                ):
                    logger.success(f"Downloaded {zip_name}")
                else:
                    continue
            except Exception as e:
                continue
        else:
            logger.success(f"File {zip_name} already exists")
            continue
            
        # * unzip by process
        unzip_data_dir = "../LumiTex"
        unzip_logs_dir = f"../LumiTex_logs/{index_name}"
        
        os.makedirs(unzip_logs_dir, exist_ok=True)
        cmd_data = f"7z x {zip_path} -o{unzip_data_dir} -aoa -mmt -bb1 > {unzip_logs_dir}/unzip.log 2>&1"
        logger.info(f"Starting background unzip: {cmd_data}")
        process = subprocess.Popen(cmd_data, shell=True)
        unzip_processes.append((process, zip_name))
    
    logger.info("All downloads completed. Waiting for unzip processes to finish...")
    for process, zip_name in unzip_processes:
        return_code = process.wait()
        if return_code == 0:
            logger.success(f"Successfully unzipped {zip_name}")
            new_unzipped_list.append(zip_name)
            os.remove(os.path.join(tmp_dir, zip_name))
        else:
            logger.error(f"Failed to unzip {zip_name}, return code: {return_code}")
    
    logger.info("All operations completed successfully")
    
    with open(os.path.join(tmp_dir, "LumiTex_unzipped_list.txt"), "a") as f:
        for zip_name in new_unzipped_list:
            f.write(zip_name + "\n")
