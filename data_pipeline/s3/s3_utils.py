import os
import boto3
import yaml
import logging
from botocore.exceptions import ClientError
import shutil
import time
from .logger import get_logger
import argparse

s3_prefix = None
s3_download_dir = None
s3_upload_dir = None

def load_yaml_config(config_path):
    """
    Load S3 configuration from YAML file
    Args:
        config_path (str): Path to S3 config YAML file
    Returns:
        dict: Configuration dictionary with S3 settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    global s3_prefix, s3_download_dir, s3_upload_dir
    s3_prefix = config['cos_paths']['prefix']
    s3_download_dir = config['cos_paths']['download_dir']
    s3_upload_dir = config['cos_paths']['upload_dir']
    return config


def create_s3_client(s3_config, logger):
    """
    Create and return an S3 client using the provided config
    Args:
        s3_config (dict): S3 configuration with access_key, secret_key, etc.
    Returns:
        boto3.client: Configured S3 client or None if authentication fails
    """
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f"{s3_config['protocol']}://cos.{s3_config['region']}.myqcloud.com",
            region_name=s3_config['region']
        )
        logger.info(f"S3 client created with endpoint: {s3_client.meta.endpoint_url}")
        # authenticate with s3
        test_key = s3_config['cos_paths']['auth_file']
        test_local_path = 'auth.txt'
        s3_client.download_file(s3_config['bucket'], test_key, test_local_path)
        if os.path.exists(test_local_path):
            logger.success(f"S3 authentication successful")
            os.remove(test_local_path)  # Clean up after test
        return s3_client
    except ClientError as e:
        logger.error(f"S3 authentication failed: {e}")
        return None


def download_from_s3(s3_client, bucket, s3_key, local_path, logger, VERBOSE=False, show_progress=False):
    """
    Download an object from S3 to local storage
    Args:
        s3_client (boto3.client): S3 client
        bucket (str): S3 bucket name
        s3_key (str): S3 object key
        local_path (str): Local path to save the downloaded file
        logger (logging.Logger): Logger object
        VERBOSE (bool): Whether to log detailed messages
        show_progress (bool): Whether to display download progress
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        full_s3_key = os.path.join(s3_prefix, s3_key)
        
        if show_progress:
            # Get file size first to calculate progress
            response = s3_client.head_object(Bucket=bucket, Key=full_s3_key)
            total_size = response.get('ContentLength', 0)
            total_size_mb = total_size / (1024 * 1024)
            
            # Create a progress callback
            downloaded_bytes = 0
            start_time = None
            
            def progress_callback(bytes_transferred):
                nonlocal downloaded_bytes, start_time
                if start_time is None:
                    start_time = time.time()
                
                downloaded_bytes += bytes_transferred
                downloaded_mb = downloaded_bytes / (1024 * 1024)
                percentage = (downloaded_bytes / total_size) * 100 if total_size > 0 else 0
                
                # Calculate speed
                elapsed_time = time.time() - start_time
                speed = downloaded_mb / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\rDownloading: {percentage:.1f}% ({downloaded_mb:.2f} MB / {total_size_mb:.2f} MB) - {speed:.2f} MB/s", end="")
            
            s3_client.download_file(
                bucket, 
                full_s3_key, 
                local_path,
                Callback=progress_callback
            )
            print()  # Add a newline after progress is complete
        else:
            s3_client.download_file(bucket, full_s3_key, local_path)
            
        if VERBOSE:
            logger.success(f"Downloaded {full_s3_key} to {local_path}")
        return True
    except ClientError as e:
        if VERBOSE:
            logger.error(f"Error downloading {full_s3_key} from S3: {e}")
        return False


def upload_to_s3(s3_client, bucket, local_path, s3_key, logger):
    """
    Upload a file to S3
    Args:
        s3_client (boto3.client): S3 client
        bucket (str): S3 bucket name
        local_path (str): Local file path to upload
        s3_key (str): S3 object key for the uploaded file
        logger (logging.Logger): Logger object
        
    Returns:
        bool: True if upload succeeded, False otherwise
    """
    try:
        if not os.path.exists(local_path):
            logger.warning(f"Local file {local_path} does not exist")
            return False
        
        s3_client.upload_file(local_path, bucket, os.path.join(s3_prefix, s3_key))
        logger.success(f"Uploaded {local_path} to cos://{bucket}/{s3_prefix}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading {local_path} to S3: {e}")
        return False


def clean_local_files(local_dir, logger):
    """
    Remove local render files after successful S3 upload
    Args:
        local_dir (str): Base local directory containing renders
        uid (str): Object UID
        logger (logging.Logger): Logger object
        
    Returns:
        bool: True if cleanup succeeded, False otherwise
    """
    try:
        if not os.path.exists(local_dir):
            logger.warning(f"Directory not found: {local_dir}")
            return False

        shutil.rmtree(local_dir)
        return True
    except Exception as e:
        logger.error(f"Error cleaning up local files: {e}")
        return False


def init_s3(config_path, logger):
    """
    Initialize S3 configuration and client
    Args:
        args (argparse.Namespace): Command line arguments
        logger (logging.Logger): Logger object
        
    Returns:
        tuple: (s3_client, s3_config) or (None, None) if S3 isn't enabled
    """
    logger.info(f"Initializing S3 with config: {config_path}")
    try:
        s3_config = load_yaml_config(config_path)
        s3_client = create_s3_client(s3_config, logger)
        logger.info(f"S3 configured with bucket={s3_config['bucket']}")
        return s3_client, s3_config
    except Exception as e:
        logger.error(f"Failed to initialize S3: {e}")
        return None, None


def parse_args():
    parser = argparse.ArgumentParser(description='S3 configuration')
    parser.add_argument('--s3_config', type=str, default="s3_config.yaml", help='Path to S3 config YAML file')
    return parser.parse_args()

if __name__ == "__main__":
    if os.path.exists("s3.log"):
        os.remove("s3.log")
    logger = get_logger("s3_test", log_file="s3.log")
    
    config_path = "s3_config.yaml"
    s3_client, s3_config = init_s3(config_path, logger)
    if s3_client is None:
        logger.error("S3 client creation failed")
        exit(1)
    logger.info(f"s3_config: {s3_config}")
    s3_prefix = s3_config['cos_paths']['prefix']
    s3_upload_dir = s3_config['cos_paths']['upload_dir']
    cos_test_download_file = os.path.join(s3_upload_dir, "test_file_download.txt")
    cos_test_upload_file = os.path.join(s3_upload_dir, "test_file_upload.txt")
    
    # test download_from_s3
    os.makedirs("test", exist_ok=True)
    download_from_s3(s3_client, s3_config['bucket'], cos_test_download_file, "test/test_file.txt", logger)
    # test upload_to_s3
    upload_to_s3(s3_client, s3_config['bucket'], "test/test_file.txt", cos_test_upload_file, logger)
    clean_local_files("test", logger)