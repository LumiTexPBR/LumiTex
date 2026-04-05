import subprocess
import logging
import zipfile
import os
from PIL import Image


def check_image_validity(image_path, logger):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {image_path}. Error: {e}")
        return False


def check_directory_images(directory_path, logger):
    valid_count = 0
    invalid_count = 0
    
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return False
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            image_path = os.path.join(root, file)
            if file.lower().endswith('.png'):
                if check_image_validity(image_path, logger):
                    valid_count += 1
                else:
                    invalid_count += 1
                
            if '_0001' in file: # blender file name
                os.rename(image_path, os.path.join(root, file.replace('_0001', '')))
    
    total_count = valid_count + invalid_count
    uid = os.path.basename(directory_path)
    valid = valid_count == total_count and valid_count == 288
    if valid:
        logger.info(f"Checked {valid_count}/{total_count} images in {uid}, complete")
    else:
        logger.warning(f"Checked {valid_count}/{total_count} images in {uid}, incomplete")
    return valid


def zip_directory(dir_path, zip_path):
    # all_files = []
    # for root, dirs, files in os.walk(dir_path):
    #     for file in files:
    #         all_files.append(os.path.join(root, file))
    # with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #     # for file in tqdm(all_files, desc="[Zipping files]"):
    #     for file in all_files:
    #         rel_file = os.path.relpath(file, dir_path)
    #         zipf.write(file, rel_file)
    
    # * RECOMMENDED: use 7z to compress the directory
    try:
        dir_path = os.path.abspath(dir_path)
        zip_path = os.path.abspath(zip_path)
        
        original_cwd = os.getcwd()
        os.chdir(dir_path)
        
        # 使用7z压缩，参数说明：
        # a: 添加文件到压缩包
        # -tzip: 指定压缩格式为zip
        # -mx=5: 压缩级别5 (0-9，平衡压缩率和速度)
        # -mmt=on: 启用多线程压缩
        cmd = ['7z', 'a', '-tzip', zip_path, '*', '-mx=5', '-mmt=on']
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        os.chdir(original_cwd)
        
        print(f"Successfully compressed {dir_path} to {zip_path}")
        
    except subprocess.CalledProcessError as e:
        os.chdir(original_cwd)  # 确保恢复工作目录
        raise Exception(f"7z compression failed: {e.stderr}")
    except Exception as e:
        os.chdir(original_cwd)  # 确保恢复工作目录
        raise Exception(f"Compression error: {str(e)}")


# Add SUCCESS level between INFO and WARNING
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)
logging.Logger.success = success

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;1m" 
    gray = "\x1b[38;5;240m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.SUCCESS: green + format + reset,
        logging.DEBUG: gray + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        if isinstance(record.msg, float):
            record.msg = f"{record.msg:.3f}"
        elif isinstance(record.msg, str):
            import re
            record.msg = re.sub(r'\d+\.\d+', lambda x: f"{float(x.group()):.3f}", record.msg)
            
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    
    if log_file:
        import os
        file_exists = os.path.exists(log_file)
        
        fh = logging.FileHandler(log_file, mode='a' if file_exists else 'w')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
    return logger


if __name__ == "__main__":
    logger = get_logger("test")
    logger.success("test success")
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")
