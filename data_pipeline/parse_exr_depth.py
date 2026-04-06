import OpenEXR
import Imath
import numpy as np
import os
from PIL import Image

def convert_to_single_channel_exr(input_path, output_path=None):
    exr_file = OpenEXR.InputFile(input_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = exr_file.header()['channels'].keys()
    # print(channels)
    first_channel = sorted(channels)[0]

    # Read single channel
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = exr_file.channel(first_channel, pt)
    data_np = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))

    if output_path is None:
        output_path = input_path

    header_out = OpenEXR.Header(width, height)
    header_out['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    header_out['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    out_exr = OpenEXR.OutputFile(output_path, header_out)
    out_exr.writePixels({'Z': data_np.tobytes()})
    out_exr.close()


def convert_exr_to_png(input_path, output_path=None):
    exr_file = OpenEXR.InputFile(input_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = exr_file.header()['channels'].keys()
    # print(channels)
    first_channel = sorted(channels)[0]

    # Read single channel
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_data = exr_file.channel(first_channel, pt)
    data_np = np.frombuffer(channel_data, dtype=np.float32).reshape((height, width))

    if output_path is None:
        output_path = input_path
    
    # Normalize to 0-255 range
    view_z = np.clip(data_np, 0, 100)
    inv_z = 1 / view_z
    inv_z = (inv_z - np.min(inv_z)) / (np.max(inv_z) - np.min(inv_z)) * 255
    inv_z = inv_z.astype(np.uint8)

    # Save as PNG
    Image.fromarray(inv_z).save(output_path)


def process_depth_exr_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.exr'):
            full_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(folder_path, file_name.replace('.exr', '_depth.exr'))
            convert_to_single_channel_exr(full_path, None)
            # output_path = os.path.join(folder_path, file_name.replace('.exr', '_depth.png'))
            # convert_exr_to_png(full_path, output_path)


if __name__ == "__main__":
    folder = "test_render/c87290b1f8af409ba8bb1652b4de063c/hdr/Z"
    process_depth_exr_folder(folder)
