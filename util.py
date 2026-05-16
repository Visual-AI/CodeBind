# verify video section adopt from https://github.com/hutuxian/RSPNet/blob/main/utils/verify_video.py
# to enable video verification, you need to have ffmpeg installed.
import imageio
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import json
import os
import shutil


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    # if image_numpy.shape[2] == 1:
    #     image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    image_numpy = np.uint8(image_numpy.astype(int))

    if image_numpy.shape[0] == 3:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    imageio.imwrite(image_path, image_numpy)


# img_arr = 255*rec_img.cpu().detach().numpy()[0,...]

# image_path = 'rec_rgb.jpg'
# inv_tensor_ori_arr = 255*inv_tensor_ori.cpu().numpy()[0,...]
# inv_tensor_ori_arr = np.uint8(inv_tensor_ori_arr.astype(int))
# inv_tensor_ori_arr = np.transpose(inv_tensor_ori_arr, (1, 2, 0))
# imageio.imwrite('rgb_ori.jpg', inv_tensor_ori_arr)
    
def is_empty_dir(path):
    for _, _, files in os.walk(path):  
        if files:
            return False  
    return True  

def copy_files_from_dir_to_dir(src_dir, dst_dir):  
    """  
    复制源文件夹内的所有文件到目标文件夹，不包括子文件夹和它们的文件。   
    """  
    for filename in os.listdir(src_dir):  
        src_file = os.path.join(src_dir, filename)  
        dst_file = os.path.join(dst_dir, filename)  
  
        if os.path.isfile(src_file):  
            shutil.copy2(src_file, dst_file)

logger = logging.getLogger(__name__)

async def verify(video_path: Path, failed: list):
    proc = await asyncio.create_subprocess_exec(
        'ffprobe', '-loglevel', 'warning', '-show_streams', '-select_streams', 'v', '-print_format', 'json', str(video_path),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error('ffprobe %s failed with return code %d', video_path, proc.returncode)
        failed.append(video_path)
        return

    probe_result = json.loads(stdout)
    if not probe_result['streams']:
        logger.error('No Video found for "%s"', video_path)
        failed.append(video_path)
        return


async def verify_video(input_dir_train, input_dir_val, save_dir, second_dir_exist=False, jobs=32):

    input_dir_train = Path(input_dir_train)
    input_dir_val = Path(input_dir_val)
    tasks = set()
    def search_files():
        if second_dir_exist:
            yield from input_dir_train.glob('**/*.mp4')
            yield from input_dir_train.glob('**/*.avi')
            yield from input_dir_val.glob('**/*.mp4')
            yield from input_dir_val.glob('**/*.avi')
        else:
            yield from input_dir_train.glob('*.mp4')
            yield from input_dir_train.glob('*.avi')
            yield from input_dir_val.glob('*.mp4')
            yield from input_dir_val.glob('*.avi')
    pending_videos = sorted(search_files())

    failed = list()

    with tqdm(total=len(pending_videos), smoothing=0.1) as progress:
        while True:
            while len(tasks) < jobs and pending_videos:
                raw_video = pending_videos.pop()
                t = asyncio.create_task(verify(raw_video, failed))
                tasks.add(t)

            if not tasks:
                break

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                await t
                progress.update()

    # print('The following video failed the test: ')
    # for p in failed:
    #     print(p.relative_to(input_dir))

    with open(save_dir, 'w') as f:
        for p in failed:   
            f.write("%s\n" % p)  

def unavailable_video_detect(video_dir_train, video_dir_val, save_dir, second_dir_exist):
    asyncio.run(verify_video(video_dir_train, video_dir_val, save_dir, second_dir_exist))

# if __name__ == "__main__":
#     unavailable_video_detect('.datasets/K400/train', '.datasets/K400/val', 
#                              save_dir='.datasets/k400_unreadable_video.txt', second_dir_exist=True)

    


