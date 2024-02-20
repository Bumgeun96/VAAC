from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

def convert_part_of_mp4_to_gif(input_path, output_path, start_time, end_time):
    # 동영상의 일부분을 추출하여 새로운 동영상 파일을 생성
    temp_clip_path = "temp_clip.mp4"
    ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=temp_clip_path)
    
    # 새로 생성된 동영상 파일을 GIF로 변환
    clip = VideoFileClip(temp_clip_path)
    clip.write_gif(output_path)

    # 임시 파일 제거
    os.remove(temp_clip_path)


mp4_file = './video/SparseWalker2d-v1(vaac)'
convert_part_of_mp4_to_gif(mp4_file+'.mp4', mp4_file+'.gif',start_time=20, end_time=60)