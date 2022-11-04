from pytube import Playlist
# from pytube.helpers import safe_filename 
# import os
p = Playlist('https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7')

count = 1
for video in p.videos:
    print(f'Downloading: {count}')
    video.streams.filter(res="720p").first().download(output_path="D:\\Reinforcement Learning Tutorial\\",filename_prefix=f"{count}_")
    # name = safe_filename(video.title)
    # os.rename(f"D:\\ML Tutorial\\{name}.mp4",f"D:\\ML Tutorial\\{count}_{name}.mp4")
    count+=1

