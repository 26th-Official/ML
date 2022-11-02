from pytube import Playlist
p = Playlist('https://www.youtube.com/playlist?list=PLQVvvaa0QuDfSfqQuee6K8opKtZsh7sA9')

count = 1
for video in p.videos:
    print(f'Downloading: {count}')
    video.streams.filter(res="720p").first().download()
    count+=1