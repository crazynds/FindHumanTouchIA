import getImagesMaps
import pandas as pd
import multiprocessing
from tqdm.contrib.concurrent import thread_map


cities = pd.read_csv('extra/worldcities.csv').sample(frac=0.01,random_state=1)

downloader = getImagesMaps.GoogleMapDownloader()


lats = []
longs = []
tipo = []
local = []
result = []

for city in cities.itertuples():
    lat,long = city.lat,city.lng
    downloader.setXY(lat,long,16)
    fileA = f"./extra/cities/city-simple({lat}_{long}).png"
    fileB = f"./extra/cities/city-satelite({lat}_{long}).png"
    try:
        img = downloader.generateImage()
    except IOError as e:
        print(e)
    else:

        lats.append(lat)
        longs.append(long)
        tipo.append('simple')
        local.append(fileA)
        result.append('city')

        #Save the image to disk
        img = img.resize((256,256))
        img.save(fileA)
        print("The map has successfully been created")
    try:
        img = downloader.generateImage(satelite = True)
    except IOError as e:
        print(e)
    else:

        lats.append(lat)
        longs.append(long)
        tipo.append('satelite')
        local.append(fileB)
        result.append('city')

        #Save the image to disk
        img = img.resize((256,256))
        img.save(fileB)
        print("The map has successfully been created")


generated = pd.DataFrame.from_dict({
    'lat': lats,
    'long': longs,
    'type': tipo,
    'local': local,
    'result': result
})

generated.to_csv('extra/cities.csv')
