import getImagesMaps
import pandas as pd
import multiprocessing
from tqdm.contrib.concurrent import thread_map


cities = pd.read_csv('extra/worldcities.csv').sample(frac=0.01,random_state=1)

downloader = getImagesMaps.GoogleMapDownloader()


lat = []
long = []
tipo = []
local = []
result = []

for city in cities.itertuples():
    print(city.lat,city.lng)
    downloader.setXY(city.lat,city.lng,16)
    fileA = f"./extra/cities/city-simple({city.lat}|{city.lng}).png"
    fileB = f"./extra/cities/city-satelite({city.lat}|{city.lng}).png"
    try:
        img = downloader.generateImage()
    except IOError as e:
        print(e)
    else:

        lat.append(city.lat)
        long.append(city.long)
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

        lat.append(city.lat)
        long.append(city.long)
        tipo.append('satelite')
        local.append(fileB)
        result.append('city')

        #Save the image to disk
        img = img.resize((256,256))
        img.save(fileB)
        print("The map has successfully been created")


generated = pd.DataFrame.from_dict({
    'lat': lat,
    'long': long,
    'type': tipo,
    'local': local,
    'result': result
})

generated.to_csv('extra/cities.csv')
