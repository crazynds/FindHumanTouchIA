import getImagesMaps
import pandas as pd
import random
import os
import psutil

downloader = getImagesMaps.GoogleMapDownloader()
zoom = 16


if os.path.exists('extra/cities.csv'):
    generated = pd.read_csv('extra/cities.csv')
    del generated['Unnamed: 0']
else:
    generated = pd.DataFrame.from_dict({
        'lat': [],
        'long': [],
        'type': [],
        'local': [],
        'result': []
    })


if len(generated)<1500 and False:

    cities = pd.read_csv('extra/worldcities.csv').sample(frac=0.005)

    for city in cities.itertuples():
        lat,long = city.lat,city.lng
        downloader.setXY(lat,long,zoom)
        fileA = f"./extra/cities/city-simple({lat}_{long}).png"
        fileB = f"./extra/cities/city-satelite({lat}_{long}).png"
        if os.path.exists(fileA):
            continue
        try:
            img = downloader.generateImage()
        except IOError as e:
            print(e)
        else:
            generated.loc[len(generated)] = [
                lat,
                long,
                'simple',
                fileA,
                'city'
            ]
            #Save the image to disk
            img = img.resize((256,256))
            img.save(fileA)
            print("The map has successfully been created")
        try:
            img = downloader.generateImage(satelite = True)
        except IOError as e:
            print(e)
        else:

            generated.loc[len(generated)] = [
                lat,
                long,
                'satelite',
                fileB,
                'city'
            ]

            #Save the image to disk
            img = img.resize((256,256))
            img.save(fileB)
            print("The map has successfully been created")

    generated.to_csv('extra/cities.csv')


#random.seed(10)

if os.path.exists('extra/random.csv'):
    generated = pd.read_csv('extra/random.csv')
    del generated['Unnamed: 0']
else:
    generated = pd.DataFrame.from_dict({
        'lat': [],
        'long': [],
        'type': [],
        'local': [],
        'result': []
    })


options = ['nature','farm','city','roads','ocean']

while len(generated)<600:
    lat = random.random() * 180 - 90
    long = random.random() * 180 - 90
    downloader.setXY(lat,long,zoom)
    try:
        img = downloader.generateImage(satelite = True)
    except IOError as e:
#        print(e)
        continue
    try:
        img2 = downloader.generateImage(satelite = False)
    except IOError as e:
#        print(e)
        continue
    bits = list(img.getdata())
    a = bits[0]
    if all(a == b for b in bits):
        print('pass')
        continue
    img.show()
    print(f'https://www.google.com.br/maps/@{lat},{long},10362m')
    print('1-nature\n2-farm\n3-city\n4-roads\n5-ocean\n0-discard (-1 exit)')
    v = int(input())
    for proc in psutil.process_iter():
        if "Photos" in proc.name():
            proc.kill()
    if v==0:
        continue
    if v==-1:
        break
    v-=1

    fileA = f"./extra/random/{options[v]}-simple({lat}_{long}).png"
    fileB = f"./extra/random/{options[v]}-satelite({lat}_{long}).png"

    img.save(fileB)
    img2.save(fileA)
    generated.loc[len(generated)] = [
        lat,
        long,
        'simple',
        fileA,
        options[v]
    ]
    generated.loc[len(generated)] = [
        lat,
        long,
        'satelite',
        fileA,
        options[v]
    ]

generated.to_csv('extra/random.csv')




