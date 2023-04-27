import getImagesMaps
import pandas as pd
import random
import os
import psutil


arr = [
"1.2667_41.3833",
"-1.7028_-45.1339",
"-4.4565_30.2614",
"4.5833_-75.9167",
"6.7_124.9333",
"7.0333_1.9167",
"8.1667_79.7167",
"-8.4046_115.5395",
"9.2833_-2.4167",
"9.8333_124.0333",
"10.9_-72.8833",
"-10.3167_150.4333",
"11.83_32.8",
"-11.4833_15.8333",
"-12.28_43.7425",
"12.766_75.122",
"-13.2858_-40.9508",
"13.3523_75.4517",
"14.0939_-11.2669",
"14.3333_-89.1833",
"14.6796_101.3976",
"-15.25_-68.1667",
"-15.95_45.9333",
"15.7167_-91.5333",
"-16.9333_49.4833",
"18.3833_-71.85",
"-18.6833_46.1",
"18.6833_-72.05",
"-18.7833_48.675",
"19.0_-98.7167",
"-19.4_47.6333",
"19.7_-72.15",
"19.9_-70.95",
"-19.7833_46.7667",
"19.8333_-71.0167",
"-21.2_47.25",
"21.015_12.3075",
"-23.55_47.5",
"23.67_85.28",
"-23.4333_43.8833",
"23.7789_120.3331",
"24.56_87.89",
"24.658_87.9794",
"-24.1578_-49.8269",
"-24.2888_-47.1332",
"24.6281_83.9199",
"24.4667_32.95",
"25.9_32.7167",
"-25.1833_45.7667",
"25.3295_87.3018",
"25.9081_85.6836",
"25.9667_85.6667",
"-26.7208_-48.9328",
"28.5421_-81.5967",
"29.8_72.8333",
"29.945_78.163",
"-29.6481_27.7336",
"29.7038_-98.6712",
"-30.9_-71.2667",
"30.6175_-86.9636",
"-31.565_143.3678",
"-31.0833_152.8333",
"31.2157_75.6218",
"33.4667_132.4167",
"35.85_8.6333",
"35.1268_-119.4243",
"35.2335_-97.3471",
"36.3619_119.1072",
"36.6718_4.1918",
"37.5_14.3833",
"37.1318_-76.3568",
"37.8833_14.95",
"-38.0333_-60.0833",
"38.737_-77.2339",
"38.981_-77.0028",
"38.8307_-76.7699",
"38.9577_-90.2156",
"40.8_17.1333",
"-41.4_-73.4833",
"41.1119_-73.8121",
"41.3265_-73.0833",
"41.7286_-73.9961",
"41.9798_-71.8735",
"6.5333_122.1667",
"38.7189_-90.4749",
]

generated = pd.read_csv('extra/cities.csv')
del generated['Unnamed: 0']

comp = [type not in local for type,local in zip(generated['type'],generated['local'])]
toErase = generated[comp]
comp2 = [any(local==inLocal for inLocal in toErase['local']) for local in generated['local']]
ncomp = [not (a or b) for a,b in zip(comp,comp2)]
toErase2 = generated[comp2]

generated = generated[ncomp]

for city in toErase.itertuples():
    try:
        name = city.local.replace('simple','satelite')
        os.remove(name)
    except FileNotFoundError:
        pass

for city in toErase.itertuples():
    try:
        os.remove(city.local)
    except FileNotFoundError:
        pass


for erase in arr:
    comp = [erase in a for a in generated['local']]
    comp2 = [erase not in a for a in generated['local']]
    toErase = generated[comp]
    generated = generated[comp2]        
    for city in toErase.itertuples():
        try:
            os.remove(city.local)
            print('apagou!')
        except FileNotFoundError:
            pass


print (generated)

#generated.to_csv('cities.csv')