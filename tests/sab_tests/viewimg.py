import sunpy.map
import matplotlib.pyplot as plt
import os

events_path = '/home/saber/Downloads/2013_lvl1/'

def draw_all():
    for filename in os.listdir(events_path)[0:5]:
        img = sunpy.map.Map(str(os.path.join(events_path, filename)))

        plt.figure()
        img.plot()

    plt.show()
    plt.close()


draw_all()
