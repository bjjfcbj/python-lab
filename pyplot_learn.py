import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

figure = plt.figure(figsize=(30, 30))
gs = gridspec.GridSpec(3, 3)

ppie = figure.add_subplot(gs[0, 0])
pbar = figure.add_subplot(gs[1, 0], projection='polar')
phist = figure.add_subplot(gs[0:-1, 1:-1])

pie_lable = ['one', 'two', 'three', 'four']
pie_size = [1, 2, 3, 9]
pie_explode = [0, 0, 0.1, 0]
ppie.pie(pie_size, explode=pie_explode, labels=pie_lable,
         autopct='%.1f%%', shadow=False, startangle=90)
ppie.axis('equal')

hist_arr = np.random.normal(0, 1, size=100)
phist.hist(hist_arr, 50, histtype='stepfilled', alpha=0.75)

theta = np.linspace(0., 2 * np.pi, 20, endpoint=False)
radii = np.random.random(20)
width = np.pi / 4 * np.random.rand(20)
bars = pbar.bar(theta, 10 * radii, width=width, bottom=0.)
for r, b in zip(radii, bars):
    b.set_facecolor(np.random.random(3))
    b.set_alpha(0.45)

figure.tight_layout()

if __name__ == "__main__":
    plt.show()
