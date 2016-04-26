from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'Classic pop and rock', 'Classical','Metal', 'Folk','Dance and Electronica','Jazz and Blues', 'Hip-hop','Pop','Punk','Soul and Reggae'
fracs = [4334, 1874, 2103, 13192,4935,4334,  434,1617,3200,4016]
explode=(0, 0, 0, 0,0, 0, 0, 0,0,0)

pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
pie.facecolor=1
#title('Genre Distribution in the dataset', bbox={'facecolor':'1', 'pad':5})

show()
