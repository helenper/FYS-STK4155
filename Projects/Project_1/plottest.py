import numpy as np
from matplotlib.pyplot import figure, show, plot
from matplotlib.ticker import MaxNLocator

x = np.arange(0.1,10.5,0.1) # arbitrary data
y = np.arange(1.1,11.5,0.1)
print(x.shape, y.shape)


ax = figure().gca()
#ax.plot(x)
plot(x, y)
#ax.yaxis.set_major_locator(MaxNLocator(integer=True))

show()