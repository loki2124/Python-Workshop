# matplotlib



import matplotlib.pyplot as plt
import numpy

# basic line plot
myarray = numpy.array([1, 2, 3])
plt.plot(myarray)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()


# basic scatter plot
x = numpy.array([1, 2, 3])
y = numpy.array([2, 4, 6])
plt.scatter(x,y)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
