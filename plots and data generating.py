import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

x = numpy.random.uniform(low=0.0, high=15.0, size=3000)

plt.hist(x, 10)
plt.show()

y = numpy.random.normal(loc=5.0, scale=1.0, size=100000)

plt.hist(y, 100)
plt.show()

# scatter plot

x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]  # age of a car
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]  # speed of a car

slope, intercept, r, p, std_err = stats.linregress(x, y)
# make linear regression for the x and y values


def myfunc(x):
    """make new value that represents where on the y-axis the corresponding x value will be placed"""
    return slope * x + intercept


mymodel = list(map(myfunc, x))
# runs every value of the x array to the functio this will give us new y values


plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

print(r)  # this is relationship value goes from -1 to 1 when 0 means that there is no relationship between
# values

# with function above we can predict future values

speed = myfunc(10)
print(speed)

# below function that will not be good described by linear regression

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x,y)

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

print(f"relationship for second data set should be low lets check it {r}")


x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]


# makes polynominal model 1 dimentional
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# the specifics about how line from created model will be displayed
myline = numpy.linspace(1, 22, 100)

plt.scatter(x, y)
# draws the polynormal regression
plt.plot(myline, mymodel(myline))
plt.show()

print(f"parameter for describing how well Polynomial Regression describes relation between datapoints,\n"
      f"is called r2_score(y, model(x)) we need to import it from sklearn.metrics import r2_score\n"
      f"and in this case it is: {r2_score(y, mymodel(x))}")

print(f"using Polynomial Regression we can predict values for example for x=17 : {myfunc(17)}")


# again new case when the Polynominal Regression would be usles

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(2, 95, 100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

print(f"lets check the value of r-squared if it fits ouer model well {r2_score(y, mymodel(x))}")