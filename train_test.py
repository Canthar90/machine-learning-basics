import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# getting some data to make test/train that evaluale ouer model
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

plt.scatter(x, y)
plt.title("Raw data plot")
plt.xlabel("Minutes spended at the shop")
plt.ylabel("Amount of money spent in shop")
plt.show()

# we are spliting the data into train and test

train_x = x[:80]
train_y = y[:80]

# we are drawing the polynomial regression line to train data

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.title("Train test plot with the polynomial regression line")
plt.xlabel("Minutes spended at the shop")
plt.ylabel("Amount of money spent in shop")
plt.show()

#  to check if model is ok there should be some R2 evaluation this will give us info if the model is i\ok

r2 = r2_score(train_y, mymodel(train_x))
print(r2)


test_x = x[80:]
test_y = y[80:]

# model is ok considering the training date time to check it with test data

r2_test = r2_score(test_y, mymodel(test_x))
print(f"the R2 working with test data it will validate created model: {r2_test}")

# the value is around 0.8 that means that model is ok and we can use it to predict data

plt.scatter(test_x, test_y)
plt.title("Test data plot")
plt.xlabel("Minutes spended at the shop")
plt.ylabel("Amount of money spent in shop")
plt.show()

# now we are predicting data
print(f"We are predicting hom much money will client spent in the shop staying 5 min in it {mymodel(5)}")