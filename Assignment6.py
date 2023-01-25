import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


df_bb = pd.read_fwf("brain_body.txt")

regr_bb = linear_model.LinearRegression()
body = df_bb[['Body']]
brain = df_bb['Brain']
regr_bb.fit(body, brain)


#Task1
print('Linear Regression Equation: y = {:.4f} * x + {:.4f}'
      .format(regr_bb.coef_[0], regr_bb.intercept_))


#Task2
plt.scatter(body, brain, color='m')
plt.plot(body, regr_bb.predict(body))
plt.title('Brain Weight by Body Weight')
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.show()


#Task3
print('R^2 score for this equation: {:.4f}'
      .format(regr_bb.score(body, brain)))
