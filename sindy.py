import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Lorenz system
def lorenz(z, t, sigma=10, rho=28, beta=8/3):
    x, y, z = z
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Generate training and test data using the Lorenz system
t_train = np.linspace(0, 20, 3000)
x0_train = [0, 2, 20]  # Initial condition
x_train = odeint(lorenz, x0_train, t_train)

t_test = np.linspace(10, 12, 200)
x0_test = x_train[-1]  # to ensure that the test trajectory continues from where the training trajectory ended 
x_test = odeint(lorenz, x0_test, t_test)

# Introduce noise to the training data
rmse = mean_squared_error(x_train, np.zeros_like(x_train), squared=False)
x_train_added_noise = x_train + np.random.normal(0, rmse/20.0, x_train.shape)

feature_names = ['x', 'y', 'z']
threshold_scan = np.linspace(0, 0.1, 11) #how the choice of threshold affects model accuracy and complexity
errors = []

for threshold in threshold_scan:
    opt = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_names=feature_names, optimizer=opt)
    model.fit(x_train_added_noise, t=t_train[1]-t_train[0])
    
    # Predict on test data
    x_pred = model.simulate(x0=x_test[0], t=t_test, u=None)
    
    # Compute the error
    error = mean_squared_error(x_test, x_pred, squared=False)
    errors.append(error)


# Plot threshold (lambda) vs. error
plt.plot(threshold_scan, errors, marker='o')
plt.xlabel('Threshold (Lambda)')
plt.ylabel('RMSE on Test Data')
plt.title('Lambda vs. Error for Lorenz System')
plt.grid(True)
plt.show()

best_threshold = threshold_scan[np.argmin(errors)]
best_optimizer = ps.STLSQ(threshold=best_threshold)
best_model = ps.SINDy(feature_names=feature_names, optimizer=best_optimizer)
best_model.fit(x_train_added_noise, t=t_train[1]-t_train[0])
best_model.print()



# Simulate the true Lorenz system 
x_true = x_test  # or you can simulate for a longer time if you wish

# Simulate the identified system
x_predicted = best_model.simulate(x0=x_test[0], t=t_test)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], label='Actual Lorenz', color='blue')
ax.plot(x_predicted[:, 0], x_predicted[:, 1], x_predicted[:, 2], label='Predicted System', linestyle='dashed', color='red')

# Labeling and show
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Comparison between Actual and Predicted Dynamics')
ax.legend()

plt.show()
