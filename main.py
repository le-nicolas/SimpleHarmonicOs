import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

#Define the constants
g = 9.81       # Acceleration due to gravity (m/s^2)
l = 1.0        # Length of the pendulum (m)
m = 1.0        # Mass of the pendulum (kg)
omega0 = 0.0        # Initial angular velocity (rad/s)
dt = 0.01           # Time step (s)

# Define the system of ODEs
def derivatives(state, t, g, l):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / l) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# Runge-Kutta 4th Order Method the core stuff
def runge_kutta_4(derivatives, state, t, dt, g, l):
    k1 = dt * derivatives(state, t, g, l)
    k2 = dt * derivatives(state + 0.5 * k1, t + 0.5 * dt, g, l)
    k3 = dt * derivatives(state + 0.5 * k2, t + 0.5 * dt, g, l)
    k4 = dt * derivatives(state + k3, t + dt, g, l)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function to run the simulation and plot the results, just a while loop thing
def run_simulation(theta0):
    # Time integration
    time = [0]
    theta = []
    omega = []
    state = np.array([theta0, omega0])  # Initial state: [theta, omega]

    while True:
        theta.append(state[0])
        omega.append(state[1])
        state = runge_kutta_4(derivatives, state, time[-1], dt, g, l)
        time.append(time[-1] + dt)

        # Stop if the pendulum comes to rest (angular velocity is close to zero)
        if np.abs(state[1]) < 1e-3:
            break

    time = np.array(time[:-1])  # Remove the last time step to match dimensions with theta and omega
    theta = np.array(theta)
    omega = np.array(omega)

    #Data Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, theta, label="Angle (rad)")
    plt.title("Simple Pendulum Simulation")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, omega, label="Angular Velocity (rad/s)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Omega (rad/s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"The pendulum stops at approximately {time[-1]:.2f} seconds.")

# Tkinter GUI for user input
root = tk.Tk()
root.withdraw()  # Hide the root window

# Ask for the initial angle in degrees
user_input = simpledialog.askfloat("Input", "Enter the initial angle in degrees:", minvalue=0.0)

if user_input is not None:
    theta0_user_input = np.deg2rad(user_input)  # Convert degrees to radians
    run_simulation(theta0_user_input)
else:
    print("No input provided.")

root.mainloop()
