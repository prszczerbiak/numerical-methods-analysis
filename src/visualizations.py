import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mtp
import numpy as np
from src.numerical_methods import lagrange_function, spline_function

def plot_3d_surface(D, X, y_val, Z_row):
    d = D.transpose()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    surf = ax1.plot_trisurf(d[0], d[1], d[2], cmap='viridis', alpha=1.0)
    
    mappable = mtp.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(d[2])
    fig.colorbar(mappable, ax=ax1, label='Function Value F(x,y) - Z coordinate')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1], zoom=1.0)
    ax1.set_title('3D Surface Visualization')

    Y_arr = np.full(X.shape, y_val)
    ax1.plot(X, Y_arr, Z_row, 'ro-', label='Function F(x,y) for y=0')
    plt.legend()
    plt.show()

def plot_2d_contour(D):
    d = D.transpose()
    levels = 10
    fig, ax = plt.subplots()
    
    contour = ax.tricontourf(d[0], d[1], d[2], levels=levels, cmap='viridis')
    
    fig.colorbar(contour, ax=ax, label='Function Value F(x,y) - Z coordinate')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("2D Contour Map Visualization")
    plt.show()

def plot_lagrange(X, Y, A, nodes):
    xt = np.linspace(X[0], X[-1], 100)
    yt = np.zeros(xt.shape[0])
    l = -1

    for i in range(xt.shape[0]):
        if i % 20 == 0:
            l += 1
        yt[i] = lagrange_function(A[l], xt[i], nodes[l])

    plt.subplots()
    plt.plot(xt, yt, 'b-', linewidth=2.0, label='f(x)')
    plt.plot(X, Y, 'ro', label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title("Lagrange Interpolation Function for y=0")
    plt.legend()
    plt.grid()
    plt.show()

def plot_spline(X, Y, K):
    xt = np.linspace(X[0], X[-1], 100)
    yt = np.zeros(xt.shape[0])
    
    for i in range(xt.shape[0]):
        yt[i] = spline_function(xt[i], X, K)
    
    plt.subplots()
    plt.plot(xt, yt, 'b-', linewidth=2.0, label='B-spline function B(x)')
    plt.plot(X, Y, 'ro', label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('B-spline Interpolation for y=0')
    plt.legend()
    plt.grid()
    plt.show()

def plot_lagrange_vs_spline(X, Y, K, A, nodes):
    xt = np.linspace(X[0], X[-1], 100)
    yL = np.zeros(xt.shape[0])
    yS = np.zeros(xt.shape[0])
    l = -1

    for i in range(yL.shape[0]):
        if i % 20 == 0:
            l += 1
        yL[i] = lagrange_function(A[l], xt[i], nodes[l])
        yS[i] = spline_function(xt[i], X, K)

    plt.subplots()
    plt.plot(xt, yL, 'b-', linewidth=2.0, label='Lagrange Interpolation')
    plt.plot(xt, yS, 'g-', linewidth=2.0, label='B-spline Interpolation')
    plt.plot(X, Y, 'ro', linewidth=2.0, label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('Comparison: Lagrange vs B-spline Interpolation')
    plt.legend()
    plt.grid()
    plt.show()

def plot_linear_approximation(X, Y, A):
    x_vals = np.linspace(X[0], X[-1], 100)
    y_vals = A[0] + A[1] * x_vals

    plt.subplots()
    plt.plot(x_vals, y_vals, 'r-', linewidth=2.0, label='Linear Approximation')
    plt.plot(X, Y, 'go', linewidth=2.0, label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('Least Squares Linear Approximation for y=0')
    plt.grid()
    plt.legend()
    plt.show()

def plot_quadratic_approximation(X, Y, A):
    x_vals = np.linspace(X[0], X[-1], 100)
    y_vals = A[0] + A[1] * x_vals + A[2] * (x_vals**2)

    plt.subplots()
    plt.plot(x_vals, y_vals, 'r-', linewidth=2.0, label='Quadratic Approximation')
    plt.plot(X, Y, 'go', linewidth=2.0, label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('Least Squares Quadratic Approximation for y=0')
    plt.grid()
    plt.legend()
    plt.show()

def plot_approximations_comparison(X, Y, A_lin, A_quad):
    x_vals = np.linspace(X[0], X[-1], 100)
    y_lin = A_lin[0] + A_lin[1] * x_vals
    y_quad = A_quad[0] + A_quad[1] * x_vals + A_quad[2] * (x_vals**2)

    plt.subplots()
    plt.plot(x_vals, y_lin, 'b-', linewidth=2.0, label='Linear Approximation')
    plt.plot(x_vals, y_quad, 'g-', linewidth=2.0, label='Quadratic Approximation')
    plt.plot(X, Y, 'ro', linewidth=2.0, label='Given nodes [x,z]')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('Comparison: Linear vs Quadratic Approximation')
    plt.grid()
    plt.legend()
    plt.show()

def plot_derivative(X, Y, Yp):
    plt.subplots()
    plt.plot(X, Y, 'bo-', linewidth=2.0, label='Nodes [x,y]')
    plt.plot(X, Yp, 'ro-', linewidth=2.0, label='Approximated Derivative')
    plt.xlabel('X')
    plt.ylabel('F(x,y)')
    plt.title('Derivative of F(x,y) for y=0')
    plt.grid()
    plt.legend()
    plt.show()

def plot_monotonicity(X, Y):
    M = np.zeros(Y.shape[0])
    for i in range(M.shape[0]):
        if Y[i] > 0: M[i] = 1
        elif Y[i] == 0: M[i] = 0
        else: M[i] = -1

    plt.subplots()
    plt.plot(X, M, 'ro-', linewidth=2.0, label='Monotonicity')
    plt.xlabel('X')
    plt.yticks([-1, 0, 1], ['Decreasing', 'Constant/Extremum', 'Increasing'])
    plt.ylabel('F(x,y)')
    plt.title('Monotonicity of F(x,y) for y=0')
    plt.grid()
    plt.legend()
    plt.show()

