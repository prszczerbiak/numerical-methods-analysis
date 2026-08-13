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
    plt.subplots()
    plt.tricontourf(d[0], d[1], d[2], levels=levels, cmap='viridis')
    mappable = mtp.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(d[2])
    plt.colorbar(mappable, label='Function Value F(x,y) - Z coordinate')
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