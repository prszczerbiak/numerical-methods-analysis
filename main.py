import numpy as np
from src.numerical_methods import *
from src.visualizations import *

def main():
    try:
        D = np.loadtxt('data/data.txt')
    except FileNotFoundError:
        print("Error: Could not find 'data/data.txt'. Please run the data generator first.")
        return

    X = np.linspace(0, 2, 21)
    Y = np.linspace(0, 1, 11)
    
    # Restructure Z values
    k = 0
    Z = np.zeros((Y.shape[0], X.shape[0]))
    for i in range(Y.shape[0]):
        for j in range(X.shape[0]):
            Z[i][j] = D[k][2]
            k += 1

    # 2. Visualizations
    print("Generating 3D Surface Plot...")
    plot_3d_surface(D, X, Y[0], Z[0])
    
    print("Generating 2D Contour Map...")
    plot_2d_contour(D)

    # 3. Statistics
    mean = np.array([calculate_mean(Z[i]) for i in range(Z.shape[0])])
    sd = np.array([calculate_std_dev(Z[i], mean[i]) for i in range(Z.shape[0])])
    median = np.array([calculate_median(Z[i]) for i in range(Z.shape[0])])
    print('Mean:\n', mean)
    print('Standard Deviation:\n', sd)
    print('Median:\n', median)

    # 4. Interpolation and Display
    l = 0
    XdL = np.zeros((5, 5))
    ZdL = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            XdL[i][j] = X[l]
            ZdL[i][j] = Z[0][l]
            l += 1
        l -= 1
        
    AL0 = np.array([lagrange_interpolation(XdL[i], ZdL[i]) for i in range(5)])
    print('Lagrange Polynomial Coefficients:\n', AL0)
    plot_lagrange(X, Z[0], AL0, XdL)

    AS0 = spline_interpolation(X, Z[0])
    print('B-spline Interpolation:\n', AS0)

    plot_lagrange_vs_spline(X, Z[0], AS0, AL0, XdL)

    #5. Approximation
    AAL0 = linear_approximation(X, Z[0])
    print('Linear approximation - coefficients:\n', AAL0)
    plot_linear_approximation(X, Z[0], AAL0)

    AAK0 = quadratic_approximation(X, Z[0])
    print('Quadratic approximation - coefficients:\n', AAK0)
    plot_quadratic_approximation(X, Z[0], AAK0)

    plot_approximations_comparison(X, Z[0], AAL0, AAK0)
    
    # 6. Derivatives & Monotonicity
    P0 = first_derivative(X, Z[0])
    print('First Degree Derivative:\n', P0)

    plot_derivative(X, Z[0], P0)
    plot_monotonicity(X, P0)

if __name__ == "__main__":
    main()