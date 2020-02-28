import datetime
import multiprocessing
import time

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas

tk = Tk()
canvas = Canvas(tk, width=1000, height=1000)
tk.title("Matrix percolation")
canvas.pack()
COLORS = ('#242424', '#2eabd9', '#c28234')  # 0=black(stone), 1=blue(water), 2=brown(sand)


def animate_matrix(matrix: np.ndarray, size):
    """
    This function animates a matrix into coloured blocks
    Args:
        matrix: A numpy ndarray object
        size: in pixels of how large the whole grid should animate to

    Returns: none

    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    box = size / max(m, n)
    gap = 2 + box if max(m, n) < 25 else box
    for i in range(m):
        for j in range(n):
            color = COLORS[matrix[i, j]]
            canvas.create_rectangle(100 + (gap) * j, 100 + (gap) * i, 100 + box + (gap) * j,
                                    100 + box + (gap) * i, fill=color)
    tk.update()
    tk.mainloop()


def generate_matrix(N, p):
    """
    Generates a random binary matrix to use as the base for percolation
    Args:
        N: The size of the matrix N x N
        p: The probability of zeros. P(2) = 1-p and P(0) = p

    Returns: Numpy ndarray

    """
    return np.random.choice([0, 2], size=(N, N), p=[p, 1 - p])


def percolate(matrix: np.ndarray):
    """Simulate percolation for unlimited water, flowing from the top row
    Returns: Numpy ndarray of 0 (rock), 1 (water) and 2 (sand)
    """
    N = matrix.shape[0]

    # This gets the first row, from the matrix. The starting blocks of the water
    for i in range(N):
        if matrix[0, i] == 2:
            matrix[0, i] = 1

# This is the percolation algorithm. It loops through 'blank' matrix and adds in 1 if water can flow
    for i in range(1, N):
        for j in range(N):
            if matrix[i, j] == 2:
                if j == 0:
                    if matrix[i - 1, j + 1] == 1 or matrix[i - 1, j] == 1 \
                             or matrix[i, j + 1] == 1:
                        matrix[i, j] = 1
                elif j == N-1:
                    if matrix[i - 1, j] == 1 \
                            or matrix[i - 1, j - 1] == 1:
                        matrix[i, j] = 1
                else:
                    if matrix[i - 1, j + 1] == 1 or matrix[i - 1, j] == 1 \
                            or matrix[i - 1, j-1] == 1 or matrix[i, j+1] == 1:
                        matrix[i, j] = 1
    return matrix


def percolate_onedrop(matrix: np.ndarray):
    """
    Percolate water through rocks and sand for only one drop, starting in the middle on the top row
    Args:
        matrix: array of sand and stone (where sand is porus and stone is not)

    Returns: matrix showing where the water flowed to

    """
    N = matrix.shape[0]

    waterdrop_h = int(N / 2)  # H for horizontal position
    waterdrop_v = 0  # V for vertical position
    matrix[waterdrop_v, waterdrop_h] = 1
    possible_to_flow = True
    while possible_to_flow:
        if N == waterdrop_v + 2:  # checks if the water has reached the bottom
            possible_to_flow = False
        if N-1 > waterdrop_h > 0:  # This incidates the drop isn't on either side of the matrix
            if matrix[waterdrop_v + 1, waterdrop_h] == 2:  # checks the position directly below
                waterdrop_v += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v + 1, waterdrop_h - 1] == 2:  # checks the position down-left
                waterdrop_v += 1
                waterdrop_h -= 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v + 1, waterdrop_h + 1] == 2:  # checks the down-right position
                waterdrop_v += 1
                waterdrop_h += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v, waterdrop_h + 1] == 2:  # checks the position directly right
                waterdrop_h += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            else:
                possible_to_flow = False  # if none of the options are sand, then end the loop
        elif waterdrop_h == 0:  # handles the drop if it is on the left side of the matrix
            if matrix[waterdrop_v + 1, waterdrop_h] == 2:  # checks the position directly below
                waterdrop_v += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v + 1, waterdrop_h + 1] == 2:  # checks the down-right position
                waterdrop_v += 1
                waterdrop_h += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v, waterdrop_h + 1] == 2:  # checks the position directly right
                waterdrop_h += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            else:
                possible_to_flow = False  # if none of the options are sand, then end the loop
        elif waterdrop_h == N-1:  # Handles the drop if its on the right hand side of the matrix
            if matrix[waterdrop_v + 1, waterdrop_h] == 2:  # checks the position directly below
                waterdrop_v += 1
                matrix[waterdrop_v, waterdrop_h] = 1
            elif matrix[waterdrop_v + 1, waterdrop_h - 1] == 2:  # checks the position down-left
                waterdrop_v += 1
                waterdrop_h -= 1
                matrix[waterdrop_v, waterdrop_h] = 1
            else:
                possible_to_flow = False  # if none of the options are sand, then end the loop

    return matrix


def test_percolation(matrix: np.ndarray) -> bool:
    """Checks if water reached the bottom"""
    N = matrix.shape[0]
    for j in range(N):
        if matrix[N - 1, j] == 1:
            return True
    return False


# NEXT STEP: write simulation tests for a thousand cases


def run_sim(size: int, p: float, iterations: int):
    """
    Simulates the percolation algorithm on randomly generated matrices and returns average successes
    Args:
        size: size of the square matrix (N in NxN)
        p: probability of rocks being generated for each cell of the matrix
        iterations: number of realisations

    Returns: The number of times water reached the bottom as a percentage of total realisations

    """
    number_of_times_bottom_reached = 0
    for i in range(iterations):
        matrix = generate_matrix(size, p)
        percolation = percolate(matrix)
        if test_percolation(percolation):
            number_of_times_bottom_reached += 1

    # This prints for each value of p if you run sim_vary_p. Comment out to avoid if necessary
    # print("\n\nMatrix size:", size, "x", size, "with p =", p)
    # print("In", iterations, "iterations, there were ", number_of_times_bottom_reached, "successes")
    # print("This is an average rate of", number_of_times_bottom_reached / iterations)
    return number_of_times_bottom_reached / iterations


def sim_vary_p(size: int, iterations: int, steps: float, graph: bool = True) -> float:
    """
    Run a simulation across different values of P, and graph the results with matplotlib
    Args:
        graph: whether or not it should draw a graph of p
        size: size of the square matrix (N in NxN)
        iterations: number of realisations for each value of p
        steps: the increase of p after each sim. (smaller values give more accurate graph results)

    Returns: critical value of p (where in less than half of the matrices the water reaches bottom)

    """
    # print("Simulation started at", datetime.datetime.now())
    x_values = np.arange(0, 1, steps)
    y_values = multiprocessing.Pool().map(simple_simulation, x_values)
    # hit_zero = False
    # for p in x_values:
    #     if not hit_zero:
    #         y_values.append(run_sim(size, p, iterations))
    #     else:
    #         y_values.append(0)
    #     if y_values[-1] == 0:  # once zero is reached, it will stop simulating and only append zeros
    #         hit_zero = True  # this will save useless computation
    # print("Simulation finished at", datetime.datetime.now())
    # print(x_values)
    # print(y_values)

    # This finds the critical value of P. It could be improved by interpolating the data
    closest = 1
    Pc, Py = 0, 0
    for y in range(len(y_values)):
        if abs(0.5 - y_values[y]) < closest:
            closest = abs(0.5 - y_values[y])
            Pc = round(x_values[y], 5)
            Py = y_values[y]
    print(size, 'x', size, 'simulation completed at', datetime.datetime.now(), "- Pc=", Pc, "Py=", Py)
    if graph:
        plt.plot(x_values, y_values, color="k")
        plt.plot([Pc, Pc], [1, 0], label="Critical value of P", color="r")
        plt.plot([0, 1], [Py, Py], color="r")
        plt.plot([Pc], [Py], "o", color="orange")
        plt.text(Pc - 0.1, 1, "Critical value of P=" + str(Pc))
        plt.text(0.75, Py + 0.005, "" + str(round(Py * 100)) + "% success rate")
        plt.ylabel("Average probability of water reaching the bottom")
        plt.xlabel("p")
        plt.title(
            "Percolation simulation: N=" + str(size) + ", nrep=" + str(iterations) + ", P steps=" + str(
                steps))

        plt.show()
    return Pc


def graph_critical_value_of_p():
    labels = ['N=10', 'N=50', 'N=100', 'N=200', 'N=400']
    y_ax = np.arange(len(labels))
    values = [0.46, 0.3, 0.25, 0.21, 0.175]
    plt.bar(y_ax, values, align='center', alpha=0.5)
    plt.xticks(y_ax, labels)
    plt.ylabel("Critical value of P")
    plt.xlabel("Size of matrix (N)")
    plt.title("Pc for different size matrices")
    plt.show()


def simple_N(n):
    nreps = 1000
    steps = 0.0025
    return sim_vary_p(n, nreps, steps, False)


def simple_simulation(p):
    N = 400
    nreps = 1000
    return run_sim(N, p, nreps)


def sim_multi(nlist):
    x_values = nlist
    y_values = multiprocessing.Pool().map(simple_N, x_values)
    print(x_values)
    print(y_values)
    plt.plot(x_values, y_values, color="k")
    plt.ylabel("Critical value of P")
    plt.xlabel("Size of matrix (N)")
    plt.title("Critical value of P for different size matrices")
    plt.show()

if __name__ == "__main__":
    print("simulation started")
    start_time = time.perf_counter()
    lsit = np.arange(10, 400, 10)
    #sim_multi(lsit)
    # graph_critical_value_of_p()
    # To run many simulations and graph the results, use this function:
    # sim_vary_p(size=10, iterations=100000, steps=0.005)
    # sim_vary_p(size=50, iterations=100000, steps=0.005)
    sim_vary_p(size=400, iterations=1_000, steps=0.01)
    # sim_vary_p(size=200, iterations=100000, steps=0.005)
    # sim_vary_p(size=400, iterations=100000, steps=0.005)
    # This just times how long the simulation took (can easily be north of 1 hour)
    finish_time = time.perf_counter()
    time_taken = finish_time - start_time
    print("Time taken:", time_taken, "seconds")

    # To simulate just one percolation and animate it with graphics, use this syntax
    # matrix1 = generate_matrix(N=50, p=0.50)  # N - size of matrix, p - proportion of rock
    # perc = percolate(matrix=matrix1)
    # animate_matrix(matrix=perc, size=800)  # size is number of pixels to draw the grid
    # Next step is to run simulations on N=1, N=2, N=3, N=4... N=1,000 and graph the Pc
