import datetime
import gc
import multiprocessing
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Canvas

tk = Tk()
canvas = Canvas(tk, width=1000, height=1000)
tk.title("Matrix percolation")
canvas.pack()
COLORS = ('#242424', '#2eabd9', '#c28234')  # 0=black(stone), 1=blue(water), 2=brown(sand)


def animate_matrix(matrix: np.ndarray, size: int = 800, delay: float = 0):
    """
    This function animates a matrix into coloured blocks.
    Args:
        delay: total time to draw the matrix in seconds. Use for after percolation has occured.
        matrix: A numpy ndarray object
        size: in pixels of how large the whole grid should animate to

    """
    canvas.delete("all") if not delay else ""
    m = matrix.shape[0]
    n = matrix.shape[1]
    box = size / max(m, n)
    gap = 2 + box if max(m, n) < 25 else box
    for i in range(m):
        time.sleep(delay / m)
        for j in range(n):
            color = COLORS[matrix[i, j]]
            canvas.create_rectangle(100 + (gap) * j, 100 + (gap) * i, 100 + box + (gap) * j,
                                    100 + box + (gap) * i, fill=color)
            tk.update() if delay else ""
    tk.update()
    time.sleep(0.3)  # adds a delay after drawing the matrix, this is useful for multiple animations


def animate_diagonal_matrix(matrix: np.ndarray, size: int = 800, delay: float = 0):
    """
    This function animates a matrix specifically for the diagonal percolation algorithm.
    Args:
        delay: total time to draw the matrix in seconds. Use for after percolation has occured.
        matrix: A numpy ndarray object
        size: in pixels of how large the whole grid should animate to

    """
    canvas.delete("all") if not delay else ""
    m = matrix.shape[0]
    n = matrix.shape[1]
    box = size / max(m, n)
    gap = 2 + box if max(m, n) < 25 else box

    for row in range(n):
        for column in range(row + 1):
            i = column
            j = row - column
            color = COLORS[matrix[i, j]]
            canvas.create_rectangle(100 + gap * j, 100 + gap * i, 100 + box + gap * j,
                                    100 + box + gap * i, fill=color)
        tk.update()
        time.sleep(delay / n)
    for row in range(n):
        for column in range(n - row - 1):
            i = n - column - 1
            j = row + column + 1
            color = COLORS[matrix[i, j]]
            canvas.create_rectangle(100 + gap * j, 100 + gap * i, 100 + box + gap * j,
                                    100 + box + gap * i, fill=color)
        tk.update()
        time.sleep(delay / n)

    tk.update()
    time.sleep(0.3)  # adds a delay after drawing the matrix, this is useful for multiple animations


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
                elif j == N - 1:
                    if matrix[i - 1, j] == 1 \
                            or matrix[i - 1, j - 1] == 1:
                        matrix[i, j] = 1
                else:
                    if matrix[i - 1, j + 1] == 1 or matrix[i - 1, j] == 1 \
                            or matrix[i - 1, j - 1] == 1 or matrix[i, j - 1] == 1:
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
        if N - 1 > waterdrop_h > 0:  # This incidates the drop isn't on either side of the matrix
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
        elif waterdrop_h == N - 1:  # Handles the drop if its on the right hand side of the matrix
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


def percolate_diagonally(matrix: np.ndarray):
    """Directed percolation from one corner, and unlimited water"""
    n, m = matrix.shape
    matrix[0, 0] = 1
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == 2:
                if i == 0:
                    if matrix[i, j - 1] == 1:
                        matrix[i, j] = 1
                elif j == 0:
                    if matrix[i - 1, j] == 1:
                        matrix[i, j] = 1
                else:
                    if matrix[i, j - 1] == 1 or matrix[i - 1, j - 1] == 1 or matrix[i - 1, j] == 1:
                        matrix[i, j] = 1
    return matrix


def test_percolation(matrix: np.ndarray) -> bool:
    """Checks if water reached the bottom"""
    N = matrix.shape[0]
    for j in range(N):
        if matrix[N - 1, j] == 1:
            return True
    return False


def test_diagonal_percolation(matrix: np.ndarray) -> bool:
    """Checks if water reached the bottom or the right hand side. Use for diagonal percolation"""
    N = matrix.shape[0]
    for j in range(N):
        if matrix[N - 1, j] == 1:
            return True
        if matrix[j, N - 1] == 1:
            return True
    return False


def run_sim(size: int, p: float, iterations: int, algorithm=percolate_onedrop,
            print_output: bool = False):
    """
    Simulates the percolation algorithm on randomly generated matrices and returns average successes
    Args:
        algorithm: the percolation algorithm to use in the simulation (percolate_onedrop by default)
        print_output: whether or not to print the number of percolations
        size: size of the square matrix (N in NxN)
        p: probability of rocks being generated for each cell of the matrix
        iterations: number of realisations

    Returns: The number of times water reached the bottom as a percentage of total realisations

    """
    number_of_times_bottom_reached = 0
    # # This method uses multiprocessing - causes memory errors unless you have loads of RAM
    # args = [(size, p, algorithm)]*iterations
    # matrices = []
    # for i in range(iterations):
    #     matrices.append(generate_matrix(size, p))
    # print("random matrices generated, sleeping 5 seconds")
    # time.sleep(5)
    # args = [(m, algorithm) for m in matrices]
    # successes = multiprocessing.Pool().starmap(single_sim, args)
    # number_of_times_bottom_reached = sum(successes)
    # print("Processing done - waiting another 5 seconds of summer")
    # time.sleep(5)

    # This way does it without multiprocessing
    test_algorithm = test_diagonal_percolation if algorithm.__name__ == 'percolate_diagonally' else test_percolation
    for i in range(iterations):
        matrix = generate_matrix(size, p)
        algorithm(matrix)
        if test_algorithm(matrix):
            number_of_times_bottom_reached += 1

    # This prints for each value of p if you run sim_vary_p
    if print_output:
        print("\n\nMatrix size:", size, "x", size, "with p =", p)
        print("In", iterations, "iterations, there were ", number_of_times_bottom_reached,
              "successes")
        print("This is an average rate of", number_of_times_bottom_reached / iterations)
    return number_of_times_bottom_reached / iterations


def find_critical_p(size, iterations, accuracy, algorithm, graph=True):
    """
    This uses binary search to find the critical value of p for given properties
    Args:
        size: size of matrix, N ie nxn
        iterations: number of realisations per p value. Higher is more accurate
        accuracy: the acceptable deviation from the actual critical value (usually 0.01)
        algorithm: the percolation algorithm to find the critical value of
        graph: whether or not to graph the outputs. (don't use when running sim_vary_N)

    Returns: the critical value of p

    """
    x_values = []
    y_values = []
    min_accept = 0.5 - accuracy
    max_accept = 0.5 + accuracy
    step = 0.5
    start = 0
    stop = 1.1
    for k in range(100_000_000):  # This is more than enough for most sims, but wont let it run away
        for i in np.arange(start, stop, step):
            pc = run_sim(size, i, iterations, algorithm)
            x_values.append(i)
            y_values.append(pc)
            # print("{}% success rate when p={}".format(round(pc*100), i))
            if min_accept < pc < max_accept:
                if graph:
                    plt.plot(x_values, y_values, "o", color="k", markersize=4)
                    plt.plot([i, i], [1, 0], label="Critical value of P", color="r")
                    plt.plot([0, 1], [pc, pc], color="r")
                    plt.plot([i], [i], "o", color="orange")
                    plt.text(i - 0.1, 1, "Critical value of P=" + str(i))
                    plt.text(0.75, pc + 0.005, "" + str(round(pc * 100)) + "% success rate")
                    plt.ylabel("Average probability of water reaching the bottom")
                    plt.xlabel("p")
                    plt.title(
                        "{} simulation: N={} nrep={}, accuracy={}".format(algorithm.__name__, size,
                                                                          iterations, accuracy))
                    plt.ylim(0, 1)
                    plt.show()
                print(x_values)
                print(y_values)
                print(size, 'x', size, 'simulation completed at', datetime.datetime.now(),
                      "- when p=",
                      i, "success rate=", pc, "algorithm =", algorithm.__name__)
                return i
            elif pc < 0.5:

                stop = round(i, 5)
                start = round(stop - step, 5)
                step /= 3
                stop += step
                print("P is in the range of {} to {}".format(start, stop))
                # print("Now stepping through range({}, {}, {})".format(start, stop, step))
                break
    print("Either you are asking for unreasonable accuracy, or something has gone wrong in the app")


def sim_vary_p(size, iterations, steps, algorithm=percolate_onedrop, graph=True,
               multi: bool = True) -> float:
    """
    Run a simulation across different values of P, and graph the results with matplotlib
    This is a less efficient way of finding the critical value of P.
    It is recommended to use find_critical_p() unless you want a continuous graph.
    Args:
        algorithm: the percolation algorithm to use in the simulation (percolate_onedrop by default)
        multi: whether or not to use multithreading to speed up performance-dont use with sim_vary_N
        graph: whether or not it should draw a graph of p
        size: size of the square matrix (N in NxN)
        iterations: number of realisations for each value of p
        steps: the increase of p after each sim. (smaller values give more accurate graph results)

    Returns: critical value of p (where in less than half of the matrices the water reaches bottom)

    """
    # print("Simulation started at", datetime.datetime.now())

    # This is the main computation. It uses multiprocessing to spread the computation across threads
    x_values = np.arange(0.38,0.45,steps) if algorithm.__name__ == 'percolate_diagonally' else np.arange(0.1,0.9,steps)
    args = [(size, x, iterations, algorithm) for x in x_values]
    y_values = multiprocessing.Pool().starmap(run_sim, args) if multi else [run_sim(*arg) for arg in args]

    # This finds the critical value of P. It could be improved by interpolating the data
    print(x_values)
    print(y_values)
    closest = 1
    Pc, Py = 0, 0
    for y in range(len(y_values)):
        if abs(0.5 - y_values[y]) < closest:
            closest = abs(0.5 - y_values[y])
            Pc = round(x_values[y], 5)
            Py = y_values[y]
    print(size, 'x', size, 'simulation completed at', datetime.datetime.now(), "- Pc=", Pc, "Py=",
          Py, "algorithm =", algorithm.__name__)

    # This is just the plotting of the graph using matplotlib, if the graph variable is true
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
            "Percolation simulation: N=" + str(size) + ", nrep=" + str(
                iterations) + ", P steps=" + str(
                steps))

        plt.show()
    return Pc


def sim_vary_N(nlist: List[int], nreps: int, steps: float, algorithm, graph: bool = False):
    """
    This function is to show the relationship between the size of a matrix, N, and its percolation
    constant, Pc. It will draw a graph of Pc for the N values provided in nlist.
    Args:
        algorithm: the percolation algorithm to use in the simulation
        nlist: a list of integers, the sizes of matrix to run the algorithm on. eg. [5, 10, 50]
        nreps: number of iterations for each matrix size, for each value of p. High = more accurate.
        I recommend about 10,000 for an accurate graph (smoother).
        steps: the increase of p each run. The interval of p. eg p=0.01 -> p~ [0.01, 0.02, 0.03 ...]
        graph: whether or not to draw individual graphs for each matrix size N simulated.

    """
    x_values = nlist
    args = [(n, nreps, steps, algorithm) for n in nlist]
    y_values = multiprocessing.Pool().starmap(find_critical_p, args)

    print(x_values)
    print(y_values)

    # This is just plotting the results
    plt.plot(x_values, y_values, color="k")
    plt.ylabel("Critical value of P")
    plt.xlabel("Size of matrix (N)")
    plt.title("Critical value of P for different size matrices")
    plt.show()


if __name__ == "__main__":
    # Examples of how to use functions:

    # Generate Matrix
    random_matrix = generate_matrix(35, 0.5)  # This generates a 35x35 matrix, half sand, half stone

    # Animate Matrix
    animate_matrix(random_matrix, 800)  # This draw the matrix, random_matrix, 800 pixels square

    # Percolate Onedrop - This will simulate just one drop of water, starting in the top middle
    percolate_onedrop(random_matrix)  # flowing down the matrix through the sand
    animate_matrix(random_matrix, delay=3)  # To visualise the results of the onedrop percolation

    # Percolate
    percolate(random_matrix)  # This will simulate unlimited water flowing down the matrix
    animate_matrix(random_matrix, delay=3)  # To visualise the results of the percolation

    # Percolate diagonally - Directed percolation
    random_matrix = generate_matrix(50, 0.4)  # This generates a 50x50 matrix, where p = 0.4
    animate_matrix(random_matrix)  # To visualise the new randomly generated matrix
    percolate_diagonally(random_matrix)  # This will simulate unlimited water starting in the corner
    animate_diagonal_matrix(random_matrix, delay=5)  # NOTE: This is a different animate function

    # Test percolation
    outcome = test_diagonal_percolation(random_matrix)  # This tests if the water reached the bottom
    print("Did the water reach the bottom?", outcome)  # Returns a boolean (True/False)

    # Run Simulation - uses percolate_onedrop by default
    success_rate = run_sim(50, 0.3, 1_000, percolate_diagonally, True)
    # This will return the proportion of randomly generated matrices, where water reached the bottom

    # Simulate a variety of p values
    find_critical_p(50, 1_000, 0.01, algorithm=percolate, graph=True)
    sim_vary_p(50, 1_000, 0.01, algorithm=percolate, graph=True)
    # This calculates and returns the critical value of p for a matrix of size N

    # Simulate a variety of N values
    matrix_sizes = [size for size in range(2, 50)]
    sim_vary_N(matrix_sizes, 1_000, 0.01, algorithm=percolate_onedrop)
    # This outputs a graph of the critical p value for different sizes of matrices

    # Some of these calculations can take a while, especially on big numbers of iterations
    # To time how long a simulation takes, you can use perf_counter() as shown:

    print("Simulation started at {}".format(datetime.datetime.now()))
    start_time = time.perf_counter()

    # Function goes here

    finish_time = time.perf_counter()
    time_taken = finish_time - start_time
    print("Time taken: {} seconds".format(time_taken))

    tk.mainloop()