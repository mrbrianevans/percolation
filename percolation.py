import datetime
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
    new_matrix = np.zeros(shape=(N, N + 2), dtype=int)
    blank = np.zeros(shape=(N, N + 2), dtype=int)

    # this creates 'walls' on either side of the matrix, to prevent wrapping around
    for i in range(N):
        new_matrix[i, 0] = 2
        new_matrix[i, N + 1] = 2

    # This copies the matrix into the one with walls
    for i in range(N):
        for j in range(N):
            new_matrix[i, j + 1] = matrix[i, j]
    print(new_matrix)
    matrix = new_matrix

    # This gets the first row, from the matrix. The starting blocks of the water
    for i in range(N):
        if matrix[0, i + 1] == 2:
            blank[0, i + 1] = 1

    # This is the percolation algorithm. It loops through 'blank' matrix and adds in 1 if water can flow
    for i in range(N):
        for j in range(N):
            if matrix[i, j + 1] == 2:
                if blank[i - 1, j + 2] == 1 or blank[i - 1, j + 1] == 1 \
                        or blank[i - 1, j] == 1 or blank[i, j] == 1 or i == 0:
                    blank[i, j + 1] = 1
                else:
                    blank[i, j + 1] = 2
    return blank


def percolate_onedrop(matrix: np.ndarray):
    """
    Percolate water through rocks and sand for only one drop, starting in the middle on the top row
    Args:
        matrix: array of sand and stone (where sand is porus and stone is not)

    Returns: matrix showing where the water flowed to

    """
    N = matrix.shape[0]

    # This adds in walls of rock on either side, to stop the drop of water flowing out of the grid
    walled_matrix = np.zeros(shape=(N, N+2), dtype=int)
    for i in range(N):
        for j in range(N):
            walled_matrix[i, j+1] = matrix[i, j]
    matrix = walled_matrix

    waterdrop_h = int(N / 2)
    waterdrop_v = 0
    matrix[waterdrop_v, waterdrop_h] = 1
    possible_to_flow = True
    while (possible_to_flow):
        if N == waterdrop_v + 2:  # checks if the water has reached the bottom
            possible_to_flow = False
        if matrix[waterdrop_v + 1, waterdrop_h] == 2:  # checks the position directly below the drop
            waterdrop_v += 1
            matrix[waterdrop_v, waterdrop_h] = 1
        elif matrix[waterdrop_v + 1, waterdrop_h - 1] == 2:  # checks the position down-left of drop
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

    # This removes the walls on either side
    walled_matrix = np.zeros(shape=(N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            walled_matrix[i, j] = matrix[i, j+1]
    matrix=walled_matrix

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
        percolation = percolate_onedrop(matrix)
        if test_percolation(percolation):
            number_of_times_bottom_reached += 1

    # This prints for each value of p if you run sim_vary_p. Comment out to avoid if necessary
    print("\n\nMatrix size:", size, "x", size, "with p =", p)
    print("In", iterations, "iterations, there were ", number_of_times_bottom_reached, "successes")
    print("This is an average rate of", number_of_times_bottom_reached / iterations)
    return number_of_times_bottom_reached / iterations


def sim_vary_p(size: int, iterations: int, steps: float) -> None:
    """
    Run a simulation across different values of P, and graph the results with matplotlib
    Args:
        size: size of the square matrix (N in NxN)
        iterations: number of realisations for each value of p
        steps: the increase of p after each sim. (smaller values give more accurate graph results)

    Returns: nothing. just outputs a graph with matplotlib

    """
    print("Simulation started at", datetime.datetime.now())
    x_values = np.arange(0, 1, steps)
    y_values = []
    hit_zero = False
    for p in x_values:
        if not hit_zero:
            y_values.append(run_sim(size, p, iterations))
        else:
            y_values.append(0)
        if y_values[-1] == 0:  # once zero is reached, it will stop simulating and only append zeros
            hit_zero = True  # this will save useless computation
    print("Simulation finished at", datetime.datetime.now())
    print(x_values)
    print(y_values)
    plt.plot(x_values, y_values, color="k")

    # This finds the critical value of P. It could be improved by interpolating the data
    closest = 1
    Pc, Py = 0, 0
    for y in range(len(y_values)):
        if abs(0.5-y_values[y]) < closest:
            closest = abs(0.5-y_values[y])
            Pc = x_values[y]
            Py = y_values[y]
    plt.plot([Pc, Pc], [1, 0], label="Critical value of P", color="r")
    plt.plot([0, 1], [Py, Py], color="r")
    plt.plot([Pc], [Py], "o", color="orange")
    plt.text(Pc-0.1, 1, "Critical value of P="+str(Pc))
    plt.text(0.75, Py+0.005, "" + str(round(Py*100)) + "% success rate")
    plt.ylabel("Average probability of water reaching the bottom")
    plt.xlabel("p")
    plt.title("Percolation simulation: N=" + str(size) + ", nrep=" + str(iterations) + ", P steps=" + str(steps))

    plt.show()


if __name__ == "__main__":

    start_time = time.perf_counter()
    # To run many simulations and graph the results, use this function:
    sim_vary_p(size=10, iterations=4000, steps=0.005)
    sim_vary_p(size=50, iterations=4000, steps=0.005)
    sim_vary_p(size=100, iterations=4000, steps=0.005)
    sim_vary_p(size=200, iterations=4000, steps=0.005)
    sim_vary_p(size=400, iterations=4000, steps=0.005)

    # This just times how long the simulation took (can easily be north of 1 hour)
    finish_time = time.perf_counter()
    time_taken = finish_time - start_time
    print("Time taken:", time_taken, "seconds")

    # To simulate just one percolation and animate it with graphics, use this syntax
    matrix1 = generate_matrix(N=10, p=0.25)  # N - size of matrix, p - proportion of rock
    perc = percolate_onedrop(matrix=matrix1)
    animate_matrix(matrix=perc, size=800)  # size is number of pixels to draw the grid
