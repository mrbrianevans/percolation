import numpy as np
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
    print(blank)
    return blank
    # This checks if the water reached the bottom or not
    for j in range(N + 2):
        if blank[N - 1, j] == 1:
            return True
    return False


def percolate_onedrop(matrix: np.ndarray):
    """
    Percolate water through rocks and sand for only one drop, starting in the middle on the top row
    Args:
        matrix: array of sand and stone (where sand is porus and stone is not)

    Returns: matrix showing where the water flowed to

    """
    N = matrix.shape[0]
    waterdrop_location = (0, int(N / 2))
    blank = np.zeros(matrix.shape, dtype=int)
    blank[waterdrop_location] = 1
    print(blank)


# NEXT STEP: animate it in Tkinter, write simulation tests for a thousand cases

if __name__ == "__main__":
    matrix1 = generate_matrix(100, 0.55)
    print(matrix1)
    # animate_matrix(matrix1, 400)
    percolate_onedrop(matrix1)
    animate_matrix(percolate(matrix1), 800)
