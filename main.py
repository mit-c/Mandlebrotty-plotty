import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("seaborn-pastel")

num_its = 50
width = 300
height = 300
fig = plt.figure()
ax = plt.axes()


def mandlebrot(num_its, width, height):
    out_grid = [[0 for i in range(width)] for j in range(height)]

    real_range = [-2 + 3 * i / width for i in range(width)]
    imag_range = [-1 + 2 * j / height for j in range(height)]

    for i in range(width):
        for j in range(height):
            c = complex(real_range[i], imag_range[j])
            new_z = complex(0, 0)
            for k in range(num_its):
                new_z = new_z * new_z + c
                if abs(new_z) > 4:
                    break
                out_grid[j][i] = k / num_its
    return out_grid


def animate(i, width, height):
    ax.clear()
    ax.tick_params(axis="both",
                   which="both",
                   bottom=False,
                   top=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)


    out_grid = mandlebrot(num_its=i,
                          width=width,
                          height=height)
    img = ax.imshow(out_grid)
    return [img]


mandlebrot_ani = FuncAnimation(fig, animate, frames=30, interval=120, blit=True, fargs=(width, height))
mandlebrot_ani.save("mandlebrot_plot.gif", writer="imagemagick")
