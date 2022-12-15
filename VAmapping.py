
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# convert values from range [0,1] to [-1,1]
def convert_range(value):
  return value * 2 - 1

# Function to associate colors with each quadrant of the Cartesian plane
def assign_colors_to_quadrants(x, y):
  # Use the colormap 'viridis' to map the coordinates to colors
  cmap = cm.get_cmap('viridis')

  # Create a list of colors for each quadrant
  colors = []

  # Iterate over the coordinates
  for i in range(len(x)):
    # Quadrant I: x > 0, y > 0
    if x[i] > 0 and y[i] > 0:
      color = cmap(0)
    # Quadrant II: x < 0, y > 0
    elif x[i] < 0 and y[i] > 0:
      color = cmap(0.25)
    # Quadrant III: x < 0, y < 0
    elif x[i] < 0 and y[i] < 0:
      color = cmap(0.5)
    # Quadrant IV: x > 0, y < 0
    elif x[i] > 0 and y[i] < 0:
      color = cmap(0.75)
    # Interpolate colors at the border between the quadrants
    else:
      # Get the colors for the two adjacent quadrants
      if x[i] == 0 and y[i] > 0:
        c1 = cmap(0)
        c2 = cmap(0.25)
      elif x[i] < 0 and y[i] == 0:
        c1 = cmap(0.25)
        c2 = cmap(0.5)
      elif x[i] == 0 and y[i] < 0:
        c1 = cmap(0.5)
        c2 = cmap(0.75)
      elif x[i] > 0 and y[i] == 0:
        c1 = cmap(0.75)
        c2 = cmap(0)
      # Interpolate the colors
      color = (c1 + c2) / 2
      # Interpolate colors in the regions near the borders
      if abs(x[i]) < 0.25 and y[i] > 0:
        color = (color + c1) / 2
      elif x[i] < 0 and abs(y[i]) < 0.25:
        color = (color + c2) / 2
      elif abs(x[i]) < 0.25 and y[i] < 0:
        color = (color + c3) / 2
      elif x[i] > 0 and abs(y[i]) < 0.25:
        color = (color + c4) / 2

    # Add the color to the list
    colors.append(color)

  return colors


# Function to plot a list of RGBA colors
def plot_colors(colors):
  # Create a figure and axes
  fig, ax = plt.subplots()

  # Set the axes limits to the range [0, 1]
  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])

  # Plot each color as a rectangle with the RGBA values as its facecolor
  for i in range(len(colors)):
    ax.add_patch(plt.Rectangle((i/len(colors), 0), 1/len(colors), 1, facecolor=colors[i]))

  plt.figure()
  plt.show()
 