"""
Author: Patricia Stoehr

Class Trajectory determines and visualizes the trajectories.
"""

import gymnasium as gym
import numpy as np
from gym_minigrid.minigrid import *
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import (TextArea, OffsetImage, AnnotationBbox)
import matplotlib.lines as mlines
from PIL import Image


def set_background_image_without_obst(plot, max, alpha=0.35):
    """
    The method creates an empty BabyAI grid image. 
    This image is the background of the trajectory plot.

    :param plot: the plot in which the background image appears.
    :param max: a tuple specifying maximum width and maximum height of grid.
    :param alpha: alpha value specifying the transparency; alpha = 0 means fully transparent; alpha = 1 is fully opaque.
    """
    grid = Grid(max[0], max[1])
    img = grid.render(70)
    plot.imshow(img, extent=[1, max[0] + 1, 1, max[1] + 1], alpha=alpha)


def set_background_image(image, plot, zoom, xy):
    """
    The method shows the initial image of an episode. 
    This image is the background of the glyphs and the trajectory plot.

    :param image:
    :param plot: the plot in which the background image appears.
    :param zoom: zooming factor of image.
    :param xy: x- and y-coordinate of the point that is annotated with the AnnotationBbox.
    """
    img = Image.fromarray(np.array(image))
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy=xy, annotation_clip=False, zorder=1, pad=0, frameon=False)
    plot.add_artist(ab)


def set_legend(type, plot, uncertainty_color):
    """
    The method places a legend on the left side of the trajectory plot. 
    The legend contains the symbols that encode the agent's path, the optimal path, 
    the agent's initial, and end position, and the target position.

    :param plot: the plot to which the legend belongs.
    :param uncertainty_color: the color for the current step; this color encodes the uncertainty value.
    """
    # create items in legend
    patch_agent = patches.Patch(color="tab:blue", label="agent path")
    patch_optimal = patches.Patch(color="yellowgreen", label="optimal path")
    patch_step = patches.Patch(color=uncertainty_color, label="uncertainty")
    red_triangle = mlines.Line2D([], [], color="red", marker=">", markersize=5, label="initial agent position")
    blue_triangle = mlines.Line2D([], [], color="blue", marker=">", markersize=5, label="agent direction")
    green_triangle = mlines.Line2D([], [], color="green", marker=">", markersize=5, label="optimal direction")

    legend_elements = [red_triangle, patch_step]

    # legend contains only the elements that are currently visible in the trajectory visualization
    if type.__eq__("both paths") or type.__eq__("agent path"):
        legend_elements.append(blue_triangle)
        legend_elements.append(patch_agent)

    if type.__eq__("both paths") or type.__eq__("optimal path"):
        legend_elements.append(green_triangle)
        legend_elements.append(patch_optimal)

    plot.legend(handles=legend_elements, bbox_to_anchor=(1, 0), loc="lyower left", fancybox=True, frameon=True)


def set_mission(plot, mission, maximum):
    """
    The method places the current mission into an AnnotationBbox. 
    The box is located below the trajectory plot.

    :param plot: the plot to which the legend belongs.
    :param mission: the agent's current mission.
    :param maximum: maximum x- and y-coordinate of the plot.
    """
    factor = 0.25 if maximum[1] == 20 else 0.45  # factor to place the mission box properly
    offsetbox = TextArea("Mission: \n" + mission)
    ab = AnnotationBbox(offsetbox, xy=(maximum[0] / 1.7, maximum[1] + (maximum[1] * factor)), annotation_clip=False,
                        frameon=False)
    plot.add_artist(ab)


def get_grid_size(level):
    """
    This method determines the grid size. The size varies depending on the level.

    :param level: the name of the BabyAI level, e.g., BabyAI-GoTo-v0.
    :return: a tuple (width, height); it can have values (6, 6) or (20, 20).
    """
    # split level name
    # for example the string BabyAI-GoTo-v0 becomes the string GoTo
    level_name = level.split("-")[1]

    if level_name in ["GoToObj", "GoToRedBall", "GoToRedBallGrey", "GoToLocal", "PutNextLocal", "PickUpLoc"]:
        size = (6, 6)
    elif level_name in ["GoToObjMaze", "GoTo", "Pickup", "UnblockPickup", "Open", "Unlock", "PutNext", "Synth",
                        "SynthLoc", "GoToSeq", "SynthSeq", "GoToImpUnlock", "BossLevel"]:
        size = (20, 20)
    elif level_name in ["MiniBossLevel"]:
        size = (7, 7)
    else:
        print("Level name does not exist.")
        return

    return size


def collect_coordinates(offset, env_name, seed, actions):
    """
    The method sets up an environment with a given seed. It determines the mission.
    It repeats the specified actions. After each action, it collects the agent's position. 

    :param offset: an offset to get coordinates within a grid field instead of coordinates on the grid field's border.
    :param env_name: BabyAI level name, e.g., BabyAI-GoTo-v0
    :param seed: the seed to set up the environment.
    :param actions: a sequence of actions; after each action, the method stores the agent's coordinates.
    :return: a tuple that stores the mission, a list of visited x-coordinates, visited y-coordinates, and the directions.
    """
    x_coord, y_coord, directions = ([] for _ in range(3))

    # set up environement
    env = gym.make(env_name)
    #gym.Env.seed(env, seed)
    env.seed(seed=seed)
    env.reset()

    # initial image
    image = env.render(mode="rgb_array")

    # add initial agent position
    x_coord.append(env.agent_pos[0] + offset)
    y_coord.append(env.agent_pos[1] + offset)
    directions.append(env.agent_dir)

    # redo the actions and collect the agent coordinates and the direction after each action
    for action in actions:
        _, _, _, _ = env.step(action)
        x_coord.append(env.agent_pos[0] + offset)
        y_coord.append(env.agent_pos[1] + offset)
        directions.append(env.agent_dir)

    return env.mission, x_coord, y_coord, directions, image


def map_direction_to_marker(directions):
    """
    The method iterates through a list of numbers. The numbers encode actions
    (0 is right, 1 is down, 2 is left, 3 is up). It adds a direction marker that corresponds to the number to a list.
    The direction marker indicates which direction the agent faces.

    :param directions: a list of numbers that encode directions.
    :return: a list of direction marker symbols. 
    """
    marker = []
    for dir in directions:
        if dir == 0:
            marker.append(">")
        elif dir == 1:
            marker.append("v")
        elif dir == 2:
            marker.append("<")
        elif dir == 3:
            marker.append("^")
        else:
            print("number not in BabyAI direction numbers")
    return marker


def place_direction_marker(plot, x_coord, y_coord, step_idx, directions, label, color, size):
    """
    The method visualizes the agent's direction. It places a direction marker in the grid.
    The direction marker indicates which direction the agent faces.

    :param plot: the plot where to place the marker.
    :param x_coord: a list of x coordinates that belong to a trajectory/ path. 
    :param y_coord: a list of y coordinates that belong to a trajectory/ path.  
    :param step_idx: an index of step in list of steps.
    :param directions: a sequence of directions obtained while tracing a path.
    :param label: the label for the direction marker.
    :param color: the direction marker's color.
    :param size: the marker size.
    """
    marker = map_direction_to_marker(directions)
    size = 200 if size[0] == 6 else 10  # marker size depends on size of grid
    plot.scatter(x_coord[step_idx], y_coord[step_idx], label=label, c=color, marker=marker[step_idx], zorder=5.0,
                 s=size)


def place_arrow(plot, x, y, direction, agent_orientation, color, offset):
    """
    The method places an arrow in a grid. The arrow indicates in which direction the agent turns.

    :param plot: a plot where to place the data.
    :param x: the agent's current x-coordinate.
    :param y: the agent's current y-coordinate.
    :param direction: the direction into which the agent turns.
    :param agent_orientation: the direction that the agent currently faces.
    :param color: the color for current step; color encodes the uncertainty.
    :param offset: offset coordinates used for placing the arrow properly. 
    """
    width = 1.9
    x1 = x - offset
    y1 = y - offset
    x2 = x + (1 - offset)
    y2 = y + (1 - offset)

    # turn left
    if direction == 0:
        if agent_orientation == 0:
            plot.annotate("", xy=(x2, y1), xytext=(x1, y2),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=0, angleB=90"))

        elif agent_orientation == 1:
            plot.annotate("", xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=90, angleB=0"))

        elif agent_orientation == 2:
            plot.annotate("", xy=(x1, y2), xytext=(x2, y1),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=0, angleB=90"))

        elif agent_orientation == 3:
            plot.annotate("", xy=(x1, y1), xytext=(x2, y2),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=90, angleB=0"))

    # turn right
    if direction == 1:
        if agent_orientation == 0 or agent_orientation == 3:
            plot.annotate("", xy=(x2, y2), xytext=(x1, y1),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=0, angleB=90"))

        elif agent_orientation == 1:
            plot.annotate("", xy=(x1, y2), xytext=(x2, y1),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=90, angleB=0"))

        elif agent_orientation == 2:
            plot.annotate("", xy=(x1, y1), xytext=(x2, y2),
                          arrowprops=dict(arrowstyle="->", color=color, lw=width,
                                          connectionstyle="angle3, angleA=0, angleB=90"))


def draw_line_collection(plot, x_coord, y_coord, step_idx, path_color, step_color):
    """
    The method creates a line collection that shows a trajectory/ path. Each single line of the collection
    represents a step of the path. It enables to use different colors for different steps.

    :param plot: the plot where to place the line collection.
    :param x_coord: a list of x-coordinates that belong to a trajectory/ path. 
    :param y_coord: a list of y-coordinates that belong to a trajectory/ path.  
    :param step_idx: an index of step in list of steps.
    :param path_color: the color for line that encodes the predicted or the correct path in a grid.
    :param step_color: the color for the current step; color encodes the uncertainty.          
    """
    # draw path
    points = np.array([x_coord, y_coord]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # color current step in another color 
    # in the optimal path, this color encodes the uncertainty value
    colors = [path_color for _ in range(len(segments))]
    colors[step_idx] = step_color

    lc = LineCollection(segments, lw=3.5, color=colors)
    lc.set_array(y_coord)
    plot.add_collection(lc)


def visualize_trajectory(env_name, plot, actions, actions_true, seed, glyph_no,
                         step_wise=False, step_idx=None, color=None):
    """
    This method visualizes the agent's path and the optimal path. 

    :param env_name: BabyAI level name.
    :param plot: an empty plot where to place the data.
    :param actions: an action sequence that the agent tries until it reaches the maximum number of steps and fails.
    :param actions_true: the true actions specified in demo.
    :param seed: the seed of an episode.
    :param glyph_no: identifier.
    :param step_wise: if true, the method creates a seperate line for each step.
                      if false, single line connects all trajectory coordinates.
    :param step_idx: index of step in a list of steps.
    :param color: color for current step that encodes the uncertainty. 
    :return: a plot visualizing the grid and the agent's path.
    """
    plot.clear()
    max = get_grid_size(env_name)
    plot.set_xlim(0, max[0] + 2, auto=False)
    plot.set_ylim(max[1] + 2, 0, auto=False)
    plot.axis("off")
    plot.set_box_aspect(1)

    # offset is used to plot the lines within a grid field
    # otherwise the lines are plotted on grid lines
    # new offset is used to avoid overplotting if agent's trajectory and optimal trajectory are identical
    offset, offset_true = 0.4, 0.6

    (mission, x_coord, y_coord, directions, image) = collect_coordinates(offset, env_name, seed, actions)
    (_, x_coord_true, y_coord_true, directions_true, _) = collect_coordinates(offset_true, env_name, seed,
                                                                                  actions_true)

    if step_wise and (step_idx is not None) and (color is not None):
        #zoom = 0.78 if max[0] == 6 else 0.29 (FOR 20)
        zoom = 6 / (max[0]+1)

        visualize_as_plot(plot, glyph_no, actions, actions_true, x_coord, y_coord, x_coord_true, y_coord_true,
                          step_idx,
                          directions, directions_true, max, mission, color)
    else:
        zoom = 6 / (max[0]+1)
        visualize_as_glyph(plot, glyph_no, x_coord, y_coord, x_coord_true, y_coord_true, max)

    xy = ((max[0]+2)/2, (max[1]+2)/2)
    set_background_image(image, plot, zoom, xy)

    return plot


def visualize_as_plot(plot, step_no, actions, actions_true, x_coord, y_coord, x_coord_true, y_coord_true,
                      step_idx, directions, directions_true, max, mission, color, type="both paths"):
    """
    This method visualizes the trajectory in a stand-alone plot.  

    :param plot: a plot where to place the data.
    :param step_no: an identifier.
    :param actions: an action sequence that the agent tries.
    :param actions_true: the true actions specified in demo.
    :param x_coord: a list of x-coordinates that belong to a predicted trajectory/ path. 
    :param y_coord: a list of y-coordinates that belong to a predicted trajectory/ path. 
    :param x_coord_true: a list of x-coordinates that belong to the correct trajectory/ path specified in training data. 
    :param y_coord_true: a list of y-coordinates that belong to correct trajectory/ path specified in training data.  
    :param step_idx: index of step in a list of steps.
    :param directions: a sequence of directions obtained while tracing a predicted path.
    :param directions_true: a sequence of directions obtained while tracing the correct path.
    :param max: maximum x- and y-coordinate of the plot.
    :param mission: the agent's current mission.
    :param color: the color for current step.
    """
    plot.set_title("Step " + str(step_no), loc="center", fontweight="bold")
    plot.set_xlim(0, max[0] + 2, auto=False)
    plot.set_ylim(max[1] + 2, 0, auto=False)

    # visualize predicted path with direction marker that shows the current agent position
    if type.__eq__("both paths") or type.__eq__("agent path"):
        draw_line_collection(plot, x_coord, y_coord, step_idx, "tab:blue", "tab:blue")
        place_direction_marker(plot, x_coord, y_coord, step_idx, directions, "direction", "blue", max)

        # place arrow if agent turns left (0) or right (1)
        if actions[step_idx] in [0, 1]:
            place_arrow(plot, x_coord[step_idx], y_coord[step_idx], actions[step_idx], directions[step_idx],
                        "tab:blue", 0.4)

    # visualize optimal path
    if type.__eq__("both paths") or type.__eq__("optimal path"):
        draw_line_collection(plot, x_coord_true, y_coord_true, step_idx, "yellowgreen", color)
        place_direction_marker(plot, x_coord_true, y_coord_true, step_idx, directions_true, "direction true",
                               "yellowgreen", max)

        # place arrow if agent turns left (0) or right (1)
        if actions_true[step_idx] in [0, 1]:
            place_arrow(plot, x_coord_true[step_idx], y_coord_true[step_idx], actions_true[step_idx],
                        directions_true[step_idx], color, 0.6)

    set_legend(plot, color)
    set_mission(plot, mission, max)


def visualize_as_glyph(plot, glyph_no, x_coord, y_coord, x_coord_true, y_coord_true, max):
    """
    This method creates a trajectory plot. This plot is shown as a glyph.

    :param plot: a plot where to place the data.
    :param glyph_no: an identifier.
    :param x_coord: a list of x-coordinates that belong to a predicted trajectory/ path. 
    :param y_coord: a list of y-coordinates that belong to a predicted trajectory/ path. 
    :param x_coord_true: a list of x-coordinates that belong to the correct trajectory/ path specified in training data. 
    :param y_coord_true: a list of y-coordinates that belong to correct trajectory/ path specified in training data.  
    :param max: maximum x- and y-coordinate of the plot.
    """
    plot.set_xlim(0, max[0] + 2, auto=False)
    plot.set_ylim(max[1] + 2, 0, auto=False)
    width = 7.5 if max[0] == 6 else 4

    # plot agent's path
    plot.plot(x_coord, y_coord, lw=width, c="lightseagreen")

    # plot optimal path
    plot.plot(x_coord_true, y_coord_true, lw=width, c="yellowgreen")


def visualize_replay_trajectory(trajectory_plot, env_name, seed, actions, actions_true):
    """
    The method redraws the agent's trajectory that the user watched in the replay.
    Additionally, it draws the optimal trajectory in the plot.

    :param trajectory_plot: a plot in which the lines are drawn.
    :param env_name: BabyAI level name.
    :param seed: the seed of the replay episode.
    :param actions: an action sequence that the agent tries.
    :param actions_true: the true actions specified in demo.
    """
    trajectory_plot.set_title("Optimal vs Agent Trajectory")

    max = get_grid_size(env_name)
    trajectory_plot.set_xlim(0, max[0] + 2, auto=False)
    trajectory_plot.set_ylim(max[1] + 2, 0, auto=False)
    trajectory_plot.axis("off")
    trajectory_plot.set_box_aspect(1)

    offset, offset_true = 0.4, 0.5

    (_, x_coord, y_coord, _, image) = collect_coordinates(offset, env_name, seed, actions)
    (_, x_coord_true, y_coord_true, directions_true, _) = collect_coordinates(offset_true, env_name, seed,
                                                                           actions_true)

    markers_true = map_direction_to_marker(directions_true)

    # draw agent and optimal path
    draw_line_collection(trajectory_plot, x_coord, y_coord, 0, "tab:blue", "tab:blue")
    draw_line_collection(trajectory_plot, x_coord_true, y_coord_true, 0, "yellowgreen", "yellowgreen")

    #  mark agent's initial position
    size = 400 if max[0] == 6 else 50
    trajectory_plot.scatter(x_coord_true[0], y_coord_true[0], label="initial position", c="blue",
                            marker=markers_true[0], zorder=5.0, s=size)

    # legend
    blue_triangle = mlines.Line2D([], [], color="blue", marker=">", markersize=5, label="agent's initial\nposition")
    red_triangle = mlines.Line2D([], [], color="red", marker=">", markersize=5, label="agent's final\nposition")
    patch_agent = patches.Patch(color="tab:blue", label="agent path")
    patch_optimal = patches.Patch(color="yellowgreen", label="optimal path")
    trajectory_plot.legend(handles=[blue_triangle, red_triangle, patch_agent, patch_optimal], bbox_to_anchor=(1, 0),
                           loc="lower left", fancybox=True, frameon=True)


def find_obstacles(env):
    """
    The method iterates through the grid that builds the agent's environment.
    It creates a dictionary for each object in the grid.
    The dictonary stores the object type, color, coordinates, and a marker.
    All objects are stored in a list.

    :param env: the agent's current environment.
    """
    obstacles = []

    for j in range(env.grid.height):
        for i in range(env.grid.width):
            o = env.grid.get(i, j)

            if o is not None and o.type != "wall":
                if o.type == "ball":
                    marker = "o"
                elif o.type == "box":
                    marker = "s"
                elif o.type == "key":
                    marker = "1"
                elif o.type == "door":
                    marker = "x"

                obstacle_dict = {"type": o.type,
                                 "marker": marker,
                                 "color": o.color,
                                 "x": i + 0.5,
                                 "y": j + 0.5
                                 }
                obstacles.append(obstacle_dict)


def generate_thumbnail(env_name, seed, actions):
    """
    Generate a thumbnail trajectory glyph for a given environment and action sequence.
    """
    import matplotlib.pyplot as plt
    # Create a new figure
    fig = plt.figure(figsize=(3, 3), dpi=300)

    # Create a single subplot
    ax = fig.add_subplot(1, 1, 1)
    # Remove the axes
    ax.axis('off')

    # Visualize the trajectory
    print("SEED", seed)
    visualize_trajectory(env_name, ax, actions, actions, seed, 0, step_wise=False)

    # Save the figure as a RGB array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    # Close the figure
    plt.close(fig)

    # Reshape the data into a 3D array (width, height, channels)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img
