import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

def plot_frame(frame, team_colors=None):
    """
    Plot a single NBA frame with players split by team and the ball.
    
    Parameters
    ----------
    frame : dict
        Single frame dictionary with keys: 'ball', 'players', 'frame_id', etc.
    team_colors : dict, optional
        Mapping from teamid to color. Example: {1610612739: 'blue', 1610612744: 'red'}
    """
    
    if team_colors is None:
        team_colors = {}
    
    plt.figure(figsize=(15, 7))
    
    # Draw court boundaries (simplified rectangle)
    plt.plot([0, 50], [0, 0], color='black')   # baseline
    plt.plot([0, 50], [94, 94], color='black') # opposite baseline
    plt.plot([0, 0], [0, 94], color='black')   # sideline
    plt.plot([50, 50], [0, 94], color='black') # opposite sideline
    
    # Draw the ball
    ball = frame["ball"]
    plt.scatter(ball["x"], ball["y"], c='orange', s=200, marker='o', label='Ball', edgecolors='black')
    
    # Plot players
    for player in frame["players"]:
        x, y = player["x"], player["y"]
        teamid = player["teamid"]
        color = team_colors.get(teamid, 'green')  # default green if teamid not in dict
        plt.scatter(x, y, c=color, s=150, label=f'Team {teamid}' if f'Team {teamid}' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.text(x+0.5, y+0.5, str(player["playerid"]), fontsize=9, color=color)
    
    plt.xlim(0, 50)
    plt.ylim(0, 94)
    plt.xlabel("Court X (ft)")
    plt.ylabel("Court Y (ft)")

    # convert game clock to MM:SS format
    minutes = int(frame['game_clock'] // 60)
    seconds = int(frame['game_clock'] % 60)

    # get shot clock if available
    shot_clock = (frame['shot_clock'])
    shot_str = f"{shot_clock:.1f}s" if shot_clock is not None else "-"

    plt.title(f"Frame {frame['frame_id']} - Game Clock: {minutes:02d}:{seconds:02d} | Shot Clock: {shot_str}")
    plt.legend()
    plt.show()


def draw_half_court(ax=None, color="black", lw=2, outer_lines=False, zorder=2):
    if ax is None:
        ax = plt.gca()

    Y_SHIFT = 47.5  # shift old court coords up so baseline becomes y=0

    # Hoop & backboard
    hoop = Circle((0, 0 + Y_SHIFT), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5 + Y_SHIFT), 60, -1, linewidth=lw, color=color)

    # Paint
    outer_box = Rectangle((-80, -47.5 + Y_SHIFT), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5 + Y_SHIFT), 120, 190, linewidth=lw, color=color, fill=False)

    # Free throw arcs
    top_free_throw = Arc((0, 142.5 + Y_SHIFT), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color)
    bottom_free_throw = Arc((0, 142.5 + Y_SHIFT), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle="dashed")

    # Restricted arc
    restricted = Arc((0, 0 + Y_SHIFT), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Corner 3 lines (14ft = 168 inches tall)
    corner_three_a = Rectangle((-220, -47.5 + Y_SHIFT), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5 + Y_SHIFT), 0, 140, linewidth=lw, color=color)

    # 3pt arc (same as your original; works visually)
    three_arc = Arc((0, 0 + Y_SHIFT), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center court arcs
    center_outer_arc = Arc((0, 422.5 + Y_SHIFT), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5 + Y_SHIFT), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

    court_elements = [
        hoop, backboard, outer_box, inner_box, top_free_throw, bottom_free_throw,
        restricted, corner_three_a, corner_three_b, three_arc,
        center_outer_arc, center_inner_arc
    ]

    if outer_lines:
        outer = Rectangle((-250, 0), 500, 470, linewidth=lw, color=color, fill=False)  # baseline now y=0
        court_elements.append(outer)

    for e in court_elements:
        e.set_zorder(zorder)
        ax.add_patch(e)

    return ax
