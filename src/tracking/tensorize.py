import numpy as np

# JSON -> Tensor (3D array: Moment, Players, (x,y) Coords)
# Tensor with all players and ball
def event_to_tensor(event, include_ball=False, max_frames=None):
    """
    Convert an event to a NumPy tensor of shape (T, N, 2),
    where:
        T = number of frames (or max_frames)
        N = number of players (+1 if include_ball)
        2 = x, y coordinates
    """
    frames = event.get("frames", [])

    if not frames:
        # No frames in this event
        return np.empty((0, 0, 0))

    if max_frames is not None:
        frames = frames[:max_frames]

    # Determine number of points (players + optional ball)
    num_players = len(frames[0].get("players", []))
    if num_players == 0:
        return np.empty((0, 0, 0))

    num_points = num_players + 1 if include_ball else num_players
    T = len(frames)

    traj = np.zeros((T, num_points, 2))

    for t, frame in enumerate(frames):
        # players
        for i, p in enumerate(frame.get("players", [])):
            traj[t, i, 0] = p.get("x", 0.0)
            traj[t, i, 1] = p.get("y", 0.0)

        # ball
        if include_ball:
            ball = frame.get("ball")
            if ball:
                traj[t, -1, 0] = ball.get("x", 0.0)
                traj[t, -1, 1] = ball.get("y", 0.0)

    return traj

# Tensor with only offensive players and ball
def event_to_tensor_offense(event, include_ball=False, max_frames=None):
    frames = event.get("frames", [])
    if max_frames is not None:
        frames = frames[:max_frames]

    # Determine offensive players
    possession_team_id = event.get("possession_team_id")
    offensive_players = [
        p for p in frames[0]["players"] if p["teamid"] == possession_team_id
    ]

    T = len(frames)
    N = len(offensive_players) + (1 if include_ball else 0)
    traj = np.zeros((T, N, 2))

    for t, frame in enumerate(frames):
        # offense
        for i, p in enumerate(frame["players"]):
            if p["teamid"] == possession_team_id:
                matching_indices = [j for j, op in enumerate(offensive_players) if op["playerid"] == p["playerid"]]
                if matching_indices:  # only update if player found
                    idx = matching_indices[0]
                    traj[t, idx, 0] = p["x"]
                    traj[t, idx, 1] = p["y"]

        # ball
        if include_ball:
            ball = frame.get("ball")
            if ball:
                traj[t, -1, 0] = ball["x"]
                traj[t, -1, 1] = ball["y"]

    return traj


def split_offense_defense(event, traj):
    """
    Split players in a tensor into offensive and defensive sets
    based on possession_team_id.
    
    traj: T x N x 2 tensor from event_to_tensor(include_ball=False)
    Returns:
        offense: T x n_offense x 2
        defense: T x n_defense x 2
    """
    frames = event["frames"]
    possession_team = event.get("possession_team_id")
    if possession_team is None or traj.shape[1] == 0:
        return np.empty((0, 0, 2)), np.empty((0, 0, 2))
    
    # Indices for offensive vs defensive players
    offense_idx = [i for i, p in enumerate(frames[0]["players"]) if p["teamid"] == possession_team]
    defense_idx = [i for i, p in enumerate(frames[0]["players"]) if p["teamid"] != possession_team]
    
    # Slice the tensor
    offense = traj[:, offense_idx, :]
    defense = traj[:, defense_idx, :]
    
    return offense, defense