import numpy as np
# For data with possession info
def identify_possession(row):
    """
    Simplified NBA possession logic
    """
    msg = int(row["EVENTMSGTYPE"])

    if msg in {1, 2, 3, 4, 5}:  # shots, rebounds, turnovers
        return row["PLAYER1_TEAM_ID"]

    if "OFF.FOUL" in str(row["HOMEDESCRIPTION"]) or "OFF.FOUL" in str(row["VISITORDESCRIPTION"]):
        return row["PLAYER1_TEAM_ID"]

    if msg == 6:  # foul
        return row["PLAYER2_TEAM_ID"]

    return None

# For data without possession info
def assign_event_possession(event):
    """
    Assign possession_team_id for an event based on the first frame.
    """
    if not event.get("frames"):
        event["possession_team_id"] = None
        return

    first_frame = event["frames"][0]
    ball = first_frame.get("ball")
    players = first_frame.get("players", [])

    if ball is None or not players:
        event["possession_team_id"] = None
        return

    ball_pos = np.array([ball["x"], ball["y"]])

    # Find the player closest to the ball
    min_dist = float("inf")
    possession_team = None
    for p in players:
        player_pos = np.array([p["x"], p["y"]])
        dist = np.linalg.norm(ball_pos - player_pos)
        if dist < min_dist:
            min_dist = dist
            possession_team = p["teamid"]

    event["possession_team_id"] = possession_team
