def summarize_game(game_data):
    """Dummy summary function - replace with your own logic"""
    return {
        "gameid": game_data.get("gameId"),
        "home_team": game_data.get("homeTeam"),
        "visitor_team": game_data.get("visitorTeam"),
        "home_score": game_data.get("homeScore"),
        "visitor_score": game_data.get("visitorScore")
    }