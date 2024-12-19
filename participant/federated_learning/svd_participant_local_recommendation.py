def normalize_string(s):
    """
    """
    return s.replace('\u200b', '').lower()

def compute_recommendations(user_U, global_V, tv_vocab, user_ratings, recent_week=12, exclude_watched=True):
    """Compute recommendations based on user preferences and recent activity."""
    print("Selecting recommendations based on most recent shows watched...")
    
    # Extract recent items
    recent_items = [title for (title, week, n_watched, rating) in user_ratings if week == recent_week]
    recent_item_ids = [tv_vocab[title] for title in recent_items if title in tv_vocab]
    print("For week (of all years)", recent_week, "watched n_shows=:", len(recent_items))
    
    # Combine long-term and recent preferences
    alpha = 0.7  # Weight for long-term preferences
    beta = 0.3   # Weight for recent preferences
    if recent_item_ids:
        U_global_activity = sum(global_V[item_id] for item_id in recent_item_ids) / len(recent_item_ids)
        U_recent = alpha * user_U + beta * U_global_activity
    else:
        U_recent = user_U  # fallback

    # Prepare candidate items
    all_items = list(tv_vocab.keys())
    watched_titles = set(normalize_string(t) for (t, _, _, _) in user_ratings)
    if exclude_watched:
        candidate_items = [title for title in all_items if normalize_string(title) not in watched_titles]
    else:
        candidate_items = all_items

    # Generate predictions
    predictions = []
    for title in candidate_items:
        item_id = tv_vocab[title]
        pred_rating = U_recent.dot(global_V[item_id])
        predictions.append((title, pred_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:6]  # Return top 6 recommendations
