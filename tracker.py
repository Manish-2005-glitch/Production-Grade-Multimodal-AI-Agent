from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age = 30)

def track(detections, frame):
    tracks = tracker.update_tracks(
        [(d["bbox"], d["conf"], d["cls"]) for d in detections],
        frame = frame
    )
    
    return[
        {"id":t.track_id, "bbox":t.to_ltrb()}
        for t in tracks if t.is_confirmed()
    ]