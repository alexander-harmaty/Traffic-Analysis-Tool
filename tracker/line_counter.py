# tracker/line_counter.py
from collections import defaultdict, deque

class LineCounter:
    def __init__(self, line_start, line_end, direction="down", band_height=20, memory_frames=5):
        """
        line_start, line_end: tuple of (x, y) for the line segment
        direction: 'up' or 'down' determines which way counts
        band_height: vertical thickness of the counting band in pixels
        memory_frames: number of previous centroids to track per object
        """
        self.line_start = line_start
        self.line_end = line_end
        self.direction = direction
        self.line_y_min = line_start[1] - band_height // 2
        self.line_y_max = line_start[1] + band_height // 2
        self.counted_ids = set()
        self.previous_centroids = defaultdict(lambda: deque(maxlen=memory_frames))

    def check_crossing(self, object_id, current_centroid):
        """
        Returns True if the object has crossed the line band and hasn't been counted yet.
        """
        if object_id in self.counted_ids:
            return False

        self.previous_centroids[object_id].append(current_centroid)
        if len(self.previous_centroids[object_id]) < 2:
            return False

        prev_y = self.previous_centroids[object_id][-2][1]  # y from second last frame
        curr_y = self.previous_centroids[object_id][-1][1]  # y from last frame

        crossed = False
        if self.direction == "down" and prev_y < self.line_y_min and curr_y >= self.line_y_min:
            crossed = True
        elif self.direction == "up" and prev_y > self.line_y_max and curr_y <= self.line_y_max:
            crossed = True

        if crossed:
            self.counted_ids.add(object_id)
            return True

        return False
