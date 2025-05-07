import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_distance=50, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid
        self.disappeared = {}  # object_id -> # of consecutive frames disappeared
        self.counted_ids = set()
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        # Compute centroids
        input_centroids = []
        for box in detections:
            x1, y1, x2, y2 = box
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            input_centroids.append((cX, cY))

        # No detections → mark all as disappeared
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # No existing objects → register all detections
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances
            D = distance.cdist(np.array(object_centroids), np.array(input_centroids))

            # Find closest matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            matched_ids = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in matched_ids or col in used_cols:
                    continue
                if D[row][col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                matched_ids.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unmatched_ids = set(range(len(object_ids))) - matched_ids
            for row in unmatched_ids:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register unmatched new detections
            unmatched_cols = set(range(len(input_centroids))) - used_cols
            for col in unmatched_cols:
                self.register(input_centroids[col])

        return self.objects
