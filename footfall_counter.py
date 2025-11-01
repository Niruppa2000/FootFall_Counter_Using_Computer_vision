"""
footfall_counter.py  (Enhanced)
- YOLOv8 person detection (Ultralytics)
- Centroid tracker with adjustable params
- Virtual counting line (horizontal/vertical)
- CLI flags:
    --conf               : detection confidence threshold (default 0.35)
    --orientation        : horizontal | vertical (default horizontal)
    --max-distance       : tracker matching max pixel distance (default 120)
    --max-lost           : frames to keep a lost track alive (default 30)
    --no-show            : run headless (no display window)
    --results            : path to write results.txt (default results.txt)
    --model              : yolov8n.pt (default) or other YOLO weights path
    --webcam             : use webcam (index 0) instead of a file
    --source             : input video path (ignored if --webcam is set)
    --output             : output video path (optional, mp4)
"""

import argparse
import time
import math
from collections import OrderedDict, deque
import numpy as np
import cv2
from ultralytics import YOLO


# ---------------- CentroidTracker ----------------
class CentroidTracker:
    def __init__(self, max_lost=30, max_distance=120):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.lost = OrderedDict()
        self.trajectories = OrderedDict()
        self.max_lost = max_lost
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.lost[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = deque([centroid], maxlen=64)
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]
        del self.trajectories[object_id]

    def update(self, rects):
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            if len(input_centroids) == 0:
                for oid in object_ids:
                    self.lost[oid] += 1
                    if self.lost[oid] > self.max_lost:
                        self.deregister(oid)
                return self.objects

            # distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for (r, c) in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = object_ids[r]
                self.objects[oid] = input_centroids[c]
                self.trajectories[oid].append(input_centroids[c])
                self.lost[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            # new registrations
            for j in range(len(input_centroids)):
                if j not in used_cols:
                    self.register(input_centroids[j])

            # mark unmatched tracks as lost
            for i in range(len(object_centroids)):
                if i not in used_rows:
                    oid = object_ids[i]
                    self.lost[oid] += 1
                    if self.lost[oid] > self.max_lost:
                        self.deregister(oid)
        return self.objects


# ---------------- Utilities ----------------
def draw_line(frame, p1, p2, color=(0, 255, 255), thickness=2):
    cv2.line(frame, p1, p2, color, thickness)


def centroid_crossed_line(prev, curr, line_p1, line_p2, direction='horizontal'):
    """Return crossing direction token or None."""
    if prev is None:
        return None
    (x1, y1) = prev
    (x2, y2) = curr
    (lx1, ly1) = line_p1
    (lx2, ly2) = line_p2
    if direction == 'horizontal':
        ly = (ly1 + ly2) // 2
        if (y1 < ly and y2 >= ly):
            return 'down'
        if (y1 >= ly and y2 < ly):
            return 'up'
    else:
        lx = (lx1 + lx2) // 2
        if (x1 < lx and x2 >= lx):
            return 'right'
        if (x1 >= lx and x2 < lx):
            return 'left'
    return None


# ---------------- Main ----------------
def process_video(
    source,
    output_path=None,
    show=True,
    model_name='yolov8n.pt',
    line_orientation='horizontal',
    conf_thresh=0.35,
    max_distance=120,
    max_lost=30,
    results_path="results.txt",
):
    model = YOLO(model_name)
    ct = CentroidTracker(max_lost=max_lost, max_distance=max_distance)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

    # counting line
    if line_orientation == 'horizontal':
        line_p1 = (0, H // 2)
        line_p2 = (W, H // 2)
    else:
        line_p1 = (W // 2, 0)
        line_p2 = (W // 2, H)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    counted_ids = set()
    count_in, count_out = 0, 0
    prev_centroids = {}
    heatmap = np.zeros((H, W), dtype=np.float32)

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame, verbose=False, conf=conf_thresh)[0]
        boxes = []
        # class 0 is 'person' for COCO
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box.tolist())
                boxes.append((x1, y1, x2, y2))

        objects = ct.update(boxes)

        # draw and count
        for oid, centroid in list(objects.items()):
            traj = ct.trajectories.get(oid, deque())
            for i in range(1, len(traj)):
                cv2.line(frame, traj[i - 1], traj[i], (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {oid}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cx, cy = centroid
            if 0 <= cx < W and 0 <= cy < H:
                heatmap[cy, cx] += 1.0

            crossed = centroid_crossed_line(
                prev_centroids.get(oid, None),
                centroid,
                line_p1,
                line_p2,
                direction=line_orientation
            )
            if crossed is not None and oid not in counted_ids:
                if line_orientation == 'horizontal':
                    if crossed == 'down':
                        count_in += 1
                    elif crossed == 'up':
                        count_out += 1
                else:
                    if crossed == 'right':
                        count_in += 1
                    elif crossed == 'left':
                        count_out += 1
                counted_ids.add(oid)

            prev_centroids[oid] = centroid

        draw_line(frame, line_p1, line_p2)
        cv2.rectangle(frame, (5, 5), (220, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"In: {count_in}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Out: {count_out}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        if show:
            cv2.imshow("Footfall Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # save heatmap image
    if heatmap.sum() > 0:
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        cv2.imwrite("heatmap.png", heatmap_img)

    elapsed = time.time() - t0
    print(f"Finished: Frames={frame_idx}, In={count_in}, Out={count_out}, Time={elapsed:.2f}s")

    # write results file
    if results_path:
        try:
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(f"In: {count_in}\nOut: {count_out}\nFrames: {frame_idx}\nTime_s: {elapsed:.2f}\n")
            print(f"Results written to {results_path}")
        except Exception as e:
            print(f"WARNING: could not write results file: {e}")

    return {"In": count_in, "Out": count_out, "Frames": frame_idx, "Time_s": elapsed}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, default=None, help='Path to input video (ignored if --webcam).')
    p.add_argument('--webcam', action='store_true', help='Use webcam index 0 instead of video file.')
    p.add_argument('--output', type=str, default=None, help='Output video path (mp4).')
    p.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model to use.')
    p.add_argument('--orientation', type=str, choices=['horizontal', 'vertical'], default='horizontal',
                   help='Counting line orientation.')
    p.add_argument('--conf', type=float, default=0.35, help='Detection confidence threshold.')
    p.add_argument('--max-distance', type=int, default=120, help='Tracker max matching distance in pixels.')
    p.add_argument('--max-lost', type=int, default=30, help='Tracker frames to keep lost track alive.')
    p.add_argument('--no-show', action='store_true', help='Run headless (no display window).')
    p.add_argument('--results', type=str, default='results.txt', help='Path to save results summary.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src = 0 if args.webcam else args.source
    if src is None and not args.webcam:
        raise SystemExit("Error: please provide --source <video.mp4> or use --webcam")
    process_video(
        source=src,
        output_path=args.output,
        show=not args.no_show,
        model_name=args.model,
        line_orientation=args.orientation,
        conf_thresh=args.conf,
        max_distance=args.max_distance,
        max_lost=args.max_lost,
        results_path=args.results,
    )
