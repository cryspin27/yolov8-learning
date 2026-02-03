from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------- SETTINGS (edit these) --------
MODEL_PATH = "runs/detect/train6/weights/best.pt"
VIDEO_PATH = "videos/traffic.mp4.mp4"   # change if you renamed it
CONF = 0.25
IOU = 0.5
SHOW = True            # set False if you don't want a window
SAVE_VIDEO = True
# --------------------------------------

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path("portfolio") / "traffic-vehicle-detection-yolov8"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_path = out_dir / f"demo_output_{timestamp}.mp4"
    out_csv_path = out_dir / f"detections_{timestamp}.csv"

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
        r = results[0]

        # Count per class in this frame
        counts = {}
        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for cid in cls_ids:
                name = model.names.get(cid, f"class{cid}")
                counts[name] = counts.get(name, 0) + 1

        # Save a row (frame time + counts)
        t = frame_idx / fps
        row = {"frame": frame_idx, "time_s": round(t, 3)}
        row.update(counts)
        rows.append(row)

        # Draw boxes
        annotated = r.plot()

        if SHOW:
            cv2.imshow("Traffic Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows).fillna(0)

    # Ensure numeric columns are ints where possible
    for c in df.columns:
        if c not in ("frame", "time_s"):
            df[c] = df[c].astype(int)

    df.to_csv(out_csv_path, index=False)

    # Print summary totals
    totals = df.drop(columns=["frame", "time_s"]).sum().sort_values(ascending=False)
    print("\n=== SUMMARY TOTALS ===")
    print(totals.to_string())

    print(f"\nSaved video: {out_video_path}")
    print(f"Saved CSV:   {out_csv_path}")

if __name__ == "__main__":
    main()
