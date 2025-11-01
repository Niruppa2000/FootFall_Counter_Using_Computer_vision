# Footfall Counter using Computer Vision

## 🎯 Project Overview
This project implements a **Footfall Counter** that automatically detects, tracks, and counts the number of people entering and exiting a zone in a video using **Computer Vision** and **YOLOv8 (Ultralytics)**.  
It also generates a **heatmap** to visualize high-traffic regions in the frame.

---

## 🧠 Technologies Used
- Python 3.10
- OpenCV
- Ultralytics YOLOv8
- Numpy
- ImageIO

---

## ⚙️ How It Works
1. The video is processed frame-by-frame using YOLOv8 to detect people (class ID = 0).
2. Each detected person is assigned an **object ID** and tracked across frames using a **centroid tracker**.
3. A **virtual line** (vertical or horizontal) is placed in the frame.
4. Whenever a tracked person crosses the line:
   - Moving one way increases the **In** count.
   - Moving the other way increases the **Out** count.
5. A heatmap is generated to show where most movement occurs.

---

## ▶️ How to Run (Command Used)
To execute this project on the provided video, use the following command:

```bash
python footfall_counter.py --source istockphoto-1474927019-640_adpp_is.mp4 --output processed_output.mp4 --orientation vertical --results results.txt
```

### Explanation:
| Argument | Description |
|-----------|-------------|
| `--source` | Input video file path |
| `--output` | Output video with bounding boxes & counts |
| `--orientation vertical` | Uses a vertical line for counting (best for side-to-side movement) |
| `--results` | Saves In/Out counts summary to a text file |

---

## 📁 Output Files Generated
| File | Description |
|------|--------------|
| `processed_output.mp4` | Output video with detections, IDs, and In/Out counts |
| `heatmap.png` | Heatmap showing frequent movement zones |
| `results.txt` | Text summary of total In, Out, and Frames processed |

Example `results.txt`:
```
In: 4
Out: 2
Frames: 1150
Time_s: 46.78
```

---

## 📊 Visual Output Example
- People detected and tracked using colored bounding boxes.
- IDs and count displayed on top-left of video.
- A vertical yellow counting line drawn in the center.
- Heatmap image generated showing foot traffic density.

---

## 🏁 Results Summary
✅ People detected and tracked accurately  
✅ Entry/Exit counted successfully  
✅ Heatmap generated for analysis  
✅ Output video and results file created correctly  

---

## 🧩 Applications
- Retail Store Footfall Analysis  
- Mall or Office Entry-Exit Monitoring  
- Crowd Management in Public Spaces  
- Smart Surveillance Systems  

---

## 👩‍💻 Developed by
**Niruppa Abishek**  
Codingal AI Assignment – *Footfall Counter using Computer Vision*

---
