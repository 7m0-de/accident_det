# 🚗 Vehicle Crash Detection System — Advanced Logic Edition

> Real-time multi-vehicle accident detection from video using a layered fusion of AI models, mathematical algorithms, and probabilistic reasoning — all built and orchestrated by hand.

---

## 🧑‍💻 What I Built — Developer Overview

I am responsible for the **entire detection logic** of this system. My role was not simply to call a pre-trained model and call it done — the core engineering challenge was designing a **multi-layered decision pipeline** that transforms raw video frames into reliable, low-false-positive accident alerts.

Here is what I personally designed and implemented:

- **Fused three independent motion signals** (bounding box position, sparse optical flow, and Kalman filter prediction) into a single stable velocity estimate per vehicle per frame.
- **Compensated for camera motion** using an affine transformation estimated between frames, so that a moving camera does not confuse the system into thinking stationary vehicles are moving.
- **Built a Bayesian probabilistic engine** that combines 11 independent collision signals (overlap, relative speed, time-to-collision, sudden deceleration, trajectory anomaly, convoy detection, and more) into a single posterior probability — replacing fragile `if/else` chains with a mathematically grounded decision.
- **Designed two independent TTC (Time-To-Collision) estimators** — one analytic (kinematic equations) and one simulation-based (Kalman rollout) — cross-validated at runtime.
- **Implemented a Finite State Machine (FSM)** for each tracked vehicle to model the full post-crash lifecycle: `NORMAL → APPROACHING → CRITICAL → COLLIDED → POST → BURNED`, with locked states that cannot regress.
- **Added convoy suppression logic** to distinguish between two cars travelling in formation (false positive) versus two cars on a collision course.
- **Integrated DBSCAN clustering** to detect dense traffic pile-ups as a separate high-level event type.
- **Wired a fire detection AI model** on top of the collision engine to flag vehicle fires and smoke independently.

The pre-trained models (YOLOv10, BoT-SORT, Fire-YOLO, DBSCAN) are external tools. Everything that decides **what is dangerous, when to fire an alert, and how confident the system is** — that is my logic.

---

## ⚙️ How the Pipeline Works — Frame by Frame

Every video frame passes through the following ordered stages:

```
[1] Read Frame
      ↓
[2] GMC  ──────────── Subtract camera movement noise
      ↓
[3] YOLOv10 + BoT-SORT ─── Detect & track vehicles (AI)
      ↓
[4] ClassLocker ────── Stabilize vehicle class label
      ↓
[5] Optical Flow + Kalman ── Compute velocity & heading
      ↓
[6] Risk Engine
      ├─ Kinematic TTC ─── Time-to-collision (physics)
      ├─ Predictive TTC ── Time-to-collision (simulation)
      ├─ Bayesian Fusion ─ Posterior collision probability
      ├─ FSM Update ─────── Lifecycle state per vehicle
      └─ DBSCAN ─────────── Traffic cluster detection
      ↓
[7] FireDetectorAI ────── Detect fire / smoke per vehicle
      ↓
[8] Draw ──────────────── Render results onto frame
      ↓
[9] Save video + JSON / CSV / TXT reports
```

---

## 🧠 Algorithms Used — Full Breakdown

### 1. YOLOv10n — Vehicle Detection

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 🤖 Pre-trained AI (Deep Neural Network) 
|
|
**
Source
**
|
 Ultralytics — used as-is 
|
|
**
Why used
**
|
 State-of-the-art real-time object detection; detects cars, trucks, motorcycles, and buses in each frame with high accuracy 
|

---

### 2. BoT-SORT — Multi-Object Tracking

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 🤖 Pre-trained AI (Multi-Object Tracker) 
|
|
**
Source
**
|
 Built into Ultralytics; configured via custom 
`botsort.yaml`
|
|
**
Why used
**
|
 Assigns a persistent ID to each vehicle across frames by combining Hungarian matching, appearance re-identification (ReID), and sparse optical flow globally 
|

---

### 3. GMC — Global Motion Compensation

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📐 Mathematical / Computer Vision 
|
|
**
Source
**
|
 Built manually using OpenCV primitives 
|
|
**
Why used
**
|
 Without it, a panning or shaking camera makes all stationary vehicles appear to be moving, polluting every downstream speed and velocity calculation 
|

**How it works:**
1. Extract up to 300 Shi-Tomasi corner points from the current frame.
2. Track those points into the next frame using Lucas-Kanade optical flow.
3. Estimate a 2×3 affine transformation between the two point sets using RANSAC (to reject outliers).
4. The resulting matrix represents camera displacement — subtracted from every vehicle's measured motion.

---

### 4. Sparse Optical Flow (Lucas-Kanade)

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📐 Mathematical / Computer Vision 
|
|
**
Source
**
|
 Built manually using 
`cv2.calcOpticalFlowPyrLK`
|
|
**
Why used
**
|
 Provides a raw per-pixel motion measurement inside each vehicle's bounding box, independent of the YOLO detection — giving a second, uncorrelated velocity signal 
|

**How it works:**
- Extract 50 feature points inside each vehicle's ROI.
- Track them frame-to-frame using pyramid Lucas-Kanade.
- Take the **median** (not mean) of all displacements to resist outliers.
- Subtract the GMC camera transform to isolate true vehicle motion.

---

### 5. Kalman Filter — Smooth Position & Velocity Estimation

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📐 Mathematical / Statistical (built from scratch) 
|
|
**
Source
**
|
 Fully implemented manually 
|
|
**
Why used
**
|
 Smooths noisy position measurements into a clean, stable velocity vector; predicts future position when detection is temporarily lost; enables trajectory rollout for TTC estimation 
|

**State vector:** `[x, y, vx, vy]`

The filter runs a predict-then-correct cycle every frame:
- **Predict:** `x̂ = F · x` (project forward using motion model)
- **Correct:** blend prediction with measurement via Kalman gain `K`

This gives `vx` and `vy` for every tracked vehicle automatically.

---

### 6. Mahalanobis Distance — Trajectory Anomaly Score

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📊 Statistical (built from scratch) 
|
|
**
Source
**
|
 Fully implemented manually inside 
`KalmanTracker`
|
|
**
Why used
**
|
 A vehicle that suddenly changes velocity in an unexpected way (collision impact) produces an innovation residual that is statistically abnormal relative to that vehicle's own history — this score captures that signal 
|

**How it works:**
- Store the last 30 Kalman innovation vectors (prediction error).
- Compute the covariance matrix of those innovations.
- For the current step, compute: `d = √(Δ · Σ⁻¹ · Δᵀ)`
- High value → anomalous motion → feeds into Bayesian fusion.

---

### 7. ClassLocker — Stable Label Voting

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 💡 Rule-based logic (manual) 
|
|
**
Source
**
|
 Built manually 
|
|
**
Why used
**
|
 YOLO can fluctuate between 
`car`
 and 
`truck`
 depending on angle and occlusion. This module votes over 2 seconds of detections per track ID and locks the majority class permanently, preventing label noise from affecting downstream logic 
|

---

### 8. TTC Kinematic — Analytic Time-To-Collision

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📐 Physics / Mathematics (manual) 
|
|
**
Source
**
|
 Built manually 
|
|
**
Why used
**
|
 Provides an instantaneous, closed-form estimate of how many seconds remain before two vehicles would intersect based on their current positions and velocity vectors 
|

**Core equation:**

```
closing_speed = -(r⃗ · rv⃗) / |r⃗|          # relative approach speed
TTC = -(r⃗ · rv⃗) / |rv⃗|²                   # time to minimum distance
```

If `TTC < 0.6s` → critical. If `TTC < 1.8s` → warning.

---

### 9. TTC Predictive — Simulation-Based Time-To-Collision

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📐 Mathematical / Predictive (manual) 
|
|
**
Source
**
|
 Built manually using Kalman rollout 
|
|
**
Why used
**
|
 The kinematic TTC assumes constant velocity. The predictive TTC uses the Kalman filter to project each vehicle's future position step-by-step (up to 2.5 seconds ahead), catching curved or decelerating trajectories that the analytic model misses 
|

The final TTC used is `min(kinematic_TTC, predictive_TTC)`.

---

### 10. Bayesian Fusion — Probabilistic Collision Confidence

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 📊 Statistical / Probabilistic (manual) 
|
|
**
Source
**
|
 Built manually using Bayes' theorem in log-odds space 
|
|
**
Why used
**
|
 Replaces fragile 
`if/else`
 thresholds with a principled probabilistic framework that combines 11 independent signals into a single posterior probability, weighted by each signal's empirical reliability 
|

**11 input signals and their likelihood ratios:**

|
 Signal 
|
 P(signal \| collision) 
|
 P(signal \| normal) 
|
|
---
|
---
|
---
|
|
 IoU ≥ 0.45 
|
 0.88 
|
 0.04 
|
|
 IoU ≥ 0.30 
|
 0.68 
|
 0.12 
|
|
 Contact (gap ≤ threshold) 
|
 0.82 
|
 0.18 
|
|
 TTC < 0.6s 
|
 0.87 
|
 0.07 
|
|
 TTC < 1.8s 
|
 0.62 
|
 0.22 
|
|
 High relative speed 
|
 0.77 
|
 0.12 
|
|
 Sudden deceleration 
|
 0.72 
|
 0.10 
|
|
 Post-impact signature 
|
 0.90 
|
 0.03 
|
|
 Impact signature 
|
 0.86 
|
 0.05 
|
|
 Mahalanobis anomaly 
|
 0.70 
|
 0.14 
|
|
 Convoy detected 
|
 0.06 
|
 0.68 
|

**Threshold:** posterior ≥ 74% → confirmed collision event logged.

---

### 11. Post-Impact FSM — Vehicle Lifecycle State Machine

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 💡 Rule-based / Finite State Machine (manual) 
|
|
**
Source
**
|
 Built manually 
|
|
**
Why used
**
|
 Tracks each vehicle through a structured lifecycle of states, preventing already-confirmed collisions from being "forgotten" if the vehicle momentarily looks normal, and detecting when a vehicle becomes permanently stationary post-crash 
|

```
NORMAL → APPROACHING → CRITICAL → COLLIDED → POST → BURNED
                                      ↑
                               No regression allowed
```

---

### 12. Convoy Suppression

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 💡 Rule-based logic (manual) 
|
|
**
Source
**
|
 Built manually 
|
|
**
Why used
**
|
 Two vehicles driving together in the same direction at similar speed produce high IoU and low gap scores — without this filter they would constantly trigger false collision alerts 
|

**Rule:** If heading difference ≤ 35° **and** relative speed ≤ 2.5 px/frame → suppress collision events for this pair.

---

### 13. DBSCAN — Traffic Cluster Detection

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 🤖 Unsupervised Machine Learning 
|
|
**
Source
**
|
`sklearn.cluster.DBSCAN`
 — used as-is 
|
|
**
Why used
**
|
 Detects dense vehicle pile-ups without needing to specify the number of clusters in advance; triggers a 
`TRAFFIC_CLUSTER`
 event when 4+ vehicles appear within a 130px radius 
|

---

### 14. Registry — Persistent Collision Record

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 💡 Rule-based logic (manual) 
|
|
**
Source
**
|
 Built manually 
|
|
**
Why used
**
|
 Ensures each unique vehicle pair is confirmed and logged exactly once using a 
`frozenset`
 key; re-emits confirmed collisions as persistent events on every subsequent frame so that vehicles remain highlighted red permanently 
|

---

### 15. FireDetectorAI — Vehicle Fire & Smoke Detection

|
 Property 
|
 Detail 
|
|
---
|
---
|
|
**
Type
**
|
 🤖 Pre-trained AI (YOLOv8n fine-tuned on fire/smoke) 
|
|
**
Source
**
|
`keremberke/yolov8n-fire-detection`
 from Hugging Face 
|
|
**
Why used
**
|
 Detects post-collision vehicle fires and suspicious smoke independently of the collision engine, providing early warning of thermal hazards 
|

**How it works:**
- Expand each vehicle's ROI by 60% upward (to capture flames above the roof).
- Run the fire model on the expanded ROI.
- Accumulate results over a 10-frame sliding window per vehicle.
- Confirm **fire** at ≥ 3 positive frames; confirm **smoke** at ≥ 4.
- Once confirmed, the fire state is permanently locked — it cannot be reversed.

---

## 📊 Algorithm Classification Summary

|
#
|
 Algorithm 
|
 Type 
|
 Source 
|
|
---
|
---
|
---
|
---
|
|
 1 
|
 YOLOv10n 
|
 🤖 Pre-trained AI 
|
 Ultralytics — ready 
|
|
 2 
|
 BoT-SORT 
|
 🤖 Pre-trained AI 
|
 Ultralytics — ready 
|
|
 3 
|
 GMC + RANSAC 
|
 📐 Mathematical 
|
 OpenCV — built manually 
|
|
 4 
|
 Optical Flow (LK) 
|
 📐 Mathematical 
|
 OpenCV — built manually 
|
|
 5 
|
 Kalman Filter 
|
 📐 Mathematical / Statistical 
|
 Built from scratch 
|
|
 6 
|
 Mahalanobis Distance 
|
 📊 Statistical 
|
 Built from scratch 
|
|
 7 
|
 ClassLocker 
|
 💡 Rule-based logic 
|
 Built from scratch 
|
|
 8 
|
 TTC Kinematic 
|
 📐 Physics / Math 
|
 Built from scratch 
|
|
 9 
|
 TTC Predictive 
|
 📐 Mathematical 
|
 Built from scratch 
|
|
 10 
|
 Bayesian Fusion 
|
 📊 Statistical 
|
 Built from scratch 
|
|
 11 
|
 FSM States 
|
 💡 Rule-based logic 
|
 Built from scratch 
|
|
 12 
|
 Convoy Suppression 
|
 💡 Rule-based logic 
|
 Built from scratch 
|
|
 13 
|
 DBSCAN 
|
 🤖 Unsupervised ML 
|
 scikit-learn — ready 
|
|
 14 
|
 Registry 
|
 💡 Rule-based logic 
|
 Built from scratch 
|
|
 15 
|
 FireDetectorAI 
|
 🤖 Pre-trained AI 
|
 Hugging Face — ready 
|

**Summary by category:**

|
 Category 
|
 Count 
|
|
---
|
---
|
|
 🤖 Pre-trained AI 
|
 4 
|
|
 📐 Mathematical / Physics 
|
 5 
|
|
 📊 Statistical / Probabilistic 
|
 2 
|
|
 🤖 Unsupervised ML 
|
 1 
|
|
 💡 Rule-based / Manual logic 
|
 5 
|

---

## 🔄 Old vs New — Full Comparison

This section compares the **baseline version** (simple detection, no advanced algorithms) with the **enhanced version** built with the full algorithm stack.

### Architecture Comparison

|
 Component 
|
 Old Version 
|
 New Version 
|
|
---
|
---
|
---
|
|
**
Object detector
**
|
 YOLOv10x (heavy) 
|
 YOLOv10n (lightweight, Colab-friendly) 
|
|
**
Tracker
**
|
 BoT-SORT (basic config) 
|
 BoT-SORT (tuned YAML, adaptive ReID) 
|
|
**
Camera stabilisation
**
|
 ❌ None 
|
 ✅ GMC — affine + RANSAC 
|
|
**
Velocity estimation
**
|
 Simple Euclidean distance between centers 
|
 Optical Flow + Kalman fusion (60/40 blend) 
|
|
**
Position smoothing
**
|
 ❌ Raw bounding box center 
|
 ✅ Kalman filter (predict + correct) 
|
|
**
Future prediction
**
|
 ❌ None 
|
 ✅ Kalman rollout (2.5s ahead) 
|
|
**
Anomaly detection
**
|
 ❌ None 
|
 ✅ Mahalanobis distance per vehicle 
|
|
**
Class stability
**
|
 Simple lock after 5s 
|
 Majority voting over 8s window 
|
|
**
Collision signal
**
|
 IoU threshold only 
|
 IoU + TTC (×2) + gap + speed + decel + Mahalanobis 
|
|
**
Decision model
**
|
`if iou > threshold`
|
 Bayesian posterior over 11 signals 
|
|
**
False positive suppression
**
|
 Basic cooldown timer 
|
 Convoy detection + Bayesian prior + cooldown 
|
|
**
Vehicle lifecycle
**
|
 None — binary alert only 
|
 FSM: 6-state lifecycle, no regression 
|
|
**
Post-crash tracking
**
|
 ❌ Alert disappears 
|
 ✅ BURNED state, permanent highlight 
|
|
**
Traffic event
**
|
 ❌ None 
|
 ✅ DBSCAN cluster detection 
|
|
**
Static obstacle check
**
|
 ❌ None 
|
 ✅ Frame-boundary wall collision 
|
|
**
Fire / smoke detection
**
|
 ❌ None 
|
 ✅ YOLOv8n-fire-detection on expanded ROI 
|
|
**
Output files
**
|
 Video only 
|
 Video + events JSON + CSV + collision report + summary 
|
|
**
Checkpointing
**
|
 ❌ None 
|
 ✅ JSON checkpoint every 500 frames 
|

---

### Detection Quality Comparison

|
 Metric 
|
 Old Version 
|
 New Version 
|
 Improvement 
|
|
---
|
---
|
---
|
---
|
|
**
False positive rate
**
|
 High (IoU alone triggers at traffic lights, convoys) 
|
 Very low (Bayesian + convoy filter) 
|
 🔻 Major reduction 
|
|
**
False negative rate
**
|
 Moderate (misses slow-speed contact) 
|
 Low (post-impact FSM catches slow-speed collisions) 
|
 🔻 Reduced 
|
|
**
Collision confidence score
**
|
 Binary — yes/no 
|
 0–100% posterior probability 
|
 📈 Quantified 
|
|
**
Detection latency
**
|
 Immediate (same frame IoU) 
|
 4-frame confirmation window 
|
 ⚖️ Slight delay for accuracy 
|
|
**
Velocity accuracy
**
|
 Noisy (raw pixel distance) 
|
 Smooth (Kalman + optical flow + GMC) 
|
 📈 High accuracy 
|
|
**
Camera motion robustness
**
|
 ❌ Breaks on camera pan/tilt 
|
 ✅ Fully compensated 
|
 📈 Robust 
|
|
**
Post-collision tracking
**
|
 ❌ Lost after cooldown 
|
 ✅ Permanent highlight via FSM 
|
 📈 Full lifecycle 
|
|
**
Multi-event types
**
|
 2 (collision, proximity) 
|
 9 (collision, TTC warn, contact warn, sudden decel, sharp turn, long stop, static collision, cluster, burned) 
|
 📈 7 new event types 
|
|
**
Fire detection
**
|
 ❌ Not present 
|
 ✅ AI-powered, frame-level confirmed 
|
 📈 New capability 
|
|
**
Explainability
**
|
 None 
|
 Full event log with IoU, TTC, Bayes score, Mahal score per event 
|
 📈 Fully traceable 
|

---

### Event Types: Old vs New

|
 Event Type 
|
 Old Version 
|
 New Version 
|
|
---
|
---
|
---
|
|
 COLLISION 
|
 ✅ (IoU only) 
|
 ✅ (Bayesian ≥ 74%) 
|
|
 DANGEROUS_PROXIMITY 
|
 ✅ (distance only) 
|
 ✅ (with speed filter) 
|
|
 TTC_WARNING 
|
 ❌ 
|
 ✅ 
|
|
 CONTACT_WARNING 
|
 ❌ 
|
 ✅ 
|
|
 SUDDEN_DECEL 
|
 ❌ 
|
 ✅ 
|
|
 SHARP_TURN 
|
 ❌ 
|
 ✅ 
|
|
 LONG_STOP 
|
 ✅ (basic) 
|
 ✅ (requires nearby moving vehicle) 
|
|
 STATIC_COLLISION 
|
 ❌ 
|
 ✅ (frame wall boundaries) 
|
|
 TRAFFIC_CLUSTER 
|
 ❌ 
|
 ✅ (DBSCAN) 
|
|
 BURNED (post-crash stationary) 
|
 ❌ 
|
 ✅ (FSM) 
|
|
 VEHICLE_FIRE 
|
 ❌ 
|
 ✅ (AI model) 
|
|
 VEHICLE_SMOKE 
|
 ❌ 
|
 ✅ (AI model) 
|

---

### Code Architecture Comparison

|
 Aspect 
|
 Old Version 
|
 New Version 
|
|
---
|
---
|
---
|
|
**
Lines of logic code
**
|
 ~350 
|
 ~900+ 
|
|
**
Classes
**
|
 4 
|
 12 
|
|
**
Decision model
**
|
 Threshold comparisons 
|
 Bayesian probabilistic fusion 
|
|
**
Motion model
**
|
 None (raw Euclidean) 
|
 Kalman filter with 4-state vector 
|
|
**
Velocity source
**
|
 1 signal (center delta) 
|
 3 fused signals (YOLO + optical flow + Kalman) 
|
|
**
TTC methods
**
|
 0 
|
 2 (kinematic + predictive) 
|
|
**
State tracking per vehicle
**
|
 None 
|
 6-state FSM 
|
|
**
Output formats
**
|
 MP4 video 
|
 MP4 + JSON events + CSV + collision report + summary JSON 
|
|
**
Persistence between sessions
**
|
 None 
|
 JSON checkpoint every 500 frames 
|
|
**
Adaptability
**
|
 Fixed thresholds 
|
 Auto-scales with video resolution and FPS 
|

---

## 📁 Output Files

|
 File 
|
 Description 
|
|
---
|
---
|
|
`output.mp4`
|
 Annotated video with bounding boxes, velocity arrows, event banners 
|
|
`events.json`
|
 Full structured log of every detected event with all diagnostic fields 
|
|
`events.csv`
|
 Flat CSV version for spreadsheet analysis 
|
|
`collision_report.txt`
|
 Human-readable report of confirmed collisions only 
|
|
`summary.json`
|
 Aggregate statistics: vehicle counts, event counts, algorithm list 
|
|
`track_log_checkpoint_N.json`
|
 Auto-save every 500 frames to prevent data loss 
|

---

## 🛠 Requirements

```bash
pip install ultralytics opencv-python-headless scikit-learn
```

Runs on **Google Colab** (CPU or GPU). GPU strongly recommended for real-time processing.

---

## 🚀 Quick Start

```python
from crash_detection import run

summary = run("your_video.mp4")
print(summary)
```

---

## 📌 Notes

- All thresholds (proximity distance, TTC limits, Bayesian weights) are **auto-scaled** to the input video's resolution and frame rate at startup.
- The Bayesian prior is set to 1.5% — reflecting that most vehicle pairs in traffic are **not** colliding. This keeps the false positive rate low even when multiple warning signals fire simultaneously.
- The fire detection module is **optional** — if the model fails to download, the system continues with all other algorithms intact.
