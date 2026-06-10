# 🚗 Traffic Accident Detection System
### A Logic-Driven Computer Vision Journey — from Basics to Physics-Aware Analysis

> Built with YOLO, OpenCV, and a lot of curiosity.

---

## What Is This?

This project is a **real-time traffic accident detection system** that analyzes video footage and identifies dangerous events — collisions, sudden stops, and hazardous proximity — using object detection and custom decision logic.

What makes it interesting is not just the final result, but the **journey**: five distinct versions of the logic layer, each solving a problem the previous one couldn't.

---

## The Journey: Logic 1 → Logic 5

### `Logic 1` — Physics-Based Position Predictor
**The Question:** *Can we predict where a car will be next frame using Newton's laws?*

Built a pure-physics `CarPredictor` class — no AI, no filters — using:

```
v = Δx / Δt             (velocity)
a = Δv / Δt             (acceleration)
x = x0 + v·t + ½·a·t²  (kinematic equation)
```

Each car kept a 3-point history. Given those three positions, the system computed velocity, then acceleration, then projected the next position.

**What worked:** The math was correct. Predictions were directionally accurate on smooth trajectories.

**What didn't:** Real-world detections are noisy. A single bad YOLO box would corrupt the history and send predictions wildly off.

> *This planted the seed for something bigger: could physics describe not just where a car goes, but what kind of collision it had?*

---

### `Logic 2` — First Full Pipeline
**The Question:** *Can we hook YOLO output into decision logic and get meaningful alerts?*

First working end-to-end system:

- **YOLOv8n** for detection + tracking (`model.track()`)
- `CalculateSpeed` class: computed `speed = distance × fps` (pixels/sec)
- Three decision functions:
  - `check_sudden_stop()` — movement < 0.5px per frame
  - `check_long_stop()` — stopped for > 5 seconds (time-based)
  - `check_pair()` — IoU overlap = collision, proximity < 80px = danger

Outputs a colored annotated video with per-frame decisions printed to console.

**What worked:** The pipeline ran. Alerts fired. The structure was solid.

**What didn't:** `speed = distance × fps` produces enormous values on high-FPS video. Time-based logic using `time.time()` tied the system to wall-clock time, making it sensitive to processing delays.

---

### `Logic 3` — Stability Pass
**The Question:** *Can we make the same logic more reliable?*

Refined Logic 2 with better error handling: explicit video open checks, dynamic filename handling, and cleaner stop-timer management. No new algorithms — a polish pass that made the system runnable without crashes in Google Colab.

---

### `Logic 4` — Frame-Based Rethink + Smoothing
**The Question:** *Can we decouple the logic from wall-clock time entirely?*

Key changes:

| What changed | Before | After |
|---|---|---|
| Stop detection | `time.time()` | Frame counter |
| Long stop threshold | `LONG_STOP_SECONDS = 5` | `LONG_STOP_FRAMES = 150` |
| Speed computation | Raw pixel distance | Exponential smoothing |
| Noise handling | None | Filter values < 0.3px → 0 |

The smoothing formula:
```
speed(t) = 0.6 × raw(t) + 0.4 × speed(t-1)
```

Also: proximity check now distinguishes **safe congestion** (both cars slow) from **dangerous proximity** (one car fast).

---

### `Logic 5 (Final)` — Production-Grade Architecture
**The Question:** *What does a real system need that none of the above had?*

A full rewrite introducing six major components:

**Global Motion Compensation (GMC)** — Optical flow estimates camera shake frame-to-frame. Vehicle movement is corrected against the camera's own motion before speed is computed.

**Kalman Filter per vehicle** — Each tracked vehicle gets its own Kalman filter: smoothed position, estimated velocity, and real future-position prediction.

**Bayesian Fusion** — Multiple weak signals (IoU, relative speed, approach angle, deceleration, TTC) are combined probabilistically into a single collision probability score.

**Time-to-Collision (TTC)** — Kinematic and predictive TTC computed per pair. The more pessimistic result is used.

**DBSCAN Cluster Detection** — Groups of stopped vehicles treated as a scene-level event. A cluster of 4+ stationary objects triggers a congestion alert.

**State Machine per vehicle** — `NORMAL → DECELERATING → STOPPED → COLLISION_CANDIDATE`

Output beyond video: `events.json`, `events.csv`, `collision_report.txt`, `summary.json`.

---

## The Physics Wall 🧱

Between Logic 1 and the later versions, there was an idea that never got fully implemented:

> **Apply Newton's collision laws to classify whether a crash was elastic or inelastic.**

In an elastic collision, kinetic energy is conserved. In an inelastic one, the vehicles crumple and move together — causing far more injury. The math is clean:

```
Elastic:   v1' = ((m1-m2)·v1 + 2·m2·v2) / (m1+m2)
Inelastic: v'  = (m1·v1 + m2·v2) / (m1+m2)
```

The blocker: **YOLO outputs pixels, not meters.** To apply Newton's laws meaningfully, you need real-world velocity in m/s and mass derived from vehicle dimensions in m². There is no reliable way to recover metric scale from a monocular camera without additional information — a known reference object, camera calibration parameters, or stereo depth data.

This remains an open and interesting problem. Solvable via road-plane homography or monocular depth estimation — but not without significant additional infrastructure.

---

## Architecture Overview

```
Video Input
    │
    ▼
YOLOv10n + BoT-SORT Tracking
    │
    ▼
ClassLocker  →  stabilizes class label per vehicle ID
    │
    ▼
GlobalMotionComp  →  estimates camera motion via optical flow
    │
    ▼
MotionAnalyzer
  ├─ KalmanTracker per vehicle
  ├─ OpticalFlow per vehicle (camera-compensated)
  └─ Speed / Acceleration / Heading history
    │
    ▼
RiskEngine
  ├─ IoU + gap check
  ├─ Relative speed
  ├─ TTC (kinematic + predictive)
  ├─ Approach angle
  ├─ BayesianFusion → collision probability
  ├─ DBSCAN congestion clustering
  └─ VehState machine per vehicle
    │
    ▼
Annotated video + JSON + CSV + report
```

---

## Version Summary

| Version | Core Contribution |
|---|---|
| Logic 1 | Physics-based position prediction (Newton's kinematics) |
| Logic 2 | First full detection-to-alert pipeline |
| Logic 3 | Crash-free Colab execution, better error handling |
| Logic 4 | Frame-stable logic, smoothed speed, congestion distinction |
| Logic 5 | Kalman + GMC + Bayesian fusion + TTC + DBSCAN |

---

## Stack

- Python 3.10+ · Ultralytics (YOLOv10/v8) · OpenCV · NumPy · scikit-learn
- Development & testing: Google Colab (T4 GPU)

---

## What's Next

- Camera calibration to recover metric scale → enables Newton's collision laws
- Elastic / inelastic collision classification from real-world velocities
- Monocular depth estimation for metric distances
- Multi-camera scene reconstruction
