readme_text = """# 🚗 Vehicle Crash Detection System — Advanced Logic Edition

Real-time multi-vehicle accident detection from video using a layered fusion of AI models, mathematical algorithms, and probabilistic reasoning — all built and orchestrated by hand.

---

## 🧑‍💻 What I Built — Developer Overview

I am responsible for the **entire detection logic** of this system. My role was not simply to call a pre-trained model and call it done — the core engineering challenge was designing a **multi-layered decision pipeline** that transforms raw video frames into reliable, low-false-positive accident alerts.

Here is what I personally designed and implemented:
* **Fused three independent motion signals** (bounding box position, sparse optical flow, and Kalman filter prediction) into a single stable velocity estimate per vehicle per frame.
* **Compensated for camera motion** using an affine transformation estimated between frames, so that a moving camera does not confuse the system into thinking stationary vehicles are moving.
* **Built a Bayesian probabilistic engine** that combines 11 independent collision signals (overlap, relative speed, time-to-collision, sudden deceleration, trajectory anomaly, convoy detection, and more) into a single posterior probability — replacing fragile `if/else` chains with a mathematically grounded decision.
* **Designed two independent TTC (Time-To-Collision) estimators** — one analytic (kinematic equations) and one simulation-based (Kalman rollout) — cross-validated at runtime.
* **Implemented a Finite State Machine (FSM)** for each tracked vehicle to model the full post-crash lifecycle: `NORMAL → APPROACHING → CRITICAL → COLLIDED → POST → BURNED`, with locked states that cannot regress.
* **Added convoy suppression logic** to distinguish between two cars travelling in formation (false positive) versus two cars on a collision course.
* **Integrated DBSCAN clustering** to detect dense traffic pile-ups as a separate high-level event type.
* **Wired a fire detection AI model** on top of the collision engine to flag vehicle fires and smoke independently.

> 💡 **Core Architecture Note:** The pre-trained models (YOLOv10, BoT-SORT, Fire-YOLO, DBSCAN) are external tools. Everything that decides **what is dangerous, when to fire an alert, and how confident the system is** — that is my proprietary logic.

---

## ⚙️ How the Pipeline Works — Frame by Frame

Every video frame passes through the following ordered stages dynamically:
