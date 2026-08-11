# Smart-Powered Traffic Accident Detection System

An end-to-end, high-performance intelligent traffic accident and hazard detection pipeline. The system utilizes Recognition Engine for custom object detection, BoT-SORT for visual object tracking, and Bayesian Decision Fusion coupled with kinematic Time-to-Collision (TTC) models to verify collisions in real-time. It also features integrated fire and smoke detection.

## Core Features

- **Custom Recognition Engine Pipeline:** Dedicated real-time detection of vehicles, motorcycles, active fires, and accidents.
- **Robust Visual Tracking:** Multi-object visual tracking using BoT-SORT with Camera Motion Compensation (GMC) and Kalman state prediction to maintain unique object IDs and filter noise.
- **Kinematic TTC Calculation:** Real-time computation of relative velocities, approach rates, and kinematic time-to-collision margins.
- **Bayesian Decision Fusion:** Probabilistic fusion of kinematic anomalies, decelerations, and overlaps to minimize false positives from high-density traffic.
- **Fire & Smoke Detection:** Dynamic region-of-interest (ROI) and full-frame monitoring for fires and smoke using a dedicated Recognition Engine model.
- **Premium Cybernetic Dashboard:** Responsive, glassmorphic HUD web interface featuring real-time diagnostic logs, analytics charts, uploader modules, and sound indicators.
- **Flexible Execution:** Supports both local hardware setups and public Google Colab GPU server tunnels.

## Repository Structure

- `server.py`: FastAPI web server hosting endpoints for video upload, processing state polling, and static file delivery.
- `logic.py`: Core processing engine orchestrating the Recognition Engine tracker, Kalman motion analyzer, Bayesian risk assessment, and fire detector.
- `config.py`: Centralized configuration manager defining host, port, model weights, detection confidence thresholds, and kinematic constraints.
- `utils.py`: Helper scripts for FFmpeg video transcoding, public LocalTunnel setup, and automated dependency validation.
- `index.html`: Fully responsive premium web frontend dashboard.

## Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify FFmpeg:**
   FFmpeg is recommended for web-friendly video transcoding (H.264). Ensure it is installed and added to your system's PATH.

3. **Start the Backend Server:**
   Run the FastAPI server locally:
   ```bash
   python server.py --host 0.0.0.0 --port 8000
   ```

4. **Access the Interface:**
   Simply open `index.html` in any modern web browser, enter your server URL (e.g., `http://localhost:8000`), and click **Establish Connection**.