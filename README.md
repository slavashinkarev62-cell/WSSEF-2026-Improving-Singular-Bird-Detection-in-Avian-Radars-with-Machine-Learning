
# WSSEF-2026-Improving-Singular-Bird-Detection-in-Avian-Radars-with-Machine-Learning
Improving singular bird detection in avian radars using YOLO and BoT-SORT. Developed for the Washington State Science and Engineering Fair (WSSEF)."

System Architecture

Phase 1: Detection & Tracking (YOLOv8 & BoT-SORT) Acting as the system's eyes, the detection software analyzes continuous streams of raw Plan Position Indicator (PPI) radar imagery. Utilizing the YOLOv8 algorithm alongside BoT-SORT tracking, it scans each frame for high-intensity returns. The model draws bounding boxes around potential biological targets (individual birds or flocks), assigns them unique tracking IDs, and actively ignores static background clutter.
Phase 2: Kinematic Analysis Once a target is isolated, the tracking script calculates its spatial movement across consecutive frames. By computing the coordinate distance between the target's past and current positions, the system estimates the object's velocity in knots. Simultaneously, it calculates the line-of-sight distance to predefined Cartesian runway coordinates to classify the target's trajectory as 'Inbound' or 'Outbound'.
Phase 3: Threat Triage (LLM Integration) Calculated telemetry—including speed, size, runway distance, and heading—is serialized into a JSON format and fed into a Large Language Model. Acting as an automated triage officer, the LLM applies strict operational rules to filter out non-threats. This directly prevents alert fatigue by translating only the critical, intersecting trajectories into concise, actionable aviation safety warnings.

Training Methodology & Data Strategy While the broader pipeline demonstrates the system's utility, the core technical contribution of this project is the custom-trained YOLOv8 model. A major challenge in development was the absence of publicly available datasets for avian PPI radar imagery. To overcome this and prove the concept, the model was trained on a Synthetic Aperture Radar (SAR) ship detection dataset. This served as a highly effective proxy for teaching the model to identify small, high-intensity returns against heavily cluttered backgrounds.

Results & Performance Metrics The training results demonstrated exceptional promise, proving the viability of using YOLO and BoT-SORT technologies to improve radar efficiency. The model achieved the following metrics:
mAP@0.5: 90.1%
mAP@0.5-0.95: 61.4%
Precision: 90.0%
Recall: 81.0%

Future Work While the proxy dataset established a strong baseline, the next phase of development requires evaluating and fine-tuning the model utilizing actual PPI imagery of biological targets to fully realize the technology's operational potential.

Results/Documentation
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/b3b58f27-4307-4664-ab2e-67cc58089cb5" />


Example Training Batch #593
<img width="1920" height="1920" alt="train_batch5930" src="https://github.com/user-attachments/assets/33d9c55c-3fdc-4a6b-9091-d9f91dd5a4e1" />

Box_f1curve
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/4ea81a5f-18a9-4874-9833-81f696c6e1e2" />

Labeling<img width="1600" height="1600" alt="labels" src="https://github.com/user-attachments/assets/ce0fad34-78c6-4bec-965a-1736e44de564" />





