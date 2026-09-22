
# Adaptive Multimodal Human–Robot Communication Using a Multi-Agent AI Framework with Safety-Aware Decision Making

## Overview

This project presents an adaptive multimodal Human–Robot Interaction (HRI) framework that enables robots to interpret human instructions through multiple communication modalities and make context-aware, safety-conscious decisions.

The system combines:

- Speech
- Hand gestures
- Visual perception
- Multimodal fusion
- Contextual memory
- Task planning
- Safety verification
- Human-in-the-loop (HITL) decision making
- ROS 2-based robot control

The primary objective is to enable reliable human–robot communication when instructions may be incomplete, ambiguous, contextual, or conflicting across different modalities.

---

## System Architecture

```text
                         HUMAN
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       Voice Agent    Gesture Agent   Vision Agent
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                 ┌─────────────────────┐
                 │  Coordinator Agent  │
                 │ Multimodal Fusion & │
                 │ Target Grounding    │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    Memory Agent     │
                 │ Context & Pronouns  │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    Task Planner     │
                 │ Action Decomposition│
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │    Safety Agent     │
                 │ Rules + HITL +      │
                 │ Safety Verification │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Robot Controller   │
                 │       ROS 2         │
                 └─────────────────────┘
```

### Processing Pipeline

```text
Human Input
     ↓
Multimodal Perception
     ↓
Intent & Target Fusion
     ↓
Contextual Reasoning
     ↓
Task Planning
     ↓
Safety Verification
     ↓
Human Confirmation (when required)
     ↓
Robot Execution
```

---

# Key Components

## 1. Voice Agent

The Voice Agent converts spoken instructions into structured commands.

### Processing Pipeline

```text
Audio
  ↓
Audio Preprocessing
  ↓
Voice Activity Detection
  ↓
Whisper ASR
  ↓
Intent & Slot Extraction
  ↓
Structured Voice Output
```

The system supports task-oriented commands such as:

- `pick_and_place`
- `navigate_to`
- `stop_robot`
- `inspect_object`

The structured output can contain:

- Action
- Target object
- Urgency
- Confidence

---

## 2. Gesture Agent

The Gesture Agent interprets hand gestures using MediaPipe Hands and geometric landmark-based classification.

The current implementation supports gestures including:

- `STOP`
- `POINT_LEFT`
- `POINT_RIGHT`
- `POINT_FORWARD`
- `NO_GESTURE`

The classifier uses hand landmark geometry to determine finger extension and pointing direction.

> HaGRIDv2 is treated as an external benchmark/reference dataset. The current real-time gesture classifier is MediaPipe-based and rule-based and is not trained on HaGRIDv2.

---

## 3. Vision Agent

The Vision Agent uses a YOLO-based object detector to identify objects in the camera scene.

The output includes:

- Object class
- Confidence score
- Bounding box
- Spatial sector
- Proximity information

The image is divided into three spatial sectors:

```text
┌──────────┬──────────┬──────────┐
│   LEFT   │  CENTER  │  RIGHT   │
└──────────┴──────────┴──────────┘
```

Spatial information is used by the Coordinator for multimodal target grounding.

---

## 4. Coordinator Agent

The Coordinator performs multimodal fusion of:

- Voice
- Gesture
- Vision

It determines:

- Intended action
- Target object
- Modality agreement or conflict
- Target grounding
- Confidence

The implementation uses a state-based workflow for multimodal coordination.

Possible processing outcomes include:

- `VALID`
- `LOW_CONFIDENCE`
- `MODALITY_CONFLICT`
- `TARGET_NOT_FOUND`
- `INVALID_INPUT`

### Referential Target Grounding

The system can resolve deictic commands such as:

> "Pick it up."

using information from:

- Current visual detections
- Gesture direction
- Object confidence
- Spatial information
- Previous interaction context

When the available evidence is insufficient or ambiguous, the interpretation can be passed to the safety layer for further verification or human confirmation.

---

## 5. Memory Agent

The Memory Agent provides short-term contextual reasoning.

It allows subsequent commands to refer to objects or actions mentioned previously.

Example:

```text
Human: Pick up the apple.
       ↓
Target: apple

Human: Pick it up.
       ↓
"it" → apple
```

The current implementation uses structured conversation history and contextual resolution.

---

## 6. Task Planner

The Task Planner converts interpreted commands into executable task steps.

Supported task types include:

- Navigation
- Pick-and-place
- Inspection
- Stop

The planner separates task interpretation from robot execution, allowing the Safety Agent to verify planned actions before execution.

---

## 7. Safety Agent

The Safety Agent provides a safety verification layer between task planning and robot execution.

It considers factors such as:

- Emergency stop conditions
- Obstacle distance
- Sensor validity
- Target availability
- Perception confidence
- Invalid task states
- Human confirmation requirements

Possible safety outcomes include:

```text
SAFE
STOP
UNSAFE
HUMAN_CONFIRMATION_REQUIRED
```

### Safety Workflow

```text
Task Plan
    ↓
Safety Verification
    ↓
 ┌───────────────────┐
 │ Is execution safe?│
 └─────────┬─────────┘
           │
      ┌────┴────┐
      │         │
     YES        NO
      │         │
      ▼         ▼
 Robot       Block /
Execution    Stop /
             Human Confirmation
```

---

# Technologies Used

| Component | Technology |
|---|---|
| Programming | Python |
| Speech Recognition | Whisper |
| Voice Activity Detection | RMS-based VAD |
| Gesture Recognition | MediaPipe Hands + Geometric Rules |
| Object Detection | YOLO / Ultralytics |
| Multimodal Coordination | LangGraph / State-Based Workflow |
| Context Management | Structured Memory |
| Task Planning | Deterministic Task Planner |
| Safety | Deterministic Rules + HITL |
| Robot Middleware | ROS 2 |
| Simulation | Gazebo |
| Computer Vision | OpenCV |
| Testing | Pytest |
| Version Control | Git / GitHub |

---

# Project Structure

```text
Adaptive-Multimodal-HRI-Agentic-AI/
│
├── src/
│   ├── agents/
│   │   ├── voice/
│   │   ├── gesture/
│   │   ├── vision/
│   │   ├── coordinator/
│   │   ├── memory/
│   │   ├── planner/
│   │   └── safety/
│   │
│   └── pipeline/
│
├── scripts/
│   ├── run_vision.py
│   ├── test_live_vision.py
│   └── test_live_gesture.py
│
├── datasets/
│   ├── Gesture_dataset/
│   └── Speech_commands/
│
├── data/
│   ├── README.md
│   └── coco_subset/
│
├── evaluation/
│   ├── test_cases.json
│   ├── run_evaluation.py
│   ├── metrics.py
│   ├── evaluators/
│   └── results/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

# Dataset and Evaluation

The project uses separate evaluation levels for perception, multimodal interaction, and software verification.

## 1. COCO 2017 Vision Benchmark

A lightweight subset of the COCO 2017 validation dataset is used to evaluate the Vision Agent.

### Dataset Subset

- **41 images**
- **253 ground-truth bounding boxes**
- **7 target object classes**

Target classes:

```text
person
bottle
cup
chair
couch
potted plant
laptop
```

Ground-truth bounding-box annotations are retained for valid object-detection evaluation.

### Vision Evaluation Results

| Metric | Result |
|---|---:|
| Precision | 79.8% |
| Recall | 42.5% |
| Mean IoU | 87.2% |
| mAP@50 | 41.3% |
| mAP@50:95 | 33.6% |

These results correspond specifically to the 41-image COCO validation mini-subset and should not be interpreted as performance over the complete COCO dataset.

---

## 2. Scenario-Based HRI Evaluation

The complete multimodal HRI system is evaluated using **21 predefined scenarios**.

The scenarios cover:

- Multimodal agreement
- Multimodal conflicts
- Target grounding
- Missing targets
- Ambiguity
- Emergency stop
- Obstacle safety
- Low-confidence inputs
- Contextual references
- Human-in-the-loop intervention
- Sensor validity
- Stop-command priority

### Current Evaluation Result

```text
Total scenarios: 21
Passed:          21
Failed:           0
Pass rate:      100%
```

The 100% result represents **scenario-level functional performance on the predefined 21 test cases**. It is not intended to represent general real-world accuracy across arbitrary human–robot interactions.

---

## 3. Software Regression Testing

The project includes an automated regression test suite.

Current result:

```text
153 tests passed
0 tests failed
```

The regression tests verify the implemented modules and help ensure that changes to the project do not break existing functionality.

---

# Evaluation Metrics

The evaluation framework supports metrics appropriate to each subsystem.

### Voice

- Intent accuracy
- Target extraction accuracy
- Word Error Rate (WER) when reference transcripts are available

### Gesture

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Vision

- Precision
- Recall
- Intersection over Union (IoU)
- mAP@50
- mAP@50:95

### Multimodal Fusion

- Target grounding accuracy
- Conflict detection
- Ambiguity handling
- Context/pronoun resolution

### Safety

- Safety decision correctness
- Correct blocking
- Human confirmation handling
- Unsafe-action prevention

### System

- Scenario pass rate
- Task completion
- Software decision-pipeline latency

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Dhivya196/Adaptive-Multimodal-HRI-Agentic-AI.git
cd Adaptive-Multimodal-HRI-Agentic-AI
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Evaluation

Run the 21 scenario evaluation:

```bash
python3 evaluation/run_evaluation.py
```

Evaluation results are saved to:

```text
evaluation/results/
├── evaluation_results.json
├── evaluation_results.csv
└── evaluation_summary.md
```

---

# Running Tests

Run the complete regression test suite:

```bash
pytest tests/
```

For verbose output:

```bash
pytest tests/ -v
```

---

# Vision Evaluation

The COCO mini-subset evaluation can be run using:

```bash
python3 scripts/evaluate_vision_coco.py
```

The evaluation uses the corresponding COCO ground-truth annotations.

Dataset metadata and acquisition information are documented in:

```text
data/README.md
```

Large binary datasets and image files are excluded from GitHub where appropriate.

---

# ROS 2 Integration

The system is designed to interface with a ROS 2-based robot controller.

The control architecture separates the AI decision layer from robot execution:

```text
AI Decision Layer
       ↓
Safety Verification
       ↓
ROS 2 Controller
       ↓
Robot
```

The current implementation supports the software control pipeline and ROS 2 command publishing.

The project should not be interpreted as a complete physical manipulation system. Pick-and-place execution is currently represented through the task-planning and control pipeline, with mock or simulated execution where applicable.

---

# Example Multimodal Interaction

A multimodal interaction can combine speech, gesture, and vision:

```text
Speech:
"Pick it up."

        +

Gesture:
POINT_RIGHT

        +

Vision:
Detected objects in the scene

        ↓

Coordinator
        ↓
Target Grounding
        ↓
Confidence Assessment
        ↓
Memory / Context
        ↓
Task Planner
        ↓
Safety Agent
        ↓
Robot Controller
```

This allows the system to use multiple sources of information rather than relying exclusively on a single modality.

---

# Safety-Aware Decision Making

A central feature of the project is the separation between **task interpretation** and **safety verification**.

Instead of directly executing every interpreted command:

```text
Human Command
      ↓
Multimodal Interpretation
      ↓
Task Planning
      ↓
Safety Verification
      ↓
Robot Execution
```

Uncertain or potentially unsafe situations can be stopped, rejected, or routed through human confirmation before execution.

---

# Limitations

The current prototype has the following limitations:

- The complete HRI evaluation is based on 21 predefined scenarios.
- The COCO evaluation uses a 41-image mini-subset rather than the complete COCO dataset.
- Scenario-level results do not represent statistical generalization to arbitrary real-world interactions.
- The current gesture classifier is MediaPipe/rule based and is not trained on HaGRIDv2.
- WER is not reported without appropriate reference transcripts.
- Physical robot manipulation is not fully implemented.
- Current latency measurements represent the software decision pipeline and do not include complete physical actuator or robot-motion latency.
- Large-scale real-world HRI testing remains future work.

---

# Future Work

Future development may include:

- Larger multimodal HRI datasets
- More diverse real-world environments
- Improved speech and gesture recognition
- Learned multimodal fusion
- Improved uncertainty-aware target grounding
- More comprehensive human-proximity safety monitoring
- Physical robot manipulation experiments
- Hardware-in-the-loop evaluation
- Large-scale user studies
- Expanded ROS 2 and Gazebo integration

---

# Evaluation Summary

| Evaluation Area | Current Result |
|---|---:|
| Vision Dataset | COCO 2017 mini-subset |
| Vision Images | 41 |
| Vision Ground-Truth Boxes | 253 |
| Vision Precision | 79.8% |
| Vision Recall | 42.5% |
| Vision Mean IoU | 87.2% |
| Vision mAP@50 | 41.3% |
| Vision mAP@50:95 | 33.6% |
| HRI Scenarios | 21 |
| HRI Scenarios Passed | 21/21 |
| HRI Scenario Pass Rate | 100% |
| Regression Tests | 153/153 |
| Mean Software Pipeline Latency | 6.53 ms |

---

# Authors

**Dhivya**  
**Sharmista**  
**Jatish**

VIT Chennai  
B.Tech Computer Science and Engineering (AI & Robotics)

---

# Academic Project

This project is developed as an academic/research prototype for studying adaptive multimodal Human–Robot Interaction, agent-based task coordination, and safety-aware decision making.
