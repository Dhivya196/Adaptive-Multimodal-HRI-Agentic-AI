# Adaptive Multimodal Human-Robot Communication Using a Multi-Agent AI Framework with Safety-Aware Decision Making

> An Agentic AI framework that enables robots to understand human commands through speech, gestures, and vision while ensuring safe decision-making.

---

## Project Overview

Human-Robot Interaction (HRI) plays a vital role in enabling intuitive communication between humans and intelligent robotic systems. Most existing HRI systems rely on a single communication modality, such as speech or vision, making them vulnerable to noisy environments, ambiguous commands, and unsafe task execution.

This project proposes a lightweight **Multi-Agent AI Framework** that combines multiple intelligent agents to understand human intentions using **speech recognition, gesture recognition, and computer vision**. Before executing any action, a dedicated **Safety Agent** evaluates the environment to prevent unsafe operations. Additional agents such as **Memory**, **Task Planner** and **Safety** improve contextual understanding, autonomous planning, and user trust.

---

##  Objectives

- Develop a **Voice Agent** for speech command recognition.
- Develop a **Gesture Agent** for pointing and hand gesture recognition.
- Develop a **Vision Agent** for object detection and scene understanding.
- Design a **Coordinator Agent** to fuse information from multiple agents.
- Develop a **Safety Agent** to evaluate environmental risks before execution.
- Develop a **Memory Agent** to store user preferences and previous interactions.
- Develop a **Task Planner Agent** for autonomous task decomposition and planning.
- Develop a **Safety Agent** for safe execution of instructions in a human-robot proximity environment
- Compare multimodal interaction with voice-only interaction in terms of accuracy and safety.

---

##  Proposed Architecture

```
                   Human User
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
  Voice Input     Hand Gesture      Camera Feed
      │                 │                 │
      ▼                 ▼                 ▼
 Voice Agent      Gesture Agent     Vision Agent
      │                 │                 │
      └─────────────────┼─────────────────┘
                        ▼
                Coordinator Agent
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
  Memory Agent   Task Planner     Safety Agent
                        │                │
                        └──────┬─────────┘
                               ▼
                      Robot Controller
                               │
                               ▼
                          Robot Action
```

---

##  Multi-Agent Framework

| Agent | Responsibility |
|--------|----------------|
|  Voice Agent | Converts speech into text and extracts structured user commands, target objects, and urgency. |
|  Gesture Agent | Uses a pretrained HaGRID gesture detection model for visual gesture recognition. Detected gestures are converted into the project's `GestureType` schema, while spatial grounding is performed at the application level for directional pointing (`POINT_LEFT`, `POINT_FORWARD`, `POINT_RIGHT`). |
|  Vision Agent | Detects objects, obstacles, spatial sectors, and visual proximity context using YOLO. |
|  Coordinator Agent | Fuses multimodal outputs across modalities (LangGraph StateGraph) to determine unified human intent. |
|  Memory Agent | Stores conversational history and resolves referential ambiguity (e.g., pronoun grounding). |
|  Task Planner Agent | Decomposes high-level multimodal tasks into primitive executable robot action sequences. |
|  Safety Agent | Multi-tier safety guard evaluating deterministic hard constraints, proximity hysteresis, LLM contextual risks, and human-in-the-loop review. |
|  Robot Controller | Translates safety-approved plans into hardware velocity/discrete actions (ROS2 / Simulation). |

---

##  Technologies Used

- Python
- OpenCV
- MediaPipe
- YOLO26n / YOLOv8
- Whisper / Vosk (Speech Recognition)
- LangChain
- LangGraph
- CrewAI
- Ollama / Llama 3
- ROS2 (Future Integration)

---

##  Datasets

| Module | Dataset |
|--------|---------|
| Speech Recognition | Google Speech Commands Dataset |
| Gesture Recognition | HaGRID Dataset |
| Object Detection | COCO Dataset |
| Safety Detection | Custom Dataset (Obstacle and Unsafe Object Scenarios) |

*Large datasets are not stored in this repository. Download instructions are provided in the `datasets/README.md` file.*

---

##  Evaluation Metrics

- Speech Recognition Accuracy
- Gesture Recognition Accuracy
- Object Detection Accuracy (mAP)
- Command Interpretation Accuracy
- Safety Detection Accuracy
- Response Time
- Task Completion Rate
- User Satisfaction

---

## 📁 Project Structure

```text
Adaptive-Multimodal-HRI-Agentic-AI/
├── configs/                       # Configuration files (.yaml / .json)
├── datasets/                      # Speech, gesture, and vision datasets
│   ├── Gesture_dataset/
│   └── Speech_commands/
├── report/                        # Architecture reports and system design documents
├── scripts/                       # Executable scripts and CLI utilities
├── src/
│   ├── agents/
│   │   └── vision/                # Vision perception agent module
│   ├── common/                    # Shared schemas, utilities, and exceptions
│   ├── pipeline/                  # End-to-end multi-agent pipelines
│   └── utils/                     # Device drivers, cameras, and helpers
├── tests/                         # Unit and integration test suite
├── .gitignore                     # Git ignore file
├── requirements.txt               # Project dependencies
└── README.md
```

---

## 🔮 Future Roadmap

- Vision Agent (YOLO26n object detection & spatial scene grounding)
- Gesture Agent (YOLO Pose / MediaPipe body & hand gesture perception)
- Voice Agent (Whisper Speech-to-Text)
- Coordinator Agent (Multimodal LLM Intent Fusion)
- Contextual Memory Agent (Vector embeddings & retrieval)
- Task Planner Agent (Constrained robot action sequencing)
- Safety Agent (Deterministic rules & LLM contextual risk assessment)
- ROS2 Integration (Robot controller execution layer)

---

## 📄 License

This project is released under the MIT License.


