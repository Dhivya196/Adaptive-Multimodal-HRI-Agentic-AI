# Adaptive Multimodal Human-Robot Communication Using a Multi-Agent AI Framework with Safety-Aware Decision Making

> An Agentic AI framework that enables robots to understand human commands through speech, gestures, and vision while ensuring safe and explainable decision-making.

---

## 📖 Project Overview

Human-Robot Interaction (HRI) plays a vital role in enabling intuitive communication between humans and intelligent robotic systems. Most existing HRI systems rely on a single communication modality, such as speech or vision, making them vulnerable to noisy environments, ambiguous commands, and unsafe task execution.

This project proposes a lightweight **Multi-Agent AI Framework** that combines multiple intelligent agents to understand human intentions using **speech recognition, gesture recognition, and computer vision**. Before executing any action, a dedicated **Safety Agent** evaluates the environment to prevent unsafe operations. Additional agents such as **Memory**, **Task Planner**, and **Explainability** improve contextual understanding, autonomous planning, and user trust.

---

## 🎯 Objectives

- Develop a **Voice Agent** for speech command recognition.
- Develop a **Gesture Agent** for pointing and hand gesture recognition.
- Develop a **Vision Agent** for object detection and scene understanding.
- Design a **Coordinator Agent** to fuse information from multiple agents.
- Develop a **Safety Agent** to evaluate environmental risks before execution.
- Develop a **Memory Agent** to store user preferences and previous interactions.
- Develop a **Task Planner Agent** for autonomous task decomposition and planning.
- Develop an **Explainability Agent** to provide human-understandable reasoning for robot decisions.
- Compare multimodal interaction with voice-only interaction in terms of accuracy and safety.

---

## 🏗️ Proposed Architecture

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
                    Explainability Agent
                               │
                               ▼
                      Robot Controller
                               │
                               ▼
                          Robot Action
```

---

## 🤖 Multi-Agent Framework

| Agent | Responsibility |
|--------|----------------|
| 🎤 Voice Agent | Converts speech into text and extracts user commands. |
| ✋ Gesture Agent | Detects pointing direction and hand gestures. |
| 👁️ Vision Agent | Detects objects, obstacles, and environmental context. |
| 🧠 Coordinator Agent | Combines outputs from all perception agents to determine user intent. |
| 🛡️ Safety Agent | Validates whether the requested action is safe before execution. |
| 🧾 Memory Agent | Stores previous commands, user preferences, and object locations. |
| 📋 Task Planner Agent | Generates an ordered sequence of actions required to complete the task. |
| 💬 Explainability Agent | Explains why the robot executed or rejected a command. |

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- YOLOv8
- Whisper / Vosk (Speech Recognition)
- LangChain
- LangGraph
- CrewAI
- Ollama / Llama 3
- ROS2 (Future Integration)

---

## 📂 Repository Structure

```
Adaptive-Human-Robot-Communication/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── docs/
│   ├── DA1_Report.pdf
│   ├── Project_Overview.pdf
│   ├── Literature_Survey.pdf
│   └── Architecture.png
│
├── datasets/
│   ├── README.md
│   ├── speech/
│   ├── gesture/
│   ├── vision/
│   └── safety/
│
├── papers/
│   ├── README.md
│   └── references.bib
│
├── src/
│   ├── voice_agent/
│   ├── gesture_agent/
│   ├── vision_agent/
│   ├── coordinator_agent/
│   ├── memory_agent/
│   ├── task_planner/
│   ├── safety_agent/
│   ├── explainability_agent/
│   └── utils/
│
├── models/
│
├── results/
│
├── images/
│   ├── architecture.png
│   └── workflow.png
│
└── presentation/
    └── DA1_Presentation.pptx
```

---

## 📊 Datasets

| Module | Dataset |
|--------|---------|
| Speech Recognition | Google Speech Commands Dataset |
| Gesture Recognition | HaGRID Dataset |
| Object Detection | COCO Dataset |
| Safety Detection | Custom Dataset (Obstacle and Unsafe Object Scenarios) |

*Large datasets are not stored in this repository. Download instructions are provided in the `datasets/README.md` file.*

---

## 📈 Evaluation Metrics

- Speech Recognition Accuracy
- Gesture Recognition Accuracy
- Object Detection Accuracy (mAP)
- Command Interpretation Accuracy
- Safety Detection Accuracy
- Response Time
- Task Completion Rate
- User Satisfaction

---

## 🚀 Future Enhancements

- ROS2 integration with physical robots
- Edge deployment on NVIDIA Jetson/Raspberry Pi
- Reinforcement learning for adaptive behaviour
- Personalized interaction using long-term memory
- Emotion-aware human-robot communication
- Multi-robot collaboration

---

## 📚 References

The literature survey consists of **18 research papers**, including **12 papers published between 2023–2026**, covering:

- Human-Robot Interaction (HRI)
- Multimodal Communication
- Agentic AI
- Multi-Agent Systems
- Computer Vision
- Gesture Recognition
- Safety-aware Robotics
- Task Planning

The complete bibliography is available in **papers/references.bib**.

---

## 👨‍💻 Team

**Project Title:** Adaptive Multimodal Human-Robot Communication Using a Multi-Agent AI Framework with Safety-Aware Decision Making

Developed as part of the **Digital Assignment (DA1)** for the Artificial Intelligence / Robotics curriculum.

---

## 📄 License

This project is released under the MIT License.
