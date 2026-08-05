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
|  Voice Agent | Converts speech into text and extracts user commands. |
|  Gesture Agent | Detects pointing direction and hand gestures. |
|  Vision Agent | Detects objects, obstacles, and environmental context. |
|  Coordinator Agent | Combines outputs from all perception agents to determine user intent. |
|  Safety Agent | Validates whether the requested action is safe before execution. |
|  Memory Agent | Stores previous commands, user preferences, and object locations. |
|  Task Planner Agent | Generates an ordered sequence of actions required to complete the task. |
|  Safety Agent | Ensures that the execution would be safe. |

---

##  Technologies Used

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

##  Future Enhancements

- ROS2 integration with physical robots
- Edge deployment on NVIDIA Jetson/Raspberry Pi
- Reinforcement learning for adaptive behaviour
- Personalised interaction using long-term memory
- Emotion-aware human-robot communication
- Multi-robot collaboration

---

##  License

This project is released under the MIT License.
