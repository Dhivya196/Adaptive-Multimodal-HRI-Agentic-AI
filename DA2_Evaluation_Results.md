# Section 4: Results and Evaluation Metrics

## 4.1 Experimental Setup
The Adaptive Multimodal HRI system was evaluated using an automated, scenario-based testing framework. The framework is designed to test the system's end-to-end integration—including Voice, Vision, Gesture, Memory, Coordinator, Task Planner, Safety, and Controller components. 

The evaluation consists of 21 carefully designed multimodal test cases representing real-world human-robot interaction scenarios. These scenarios cover various combinations of modalities (e.g., Voice only, Voice + Vision, Voice + Vision + Gesture), spatial conflicts (e.g., pointing in the wrong direction), ambiguity (multiple objects in a sector), safety violations, and multi-turn contextual references (pronoun resolution).

The evaluation executes the actual `IntegratedHRIPipeline` orchestrated by LangGraph. Due to the absence of large-scale ground-truth bounding-box and transcribed audio datasets tailored to the robot's specific operating environment, certain traditional ML metrics (such as object detection mAP/IoU and Word Error Rate) were excluded from this evaluation. Instead, the focus is on the decision-making accuracy, fusion precision, and end-to-end latency of the integrated system.

## 4.2 Overall System Performance
The system was evaluated against the 21 interaction scenarios. A test case is marked as "Passed" if the system correctly extracts the intent, accurately associates the target across modalities, appropriately enforces safety rules, and generates the correct final action state.

* **Total Test Cases**: 21
* **Passed**: 21
* **Failed**: 0
* **System Task Completion Rate**: 100.0%

## 4.3 Multimodal Fusion Evaluation
The core capability of the `CoordinatorAgent` is to fuse inputs from Voice, Vision, and Gesture. The system successfully cross-correlates pointing gestures with detected objects and voice targets, handling ambiguities and conflicts robustly.

| Metric | Precision |
| :--- | :--- |
| **VALID Decision Precision** | 100.0 % |
| **Conflict Detection Precision** | 100.0 % |
| **Target Not Found Precision** | 100.0 % |

*The system correctly detected 100% of modality conflicts (e.g., user asks for "cup" but points at a sector containing only a "laptop").*

## 4.4 Intent and Gesture Recognition
The evaluation validates the pipeline's ability to parse and route structured intents and spatial gestures accurately under varying confidence levels.

| Metric | Accuracy | Reason |
| :--- | :--- | :--- |
| **Voice Intent Accuracy** | 100.0 % | Based on scenario-based intent extraction and routing success. |
| **Gesture Classification Accuracy** | 100.0 % | Based on scenario-based spatial grounding success. |

*(Note: Traditional model-level benchmarks such as WER and mAP are left for future work when a dedicated ground-truth evaluation dataset is collected).*

## 4.5 Memory and Context Grounding
The `MemoryAgent` successfully handled multi-turn context retention. In scenarios where a user gave sequential commands (e.g., "Inspect the apple" followed by "Pick it up"), the system accurately resolved the referential pronoun ("it") to the grounded object ("apple") from previous conversational turns without requiring visual or gesture re-confirmation.

## 4.6 Safety Evaluation
The `SafetyAgent` sits as the final checkpoint before robot execution. The evaluation tested emergency stops, obstacle proximity violations, and sensor failures.

| Metric | Result |
| :--- | :--- |
| **Safety Decision Accuracy** | 100.0 % |

The Safety Agent successfully blocked 100% of intentionally unsafe commands (e.g., issuing a "move_forward" command when human proximity was dangerously close at 0.2 meters, or when sensor validity failed).

## 4.7 End-to-End Latency
System responsiveness is critical for real-time HRI. Latency was measured across the entire LangGraph execution pipeline (from multi-agent input reception to final controller action decision) bypassing hardware execution times.

| Metric | Latency (ms) |
| :--- | :--- |
| **Mean** | 6.96 ms |
| **Median** | 4.55 ms |
| **Minimum** | 2.98 ms |
| **Maximum** | 48.07 ms |
| **Standard Deviation** | 9.36 ms |

The sub-10ms median processing time demonstrates that the modular LangGraph architecture introduces negligible orchestration overhead, making it highly suitable for real-time robotic applications.
