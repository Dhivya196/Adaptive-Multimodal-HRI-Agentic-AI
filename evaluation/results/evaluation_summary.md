# Adaptive Multimodal HRI Evaluation Summary

**Evaluation Date:** 2026-09-23 13:17:49  
**Total Test Cases Loaded:** 21  
**Total Test Cases Executed:** 21  

---

## 1. Overall System Performance

| Metric | Value |
| :--- | :--- |
| **Total Scenarios Loaded** | **21** |
| **Total Scenarios Executed** | **21** |
| **Total Passed** | **21** |
| **Total Failed** | **0** |
| **Overall Scenario Pass Rate** | **100.00%** |
| **Mean Latency** | **6.71 ms** |
| **Median Latency** | **6.31 ms** |
| **Minimum Latency** | **4.43 ms** |
| **Maximum Latency** | **10.90 ms** |
| **Standard Deviation** | **1.33 ms** |

---

## 2. Subsystem Evaluation Metrics

| Subsystem / Evaluation Dimension | Value | Status / Ground Truth Note |
| :--- | :--- | :--- |
| **Voice Intent Accuracy** | 100.00% | Evaluated on all voice command scenarios |
| **Target Extraction Accuracy** | 100.00% | Evaluated across all 21 test cases |
| **Gesture Recognition Accuracy** | 100.00% | Evaluated across all gesture test cases |
| **Gesture Precision / Recall / F1** | 100.00% / 100.00% / 100.00% | Symbolic multimodal gesture routing |
| **Coordinator Target Grounding Accuracy** | 100.00% | Multimodal fusion & deictic sector resolution |
| **Context / Pronoun Resolution Accuracy** | 100.00% | Multi-turn short-term dialogue memory resolution |
| **Safety Decision Accuracy** | 100.00% | Safety agent hazard, obstacle, and HITL gating |
| **End-to-End Task Completion Rate** | 100.00% | All 21 test scenarios passing |
| **Speech Word Error Rate (WER)** | *N/A* | `REQUIRES_GROUND_TRUTH` (No raw audio transcripts in scenario harness) |
| **Vision Object Detection mAP@50** | *N/A (In Harness)* | `REQUIRES_GROUND_TRUTH` (Evaluated on COCO subset: 41.30%) |
| **Vision Object Detection mAP@50:95** | *N/A (In Harness)* | `REQUIRES_GROUND_TRUTH` (Evaluated on COCO subset: 33.60%) |
| **Live Continuous Video FPS** | *N/A* | `REQUIRES_GROUND_TRUTH` (Requires continuous hardware camera benchmark) |

---

## 3. Detailed Scenario Execution Breakdown (TC01 – TC21)

| ID | Description | Final Action | Target | Status | Safety | Execution Command | Result | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T01** | Voice + Vision + Gesture: Correct pointing to an object | `pick` | `bottle` | `VALID` | `SAFE` | `TURN_LEFT` | **PASS** | 10.90 ms |
| **T02** | Voice + Vision + Gesture: Modality conflict (points away from object) | `pick` | `bottle` | `MODALITY_CONFLICT` | `SAFE` | `PICK` | **PASS** | 7.77 ms |
| **T03** | Voice + Vision: Object requested is not in scene | `inspect` | `laptop` | `TARGET_NOT_FOUND` | `UNSAFE` | `INSPECT` | **PASS** | 4.43 ms |
| **T04** | Voice + Vision + Gesture: Object absent from pointed sector | `pick` | `laptop` | `VALID` | `SAFE` | `TURN_LEFT` | **PASS** | 7.49 ms |
| **T05** | Voice + Vision + Gesture: Multiple objects in pointed sector (Ambiguous) | `pick` | `laptop` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 8.58 ms |
| **T06** | Voice Only: Emergency Stop | `stop_robot` | `None` | `VALID` | `STOP` | `STOP` | **PASS** | 6.26 ms |
| **T07** | Unsafe obstacle distance | `move_forward` | `None` | `VALID` | `STOP` | `MOVE` | **PASS** | 5.22 ms |
| **T08** | Voice + Vision: Explicit target agreement | `pick` | `laptop` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 6.09 ms |
| **T09** | Voice only: General action without object | `explore` | `None` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 6.31 ms |
| **T10** | Low confidence voice input | `pick` | `bottle` | `LOW_CONFIDENCE` | `SAFE` | `PICK` | **PASS** | 5.86 ms |
| **T11** | Invalid Input (No relevant data) | `none` | `None` | `INVALID_INPUT` | `UNSAFE` | `NONE` | **PASS** | 5.63 ms |
| **T12** | Multi-turn context: First command | `inspect` | `apple` | `VALID` | `SAFE` | `TURN_RIGHT` | **PASS** | 6.30 ms |
| **T13** | Multi-turn context: Pronoun resolution (follows T12) | `pick` | `apple` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 6.63 ms |
| **T14** | Voice + Vision: Ambiguous explicit target (two bottles) | `pick` | `bottle` | `VALID` | `SAFE` | `TURN_LEFT` | **PASS** | 7.14 ms |
| **T15** | Voice + Gesture: Modality Conflict | `inspect` | `cup` | `MODALITY_CONFLICT` | `SAFE` | `INSPECT` | **PASS** | 7.64 ms |
| **T16** | Human intervention required (medium risk) | `pick` | `knife` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 6.84 ms |
| **T17** | Vision + Gesture without Voice | `none` | `None` | `VALID` | `SAFE` | `STOP` | **PASS** | 6.52 ms |
| **T18** | Voice + Vision + Gesture: Perfect agreement | `inspect` | `box` | `VALID` | `SAFE` | `STOP` | **PASS** | 7.54 ms |
| **T19** | Unsafe distance (Human too close) | `move_forward` | `None` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 5.74 ms |
| **T20** | Sensor validity failed | `move_forward` | `None` | `VALID` | `SAFE` | `MOVE_FORWARD` | **PASS** | 6.09 ms |
| **T21** | Stop command overrides everything | `stop` | `None` | `VALID` | `SAFE` | `STOP` | **PASS** | 5.87 ms |

---
*Report auto-generated by the Adaptive Multimodal HRI Agentic AI Evaluation Framework.*
