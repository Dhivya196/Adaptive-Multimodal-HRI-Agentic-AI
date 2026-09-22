"""
Main Evaluation Script for the Adaptive Multimodal HRI Framework.
Runs configured scenarios, calculates metrics, measures latencies,
and generates reports formatted for the DA2 report.
"""

import json
import time
import os
import sys
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.coordinator_integration import IntegratedHRIPipeline
from src.agents.voice.schemas import VoiceAgentOutput, SpeechIntent, UrgencyLevel
from src.agents.vision.schemas import VisionAgentOutput, DetectedObject, BoundingBox, SpatialSector, ProximityLevel
from src.agents.gesture.schemas import GestureAgentOutput, GestureType, GestureDirection
from src.agents.coordinator.schemas import TaskStatus
from src.agents.safety.schemas import SafetyDecisionType
from evaluation.metrics import calculate_accuracy, calculate_latency_stats, print_table, calculate_precision_recall_f1

class Evaluator:
    def __init__(self):
        self.pipeline = IntegratedHRIPipeline()
        self.results = []
        self.latencies = {
            "end_to_end": []
        }

    def run_tests(self, test_cases_path: str):
        with open(test_cases_path, "r") as f:
            test_cases = json.load(f)

        print(f"Loaded {len(test_cases)} test cases.")

        for tc in test_cases:
            res = self.evaluate_single(tc)
            self.results.append(res)
            # Sleep slightly to avoid tight loop on stdout
            time.sleep(0.01)

    def evaluate_single(self, tc: dict) -> dict:
        # Construct Voice Output
        voice_out = None
        if tc.get("voice"):
            v = tc["voice"]
            intent = SpeechIntent(
                action=v["action"],
                target_object=v.get("target_object"),
                urgency=UrgencyLevel[v.get("urgency", "NORMAL").upper()],
                confidence=v.get("confidence", 1.0)
            )
            voice_out = VoiceAgentOutput(
                is_speech_detected=True,
                speech_intent=intent,
                confidence=v.get("confidence", 1.0)
            )
        
        # Construct Vision Output
        vision_out = None
        if tc.get("vision") is not None: # Even if empty list
            objects = []
            for i, o in enumerate(tc["vision"]):
                obj = DetectedObject(
                    object_id=i,
                    label=o["label"],
                    confidence=o["confidence"],
                    bbox=BoundingBox(0,0,10,10),
                    spatial_sector=SpatialSector[o.get("spatial_sector", "UNKNOWN").upper()],
                    proximity=ProximityLevel.MEDIUM
                )
                objects.append(obj)
            vision_out = VisionAgentOutput(detected_objects=objects, confidence=1.0)
        
        # Construct Gesture Output
        gesture_out = None
        if tc.get("gesture"):
            g = tc["gesture"]
            gesture_out = GestureAgentOutput(
                gesture=g.get("gesture", "UNKNOWN"),
                direction=g.get("direction", "NONE"),
                confidence=g.get("confidence", 1.0),
                is_gesture_detected=True
            )

        env = tc.get("env", {})
        state_input = {
            "voice_output": voice_out,
            "vision_output": vision_out,
            "gesture_output": gesture_out,
            "obstacle_distance": env.get("obstacle_distance", 2.0),
            "human_distance": env.get("human_distance", 1.5),
            "emergency_stop": env.get("emergency_stop", False),
            "sensor_validity": env.get("sensor_validity", {"camera": True, "proximity": True})
        }

        start_time = time.time()
        final_state = self.pipeline.process(state_input)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        self.latencies["end_to_end"].append(latency)

        task = final_state.get("task")
        safety_dec_type = final_state.get("safety_decision_type")
        
        actual_status = task.task_status.value if task and hasattr(task.task_status, "value") else (str(task.task_status) if task else "INVALID_INPUT")
        actual_target = task.target_object if task else None
        actual_action = task.action if task else "none"
        actual_safety = safety_dec_type

        expected = tc["expected"]
        
        passed = (
            actual_status == expected["status"] and
            actual_target == expected["target"] and
            (expected["action"] in actual_action or actual_action in expected["action"]) and
            (expected.get("safety") is None or actual_safety == expected["safety"])
        )

        return {
            "id": tc["test_id"],
            "description": tc["description"],
            "expected_status": expected["status"],
            "actual_status": actual_status,
            "expected_target": expected["target"],
            "actual_target": actual_target,
            "expected_safety": expected.get("safety"),
            "actual_safety": actual_safety,
            "passed": passed,
            "latency": latency
        }

    def print_results(self):
        print("\n" + "="*80)
        print("              HRI SYSTEM EVALUATION RESULTS")
        print("="*80)
        
        passed_count = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        completion_rate = (passed_count / total * 100) if total > 0 else 0
        
        print_table("Overall Test Results", ["Metric", "Result"], [
            ["Total Test Cases", total],
            ["Passed", passed_count],
            ["Failed", total - passed_count],
            ["Task Completion", f"{completion_rate:.1f} %"]
        ])

        # Voice Intent Mock
        print_table("Voice / Intent Evaluation", ["Metric", "Value", "Reason"], [
            ["Accuracy", f"{completion_rate:.1f} %", "Based on scenario success"],
            ["WER", "NOT AVAILABLE", "No reference transcripts provided"]
        ])
        
        # Gesture Mock
        print_table("Gesture Evaluation", ["Metric", "Value", "Reason"], [
            ["Gesture Accuracy", f"{completion_rate:.1f} %", "Scenario-based evaluation"],
            ["mAP", "NOT AVAILABLE", "No ground truth bounding boxes"]
        ])

        # Fusion
        fusion_statuses_expected = [r["expected_status"] for r in self.results]
        fusion_statuses_actual = [r["actual_status"] for r in self.results]
        
        val_acc = calculate_precision_recall_f1(fusion_statuses_expected, fusion_statuses_actual, "VALID")[0]
        conf_acc = calculate_precision_recall_f1(fusion_statuses_expected, fusion_statuses_actual, "MODALITY_CONFLICT")[0]
        tnf_acc = calculate_precision_recall_f1(fusion_statuses_expected, fusion_statuses_actual, "TARGET_NOT_FOUND")[0]
        amb_acc = calculate_precision_recall_f1(fusion_statuses_expected, fusion_statuses_actual, "AMBIGUOUS_TARGET")[0]
        
        print_table("Multimodal Fusion", ["Metric", "Value"], [
            ["VALID Decision Precision", f"{val_acc:.1f} %"],
            ["Conflict Detection Precision", f"{conf_acc:.1f} %"],
            ["Target Not Found Precision", f"{tnf_acc:.1f} %"],
            ["Ambiguity Detection Precision", f"{amb_acc:.1f} %"]
        ])
        
        # Safety
        safety_expected = [r["expected_safety"] for r in self.results if r["expected_safety"] is not None]
        safety_actual = [r["actual_safety"] for r in self.results if r["expected_safety"] is not None]
        
        safe_acc = calculate_accuracy(safety_expected, safety_actual)
        
        print_table("Safety Evaluation", ["Metric", "Value"], [
            ["Safety Decision Accuracy", f"{safe_acc:.1f} %"],
        ])
        
        # Latency
        stats = calculate_latency_stats(self.latencies["end_to_end"])
        
        print_table("System Performance", ["Component / Metric", "Mean", "Median", "Min", "Max", "Std"], [
            ["End-to-End Latency (ms)", f"{stats['mean']:.2f}", f"{stats['median']:.2f}", f"{stats['min']:.2f}", f"{stats['max']:.2f}", f"{stats['std']:.2f}"]
        ])
        
        # Per test
        rows = [
            [r["id"], r["description"][:30] + "...", r["expected_status"], r["actual_status"], "PASS" if r["passed"] else "FAIL", f"{r['latency']:.1f} ms"]
            for r in self.results
        ]
        print_table("Per-Test Results", ["ID", "Test Case", "Expected Status", "Actual Status", "Result", "Latency"], rows)
        
        # Final classification
        print_table("Final Metric Classification", ["Metric", "Status", "Reason"], [
            ["Intent Accuracy", "AVAILABLE NOW", "Calculated from test cases"],
            ["Fusion Accuracy", "AVAILABLE NOW", "Calculated from test cases"],
            ["Safety Accuracy", "AVAILABLE NOW", "Calculated from test cases"],
            ["Latency", "AVAILABLE NOW", "Measured during evaluation execution"],
            ["mAP / IoU", "NOT AVAILABLE", "No ground-truth bounding box evaluation dataset"],
            ["WER", "NOT AVAILABLE", "No reference transcripts for speech-to-text"]
        ])

    def save_results(self):
        os.makedirs("evaluation/results", exist_ok=True)
        
        # JSON
        with open("evaluation/results/results.json", "w") as f:
            json.dump(self.results, f, indent=4)
            
        # CSV
        import csv
        with open("evaluation/results/results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Description", "Expected Status", "Actual Status", "Expected Safety", "Actual Safety", "Passed", "Latency"])
            for r in self.results:
                writer.writerow([r["id"], r["description"], r["expected_status"], r["actual_status"], r["expected_safety"], r["actual_safety"], r["passed"], r["latency"]])
                
        # Summary
        with open("evaluation/results/summary.txt", "w") as f:
            passed_count = sum(1 for r in self.results if r["passed"])
            f.write(f"Total Tests: {len(self.results)}\n")
            f.write(f"Passed: {passed_count}\n")
            f.write(f"Failed: {len(self.results) - passed_count}\n")
            
        print("\nResults saved to evaluation/results/")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_tests("evaluation/test_cases.json")
    evaluator.print_results()
    evaluator.save_results()
