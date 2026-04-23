from __future__ import annotations

from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pawpal_ai.agent import PawPalAgent
from pawpal_ai.models import Owner


def load_owner() -> Owner:
    return Owner.load_from_json(str(BASE_DIR / "data" / "sample_owner.json"))


def main() -> None:
    cases = json.loads((BASE_DIR / "eval" / "eval_cases.json").read_text(encoding="utf-8"))["cases"]
    agent = PawPalAgent(str(BASE_DIR / "data" / "pet_care_kb.json"))

    passed = 0
    confidence_total = 0.0
    results = []

    for case in cases:
        owner = load_owner()
        response = agent.run(case["query"], owner)
        expect = case["expect"]
        ok = True
        reasons = []

        if "intent" in expect and response.intent != expect["intent"]:
            ok = False
            reasons.append(f"intent expected {expect['intent']} got {response.intent}")

        if "guardrail_status" in expect and response.guardrail_status != expect["guardrail_status"]:
            ok = False
            reasons.append(
                f"guardrail expected {expect['guardrail_status']} got {response.guardrail_status}"
            )

        if "blocked" in expect and response.blocked != expect["blocked"]:
            ok = False
            reasons.append(f"blocked expected {expect['blocked']} got {response.blocked}")

        if "answer_contains" in expect and expect["answer_contains"] not in response.answer:
            ok = False
            reasons.append(f"answer missing substring '{expect['answer_contains']}'")

        if "proposed_tasks_min" in expect and len(response.proposed_tasks) < expect["proposed_tasks_min"]:
            ok = False
            reasons.append("not enough proposed tasks")

        if "tool_key" in expect and expect["tool_key"] not in response.tool_outputs:
            ok = False
            reasons.append(f"tool output missing key {expect['tool_key']}")

        if ok:
            passed += 1
        confidence_total += response.confidence
        results.append(
            {
                "case": case["name"],
                "passed": ok,
                "confidence": response.confidence,
                "reasons": reasons,
            }
        )

    print("PawPal+ AI Assistant Evaluation")
    print("==============================")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['case']} | confidence={result['confidence']:.2f}")
        for reason in result["reasons"]:
            print(f"  - {reason}")

    total = len(results)
    average_confidence = confidence_total / total if total else 0.0
    print()
    print(f"Summary: {passed}/{total} passed")
    print(f"Average confidence: {average_confidence:.2f}")


if __name__ == "__main__":
    main()
