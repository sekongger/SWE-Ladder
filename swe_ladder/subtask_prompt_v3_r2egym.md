You label ONE step of a software-engineering agent trajectory with its current SUBTASK.
The label decides which privileged teacher evidence scores this step's tokens, so label by the PURPOSE of the action in the plan, not by the tool name and not by what the thought text rehearses.

Classes (privileged context each implies):
- LOCATE   = identifying WHICH file/hunk causally explains the issue: repo-wide search, viewing candidate files, grepping for the bug's origin. Teacher evidence: gold changed files/hunks.
- REPRODUCE = confirming the bug reproduces BEFORE any source fix: authoring OR running a reproduction/failing-test script. Teacher evidence: gold repro/test diff.
- PATCH   = crafting, applying, or repairing the SOURCE fix. Includes any search/view whose purpose is to make a source edit succeed (recovering exact old_str, reading context around the target line) once the causal location is known. Teacher evidence: gold repair patch.
- VERIFY  = confirming the fix works AFTER a successful source edit: re-running repro/tests, repairing the test/verification harness itself, or the closing submit/review message. Teacher evidence: none needed / self-check.

Scaffold note: the editor tool may be named str_replace_editor (OpenHands) or file_editor (R2E-Gym); sub-commands (view/create/str_replace/insert) are identical - treat the two names as the same tool everywhere.

Boundary rules (resolve ALL known confusions):
R1. After the causal file/hunk is known, greps/views that serve a pending or failed SOURCE edit are PATCH, never LOCATE — even if the thought says "find/search/locate".
R2. Creating a new script whose purpose is to exhibit/confirm the bug, before the first successful SOURCE edit, is REPRODUCE — even if the thought also states the fix plan.
R3. Running the repro/test script AFTER a successful source edit is VERIFY — even if the run fails, and even if the thought says "Phase 7 VERIFICATION" or "verify current behavior" ambiguously: pre-fix runs are REPRODUCE, post-fix runs are VERIFY.
R4. The first SOURCE edit attempt ends REPRODUCE and starts PATCH — even if the thought still talks about the reproduction script being "ready".
R5. After a successful source edit, editing/repairing the TEST or repro script itself (not source) is VERIFY, not PATCH.
R6. A submit/finish/closing-summary message is VERIFY.
R7. Views/greps/finds BEFORE the first source edit are LOCATE, unless R1 applies (they follow a failed edit on a known file).

Examples:
thought: "search repo-wide for files mentioning the broken function" | find/grep repo -> LOCATE
thought: "now I see the issue; write a script that triggers the TypeError" | create reproduce_issue.py -> REPRODUCE
thought: "run the reproduction script to confirm current behavior" (no source edit yet) | python3 reproduce_issue.py -> REPRODUCE
thought: "str_replace failed; get the exact current text of line 687 for old_str" | grep -n "for p in ignore" source.py -> PATCH
thought: "the reproduction script is ready; apply the fix" | str_replace source.py (first attempt) -> PATCH
thought: "edit applied; re-run my reproduce script" (after successful edit) | python3 reproduce_issue.py -> VERIFY
thought: "the test harness itself is wrong; fix the test setup" (after source edit) | str_replace test_x.py -> VERIFY
thought: "the fix has been successfully implemented" | submit -> VERIFY

Recent context and the step to label follow. Answer with the single class word only.
