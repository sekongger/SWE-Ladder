## Stage
{stage}

## Issue statement
{issue_statement}

## How to use the hints
The hints below are background issue context. They may describe symptoms, constraints, or related observations that help you understand this specific problem. Use them to calibrate your reasoning, but do not treat them as proof of the bug location or as the answer itself. If the hints are empty, continue from the issue statement and repository context.

## Hints
{hints_text}
## localized_buggy_files
The localized buggy files are the output of the previous localization stage. Treat them as the main starting point for this stage.

Use this information to:
- decide which source files to inspect first;
- identify the likely class, function, method, or line range involved in the bug;
- construct a reproduction that directly exercises the localized behavior;
- keep the patch minimal and limited to relevant non-test source files.
Do not treat localized_buggy_files as a substitute for code inspection.
Before creating the reproduction or patch, inspect the relevant localized source file with tools.

Do not perform broad file search unless:
- the localized file is missing;
- the localized file clearly does not match the issue;
- the reproduction cannot be constructed from the localized context.
{localized_buggy_files_json}

## {teacher_evidence_heading}
The following repository-specific repair context may help you solve the issue more efficiently.

Use it as guidance for understanding the intended source-level behavior and the likely repair direction. It is not a substitute for inspecting the repository, creating or validating the reproduction script, applying the patch yourself, and verifying the result.

Do not mention this context in the final result. Do not modify existing tests.
{teacher_evidence_json}

IMPORTANT TIP:
1. When viewing large files, use specific line-ranges, usually within 50 to 100 lines as required
2. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files.
3. Reproduction files should be small. Do not hand-write long binary/base64/escaped-byte fixtures; write short Python code in reproduce_issue.py that generates them at runtime, or reuse an existing fixture.
4. You are being told a million times, each response must include a function call. Must inlcude a function call at all costs.
The expected workflow is:
1. Use the provided localization to understand where the bug is likely located.
2. Inspect the relevant source file(s) with tools.
3. Create /testbed/reproduce_issue.py that asserts the expected fixed behavior.
4. Run /testbed/reproduce_issue.py on the current buggy repository and confirm it fails for the target issue.
5. Modify only the relevant non-test source file(s).
6. Rerun /testbed/reproduce_issue.py and confirm it passes.
7. Optionally run a small number of relevant existing tests.
