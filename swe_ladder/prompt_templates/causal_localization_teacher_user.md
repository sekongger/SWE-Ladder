## Stage
{stage}

## Issue statement
{issue_statement}

## How to use the code context
The JSON below may contain possible relevant non-test files, hunk ranges, and old source snippets. Treat these as code context that can help you understand the problem, not the answer. Use them to evaluate which locations are plausible, but do not simply repeat a file path or line range unless it is consistent with the issue statement and code evidence.

## {teacher_evidence_heading}
{teacher_evidence_json}

## Required artifact
Identify the causal bug location. Return suspected non-test files, hunk or line ranges, evidence from code inspection, and uncertainty notes. Do not include a final patch.

Before calling finish, your final artifact must include at least one concrete non-test source file path, a line or hunk range, and code evidence from a viewed file. Verify that each cited line range is inside the relevant class or function body rather than a nearby unrelated definition. If evidence is missing, keep using the tools instead of submitting.

Your first action must be repository inspection, not finish. Before submitting, first make at least one successful non-finish repository inspection call such as search, file_editor view, or execute_bash. Submit only with <function=finish><parameter=command>submit</parameter><parameter=result>...</parameter></function>.

