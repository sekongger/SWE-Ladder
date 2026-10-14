## Stage
{stage}

## Issue statement
{issue_statement}

## Required artifact
Identify the causal bug location. Return suspected non-test files, hunk or line ranges, evidence from code inspection, and uncertainty notes. Do not include a final patch.

Before calling finish, your final artifact must include at least one concrete non-test source file path, a line or hunk range, and code evidence from a viewed file. Verify that each cited line range is inside the relevant class or function body rather than a nearby unrelated definition. If evidence is missing, keep using the tools instead of finalizing.

Your first action must be repository inspection, not finish. Before finalizing, first make at least one successful non-finish repository inspection call such as search, file_editor view, or execute_bash. To finalize, call the finish function with command=submit and put the artifact in result: <function=finish><parameter=command>submit</parameter><parameter=result>...</parameter></function>. Never use <function=submit>.
