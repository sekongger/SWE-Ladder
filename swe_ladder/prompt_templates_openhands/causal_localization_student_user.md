## Stage
{stage}

## Issue statement
{issue_statement}

## Required artifact
Identify the causal bug location. Return suspected non-test files, hunk or line ranges, evidence from code inspection, and uncertainty notes. Do not include a final patch.

Before calling submit, your final artifact must include at least one concrete non-test source file path, a line or hunk range, and code evidence from a viewed file. Verify that each cited line range is inside the relevant class or function body rather than a nearby unrelated definition. If evidence is missing, keep using the tools instead of finalizing.

Your first action must be repository inspection, not submit. Before finalizing, first make at least one successful non-submit repository inspection call such as `grep -rn` through execute_bash or str_replace_editor view. To finalize, write the artifact in your final message text and then call <function=submit>
</function>; the submit function takes no parameters.
