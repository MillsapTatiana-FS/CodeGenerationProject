import traceback

def safe_exec(code: str, global_env: dict):
    """
    Safely execute Python code inside a controlled environment.
    Returns True if execution succeeds, False otherwise.
    """
    try:
        exec(code, global_env)
        return True
    except Exception:
        return False


def run_unit_tests(func_name: str, global_env: dict):
    """
    Run simple unit tests for a given function name.
    Each test returns True/False.
    """
    tests = []

    try:
        func = global_env[func_name]
    except KeyError:
        return 0, ["Function not found in generated code."]

    errors = []

    # Define basic tests based on function name
    try:
        if func_name == "factorial":
            tests.append(func(5) == 120)
            tests.append(func(0) == 1)

        elif func_name == "is_prime":
            tests.append(func(7) is True)
            tests.append(func(8) is False)

        elif func_name == "fibonacci":
            tests.append(func(5) == [0, 1, 1, 2, 3])

        elif func_name == "sum_list":
            tests.append(func([1, 2, 3]) == 6)

        elif func_name == "sort_list":
            tests.append(func([3, 1, 2]) == [1, 2, 3])

        elif func_name == "remove_duplicates":
            tests.append(sorted(func([1, 2, 2, 3])) == [1, 2, 3])

        elif func_name == "count_occurrences":
            tests.append(func([1, 2, 2, 3], 2) == 2)

        elif func_name == "merge_lists":
            tests.append(func([1, 2], [3, 4]) == [1, 2, 3, 4])

        elif func_name == "multiply_elements":
            tests.append(func([1, 2, 3], 2) == [2, 4, 6])

        elif func_name == "absolute_value":
            tests.append(func(-5) == 5)
            tests.append(func(3) == 3)

        elif func_name == "count_evens":
            tests.append(func([1, 2, 3, 4]) == 2)

        elif func_name == "square_map":
            tests.append(func([1, 2, 3]) == {1: 1, 2: 4, 3: 9})

        else:
            errors.append(f"No tests defined for function: {func_name}")

    except Exception as e:
        errors.append(str(e))

    passed = sum(tests)
    return passed, errors


def evaluate(generated_code: str, expected_code: str):
    """
    Evaluate generated code against expected code using:
    - safe execution
    - unit tests
    - pass/fail scoring
    """
    gen_env = {}
    exp_env = {}

    # Extract function name from expected code
    first_line = expected_code.split("\n")[0]
    func_name = first_line.replace("def ", "").split("(")[0]

    # Execute generated code
    gen_ok = safe_exec(generated_code, gen_env)

    if not gen_ok:
        return {
            "function": func_name,
            "passed": 0,
            "total_tests": 0,
            "errors": ["Generated code failed to execute."]
        }

    # Execute expected code (for reference)
    safe_exec(expected_code, exp_env)

    # Run tests
    passed, errors = run_unit_tests(func_name, gen_env)

    return {
        "function": func_name,
        "passed": passed,
        "total_tests": len(errors) + passed if errors else passed,
        "errors": errors
    }


if __name__ == "__main__":
    # Quick manual test
    sample_generated = "def reverse_string(s):\n    return s[::-1]"
    sample_expected = "def reverse_string(s):\n    return s[::-1]"

    result = evaluate(sample_generated, sample_expected)
    print(result)
