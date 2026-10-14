import argparse
import importlib


def parse_command_line_arguments() -> tuple[int, int]:

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--week", type=int, default=1)
    parser.add_argument("-t", "--task", type=int, default=1)
    args, _ = parser.parse_known_args()
    return (args.week, args.task)


def main():
    (week, task) = parse_command_line_arguments()
    module_name = f"weeks.week{week:02d}.task{task:02d}"
    try:
        task_module = importlib.import_module(module_name)
        task_module.main()
    except ModuleNotFoundError as error:
        if error.name == module_name or module_name.startswith(f"{error.name}."):
            raise ValueError(
                f"Task {task:02d} does not exist in week {week:02d}"
            ) from error
    return 0
