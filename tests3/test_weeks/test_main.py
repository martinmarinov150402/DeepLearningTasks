import sys
import unittest

import pytest

from weeks import main


class TestParseCommandLineArguments(unittest.TestCase):
    def setUp(self) -> None:
        self.original_argv = sys.argv

    def tearDown(self) -> None:
        sys.argv = self.original_argv

    def test_when_arguments_are_not_passed(self):
        # Arrange
        expected = (1, 1)

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_weeks_are_passed_then_returns_weeks_one(self):
        # Arrange
        expected_week = 2
        expected = (expected_week, 1)
        sys.argv = [self.original_argv[0], "-w", str(expected_week)]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_tasks_are_passed_then_returns_tasks_one(self):
        # Arrange
        expected_task = 2
        expected = (1, expected_task)
        sys.argv = [self.original_argv[0], "-t", str(expected_task)]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    def test_when_weeks_and_tasks_are_passed_then_returns_weeks_and_tasks(self):
        # Arrange
        expected_week = 2
        expected_task = 3
        expected = (expected_week, expected_task)
        sys.argv = [
            self.original_argv[0],
            "-w",
            str(expected_week),
            "-t",
            str(expected_task),
        ]

        # Act
        actual = main.parse_command_line_arguments()

        # Assert
        self.assertEqual(actual, expected)

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_correct_arguments_are_passed_then_executes_the_correct_task(self):
        executed_week = 5
        executed_task = 2
        sys.argv = [
            self.original_argv[0],
            "-w",
            str(executed_week),
            "-t",
            str(executed_task),
        ]

        main.main()
        output = self.capsys.readouterr()

        self.assertIn(f"Running week {executed_week}, task {executed_task}", output.out)

    def test_when_unexisting_task_is_passed_then_raises_task_not_found_error(self):
        executed_week = 5
        executed_task = 99
        sys.argv = [
            self.original_argv[0],
            "-w",
            str(executed_week),
            "-t",
            str(executed_task),
        ]

        with self.assertRaisesRegex(
            ValueError,
            f"Task {executed_task:02d} does not exist in week {executed_week:02d}",
        ):
            main.main()


class TestMain(unittest.TestCase):
    def test_when_called_then_returns_zero(self):
        # Arrange
        expected = 0

        # Act
        actual = main.main()

        # Assert
        self.assertEqual(actual, expected)
