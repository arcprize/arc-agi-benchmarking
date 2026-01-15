"""Tests for the terminal viewer module."""

import json
import tempfile
from pathlib import Path

import pytest

from arc_agi_benchmarking.utils.viewer import (
    render_grid,
    render_pair,
    ARC_COLORS,
    ansi_fg,
    ansi_reset,
)


class TestRenderGrid:
    """Tests for render_grid function."""

    def test_simple_grid(self):
        """Test rendering a simple 2x2 grid."""
        grid = [[0, 1], [2, 3]]
        lines = render_grid(grid)

        # Should have border lines + 2 content lines
        assert len(lines) == 4  # top border, 2 rows, bottom border
        assert "┌" in lines[0]
        assert "└" in lines[-1]

    def test_grid_with_label(self):
        """Test rendering a grid with a label."""
        grid = [[0, 1]]
        lines = render_grid(grid, label="Test Label")

        assert len(lines) == 4  # label, top border, 1 row, bottom border
        assert "Test Label" in lines[0]

    def test_empty_grid(self):
        """Test rendering an empty grid."""
        grid = []
        lines = render_grid(grid)
        # Should still have borders
        assert len(lines) >= 2

    def test_colors_in_output(self):
        """Test that ANSI color codes are present in output."""
        grid = [[1, 2, 3]]  # Blue, Red, Green
        lines = render_grid(grid)

        # Find the content line (not borders)
        content_line = lines[1]

        # Should contain ANSI escape codes
        assert "\033[" in content_line


class TestRenderPair:
    """Tests for render_pair function."""

    def test_input_only(self):
        """Test rendering just an input grid."""
        input_grid = [[0, 1], [2, 3]]
        lines = render_pair(input_grid=input_grid)

        # Should contain "Input" label
        assert any("Input" in line for line in lines)

    def test_input_and_output(self):
        """Test rendering input and output grids."""
        input_grid = [[0, 1]]
        output_grid = [[2, 3]]
        lines = render_pair(input_grid=input_grid, output_grid=output_grid)

        # Should contain both labels
        assert any("Input" in line for line in lines)
        assert any("Expected Output" in line for line in lines)

    def test_with_prediction(self):
        """Test rendering with a predicted grid."""
        input_grid = [[0]]
        output_grid = [[1]]
        predicted_grid = [[1]]

        lines = render_pair(
            input_grid=input_grid,
            output_grid=output_grid,
            predicted_grid=predicted_grid,
            correct=True
        )

        # Should contain predicted label with checkmark
        assert any("Predicted" in line and "✓" in line for line in lines)

    def test_incorrect_prediction(self):
        """Test rendering with an incorrect prediction."""
        input_grid = [[0]]
        output_grid = [[1]]
        predicted_grid = [[2]]

        lines = render_pair(
            input_grid=input_grid,
            output_grid=output_grid,
            predicted_grid=predicted_grid,
            correct=False
        )

        # Should contain predicted label with X
        assert any("Predicted" in line and "✗" in line for line in lines)

    def test_with_pair_label(self):
        """Test rendering with a pair label."""
        input_grid = [[0]]
        lines = render_pair(input_grid=input_grid, pair_label="Test Pair 1")

        assert any("Test Pair 1" in line for line in lines)


class TestArcColors:
    """Tests for color mapping."""

    def test_all_colors_defined(self):
        """Test that all 10 colors (0-9) are defined."""
        for i in range(10):
            assert i in ARC_COLORS

    def test_color_values_valid(self):
        """Test that all color values are valid 256-color codes."""
        for value, color_code in ARC_COLORS.items():
            assert 0 <= color_code <= 255


class TestAnsiHelpers:
    """Tests for ANSI helper functions."""

    def test_ansi_fg(self):
        """Test foreground color code generation."""
        code = ansi_fg(196)
        assert code == "\033[38;5;196m"

    def test_ansi_reset(self):
        """Test reset code."""
        code = ansi_reset()
        assert code == "\033[0m"
