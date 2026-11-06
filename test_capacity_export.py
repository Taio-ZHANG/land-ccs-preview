import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import land_ccs_preview as preview


class InlineThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class CapacityExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = preview.CCSPreview()
        cls.app.withdraw()
        cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.output = self.folder / "merged.csv"
        self.curves = [
            preview.CapacityCurve(2, 1, 1, np.array([0., 1., 2.]), np.array([3., 3.5, 4.])),
            preview.CapacityCurve(2, 2, 1, np.array([0., .1]), np.array([3., 3.1])),
            preview.CapacityCurve(2, 3, -1, np.array([0., 1.5]), np.array([4., 3.])),
            preview.CapacityCurve(4, 4, -1, np.array([0., 3.]), np.array([4., 2.5])),
        ]
        app = self.app
        app.data = SimpleNamespace(path=Path("样例电池.ccs"), capacity_curves=self.curves)
        app.loading = app.exporting_capacity = False
        app.capacity_cycle_var.set("2")
        app.capacity_specific_var.set(False)
        app.mass_var.set("2")
        app.capacity_start_box.configure(from_=2, to=4)
        app.capacity_end_box.configure(from_=2, to=4)
        app.capacity_start_var.set("2")
        app.capacity_end_var.set("4")
        app.update_capacity_export_state()
        self.ask = self.enter_patch(patch.object(preview.filedialog, "asksaveasfilename", return_value=str(self.output)))
        self.info = self.enter_patch(patch.object(preview.messagebox, "showinfo"))
        self.error = self.enter_patch(patch.object(preview.messagebox, "showerror"))
        self.enter_patch(patch.object(preview.threading, "Thread", InlineThread))
        self.enter_patch(patch.object(app, "after", side_effect=lambda delay, callback: callback()))

    def enter_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    @staticmethod
    def read_rows(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_inclusive_range_one_file_xy_pairs_primary_segments_and_padding(self):
        self.app.export_capacity_batch()
        files = sorted(self.folder.rglob("*.csv"))
        self.assertEqual(files, [self.output])
        with self.output.open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.reader(handle))
        self.assertEqual(table[0], [
            "Cycle_2_capacity_mAh(X)", "Cycle_2_voltage_V(Y)",
            "Cycle_4_capacity_mAh(X)", "Cycle_4_voltage_V(Y)",
        ])
        self.assertEqual(table[1:], [
            ["0", "3", "0", "4"],
            ["1", "3.5", "3", "2.5"],
            ["2", "4", "", ""],
            ["", "", "", ""],
            ["0", "4", "", ""],
            ["1.5", "3", "", ""],
        ])
        self.assertTrue(self.output.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertIn("跳过 1", self.info.call_args.args[1])
        self.assertFalse(self.app.exporting_capacity)
        self.assertFalse(self.app.capacity_export_button.instate(["disabled"]))

    def test_specific_capacity_and_single_cycle_match_existing_export(self):
        app = self.app
        app.capacity_specific_var.set(True)
        app.capacity_start_var.set("2")
        app.capacity_end_var.set("2")
        app.export_capacity_batch()
        batch_path = next(self.folder.rglob("*.csv"))
        rows = self.read_rows(batch_path)
        capacity_column = "Cycle_2_specific_capacity_mAh_per_g_mass_2mg(X)"
        self.assertEqual([row[capacity_column] for row in rows], ["0", "500", "1000", "", "0", "750"])
        self.assertEqual([row["Cycle_2_voltage_V(Y)"] for row in rows], ["3", "3.5", "4", "", "4", "3"])
        app.notebook.select(app.capacity_tab)
        single_path = self.folder / "single.csv"
        with patch.object(preview.filedialog, "asksaveasfilename", return_value=str(single_path)):
            app.export_csv()
        single_rows = self.read_rows(single_path)
        self.assertEqual([row["specific_capacity_mAh_per_g"] for row in single_rows], ["0", "500", "1000", "0", "750"])
        self.assertEqual([row[capacity_column] for row in rows if row[capacity_column]], [row["specific_capacity_mAh_per_g"] for row in single_rows])
        self.assertTrue(all(row["active_material_mass_mg"] == "2" for row in single_rows))

    def test_range_reversal_clamping_and_single_cycle(self):
        self.app.capacity_start_var.set("99")
        self.app.capacity_end_var.set("4")
        self.app.export_capacity_batch()
        rows = self.read_rows(next(self.folder.rglob("*.csv")))
        self.assertEqual(list(rows[0]), ["Cycle_4_capacity_mAh(X)", "Cycle_4_voltage_V(Y)"])
        self.app.capacity_start_var.set("4")
        self.app.capacity_end_var.set("2")
        self.app.export_capacity_batch()
        self.assertEqual((self.app.capacity_start_var.get(), self.app.capacity_end_var.get()), ("2", "4"))

    def test_invalid_mass_does_not_create_files(self):
        self.app.capacity_specific_var.set(True)
        for mass in ("", "0", "-1", "nan", "inf", "错误"):
            with self.subTest(mass=mass):
                self.app.mass_var.set(mass)
                self.app.export_capacity_batch()
        self.ask.assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_missing_cycles_and_cancel_do_not_create_folders(self):
        self.app.capacity_start_var.set("3")
        self.app.capacity_end_var.set("3")
        self.app.export_capacity_batch()
        self.ask.assert_not_called()
        self.app.capacity_start_var.set("2")
        self.ask.return_value = ""
        self.app.export_capacity_batch()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_loading_and_missing_data_disable_batch(self):
        self.app.loading = True
        self.app.update_capacity_export_state()
        self.assertTrue(self.app.capacity_export_button.instate(["disabled"]))
        self.app.export_capacity_batch()
        self.ask.assert_not_called()
        self.app.loading = False
        self.app.data.capacity_curves = []
        self.app.update_capacity_export_state()
        self.assertTrue(self.app.capacity_export_button.instate(["disabled"]))

    def test_write_failure_preserves_existing_file_and_restores_button(self):
        self.output.write_bytes(b"existing result")
        with patch.object(preview, "write_capacity_range_csv", side_effect=OSError("磁盘写入失败")):
            self.app.export_capacity_batch()
        self.assertFalse(self.app.exporting_capacity)
        self.assertFalse(self.app.capacity_export_button.instate(["disabled"]))
        self.error.assert_called_once()
        self.assertIn("磁盘写入失败", self.error.call_args.args[1])
        self.assertEqual(self.output.read_bytes(), b"existing result")
        self.assertEqual(list(self.folder.iterdir()), [self.output])

    def test_full_resolution_export(self):
        count = 12001
        curve = preview.CapacityCurve(2, 1, 1, np.linspace(0, 10, count), np.linspace(3, 4, count))
        output = io.StringIO(newline="")
        preview.write_capacity_range_csv(csv.writer(output), {2: [curve]})
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), count)
        self.assertEqual(float(rows[-1]["Cycle_2_capacity_mAh(X)"]), 10.)


if __name__ == "__main__":
    unittest.main()
