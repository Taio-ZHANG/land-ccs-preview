import csv
import math
import struct
import sys
import threading
import tempfile
import tkinter as tk
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({
    "axes.titlesize": 12, "axes.titleweight": "normal", "axes.titlepad": 16,
    "axes.labelsize": 10, "axes.labelcolor": "#42516A",
    "axes.edgecolor": "#D5DFEB", "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.color": "#63738B", "ytick.color": "#63738B",
    "grid.color": "#D5DFEB", "grid.linewidth": 0.7,
    "text.color": "#22324C", "legend.frameon": False,
})

ROW_DTYPE = np.dtype([
    ("time", "<u4"),
    ("voltage", "<f4"),
    ("current", "<f4"),
    ("capacity", "<f4"),
    ("energy", "<f4"),
])
RAW_DTYPE = np.dtype([("voltage", "<f4"), ("flags", "<u4")])


@dataclass
class Step:
    offset: int
    cycle: int
    start_ms: int
    wall_ms: int
    period_ms: float
    start_voltage: float
    capacity: float
    entries: list = field(default_factory=list)
    parent_rows: np.ndarray | None = None


@dataclass
class CapacityCurve:
    cycle: int
    segment: int
    direction: int
    capacity_mah: np.ndarray
    voltage_v: np.ndarray


@dataclass
class CCSData:
    path: Path
    columns: dict
    capacity_curves: list
    cycle_numbers: np.ndarray
    charge_mah: np.ndarray
    discharge_mah: np.ndarray
    coulombic_efficiency: np.ndarray
    total_cycles: int
    duration_h: float
    active_material_mass_mg: float | None = None
    mass_metadata_note: str = ""


def read_active_material_mass(raw):
    # 仅支持已用实际测试质量核对的 LAND 版本：两组参数中的质量以 g 存储。
    # 两组值含义不一致时不擅自选择，交由用户核对后手动输入。
    if (
        len(raw) < 1190 or raw[:4] != b"\x00\x00\x00\x03"
        or raw[1180:1190] != b"WuHan LAND"
        or raw[0x328:0x32C] != b"\x02\x00\x00\x00"
        or raw[0x334:0x338] != b"\x02\x00\x00\x01"
    ):
        return None, "CCS 无可确认质量 · 请手动输入"
    first_g = struct.unpack_from("<f", raw, 0x330)[0]
    second_g = struct.unpack_from("<f", raw, 0x33C)[0]
    if not all(math.isfinite(value) and value > 0 for value in (first_g, second_g)):
        return None, "CCS 质量未填写或无效 · 请手动输入"
    if not math.isclose(first_g, second_g, rel_tol=1e-6):
        return None, "CCS 两组质量不一致 · 请手动核对"
    mass_mg = first_g * 1000.0
    return mass_mg, f"来自 CCS · {mass_mg:.7g} mg"


def valid_rows(rows):
    if not len(rows):
        return rows
    values = np.column_stack([
        rows["voltage"], rows["current"], rows["capacity"], rows["energy"]
    ]).astype(float)
    mask = (
        (rows["time"] <= 864_000_000)
        & np.isfinite(values).all(axis=1)
        & (np.abs(rows["voltage"]) <= 100)
        & (np.abs(rows["current"]) <= 1000)
        & (np.abs(rows["capacity"]) <= 1e7)
        & (np.abs(rows["energy"]) <= 1e9)
    )
    nonzero = np.abs(values).sum(axis=1) > 0
    return rows[mask & nonzero]


def build_capacity_curves(steps):
    row_groups = []
    for step in sorted(steps.values(), key=lambda item: item.offset):
        for _, kind, rows in sorted(step.entries, key=lambda item: item[0]):
            if kind == "rows" and len(rows):
                row_groups.append((step.cycle + 1, rows))
    if not row_groups:
        return []

    cycles = np.concatenate([np.full(len(rows), cycle, dtype=np.int32) for cycle, rows in row_groups])
    rows = np.concatenate([rows for _, rows in row_groups])
    currents = rows["current"].astype(float)
    directions = np.sign(currents).astype(np.int8)
    current_ratios = np.ones(len(currents) - 1)
    nonzero = (np.abs(currents[1:]) > 0) & (np.abs(currents[:-1]) > 0)
    np.divide(
        np.maximum(np.abs(currents[1:]), np.abs(currents[:-1])),
        np.minimum(np.abs(currents[1:]), np.abs(currents[:-1])),
        out=current_ratios,
        where=nonzero,
    )
    boundaries = np.where(
        (cycles[1:] != cycles[:-1])
        | (directions[1:] != directions[:-1])
        | (nonzero & (current_ratios > 10))
    )[0] + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(rows)]
    curves = []
    for segment, (start, end) in enumerate(zip(starts, ends), start=1):
        direction = int(directions[start])
        if direction == 0 or end - start < 2:
            continue
        segment_rows = rows[start:end]
        increments = np.abs(segment_rows["capacity"].astype(float))
        capacity = np.r_[0.0, np.cumsum(increments) * 1000.0]
        voltage = np.r_[segment_rows["voltage"][0], segment_rows["voltage"]]
        if capacity[-1] > 0 and np.isfinite(capacity).all() and np.isfinite(voltage).all():
            curves.append(CapacityCurve(
                cycle=int(cycles[start]),
                segment=segment,
                direction=direction,
                capacity_mah=capacity,
                voltage_v=voltage,
            ))
    return curves


def primary_capacity_curves(curves, cycle):
    primary = {}
    for curve in curves:
        if curve.cycle != cycle:
            continue
        current = primary.get(curve.direction)
        if current is None or curve.capacity_mah[-1] > current.capacity_mah[-1]:
            primary[curve.direction] = curve
    return sorted(primary.values(), key=lambda curve: -curve.direction)


def write_capacity_csv(writer, curves, specific=False, mass_mg=None):
    if specific and (mass_mg is None or not math.isfinite(mass_mg) or mass_mg <= 0):
        raise ValueError("请先输入大于 0 的活性物质质量 (mg)。")
    capacity_column = "specific_capacity_mAh_per_g" if specific else "capacity_mAh"
    header = ["cycle", "segment", "direction", capacity_column, "voltage_V"]
    if specific:
        header.append("active_material_mass_mg")
    writer.writerow(header)
    scale = 1000.0 / mass_mg if specific else 1.0
    for curve in curves:
        direction = "charge" if curve.direction > 0 else "discharge"
        for capacity, voltage in zip(curve.capacity_mah, curve.voltage_v):
            row = [
                curve.cycle, curve.segment, direction,
                f"{capacity * scale:.8g}", f"{voltage:.8g}",
            ]
            if specific:
                row.append(f"{mass_mg:.8g}")
            writer.writerow(row)


def write_capacity_range_csv(writer, grouped, specific=False, mass_mg=None):
    """每个循环一对容量／电压列，充放电之间空一行，短循环末尾留空。"""
    if specific and (mass_mg is None or not math.isfinite(mass_mg) or mass_mg <= 0):
        raise ValueError("请先输入大于 0 的活性物质质量 (mg)。")
    cycles = [(cycle, primary_capacity_curves(items, cycle)) for cycle, items in sorted(grouped.items())]
    header = []
    quantity = f"specific_capacity_mAh_per_g_mass_{mass_mg:.8g}mg" if specific else "capacity_mAh"
    for cycle, _ in cycles:
        prefix = f"Cycle_{cycle}"
        header.extend([f"{prefix}_{quantity}(X)", f"{prefix}_voltage_V(Y)"])
    writer.writerow(header)
    scale = 1000.0 / mass_mg if specific else 1.0
    def cycle_points(curves):
        for index, curve in enumerate(curves):
            if index:
                yield None
            yield from zip(curve.capacity_mah, curve.voltage_v)

    points = [cycle_points(curves) for _, curves in cycles]
    for pairs in zip_longest(*points, fillvalue=None):
        row = []
        for pair in pairs:
            if pair is None:
                row.extend(["", ""])
            else:
                capacity, voltage = pair
                row.extend([f"{capacity * scale:.8g}", f"{voltage:.8g}"])
        writer.writerow(row)


def parse_ccs(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 128 or len(raw) % 128:
        raise ValueError("文件长度不符合蓝和 CCS 的 128 字节记录结构。")

    steps = {}
    for offset in range(0, len(raw) - 127, 128):
        if raw[offset] != 2 or raw[offset + 73] != 4:
            continue
        wall_ms = struct.unpack_from("<Q", raw, offset + 56)[0]
        period_ms = struct.unpack_from("<f", raw, offset + 76)[0]
        cycle = struct.unpack_from("<I", raw, offset + 12)[0]
        if not (1_400_000_000_000 <= wall_ms <= 2_100_000_000_000):
            continue
        if not (math.isfinite(period_ms) and 0 < period_ms <= 86_400_000):
            continue
        if cycle > 10_000_000:
            continue
        start_voltage = struct.unpack_from("<f", raw, offset + 52)[0]
        capacity = struct.unpack_from("<f", raw, offset + 20)[0]
        if not math.isfinite(start_voltage) or not math.isfinite(capacity):
            continue
        step = Step(
            offset=offset,
            cycle=cycle,
            start_ms=struct.unpack_from("<I", raw, offset + 24)[0],
            wall_ms=wall_ms,
            period_ms=float(period_ms),
            start_voltage=float(start_voltage),
            capacity=abs(float(capacity)),
        )
        rows = np.frombuffer(raw, dtype=ROW_DTYPE, count=2, offset=offset + 88)
        rows = valid_rows(rows)
        if len(rows):
            step.parent_rows = rows
            step.entries.append((offset, "parent_rows", rows))
        steps[offset] = step

    if not steps:
        raise ValueError("未找到可识别的蓝和测试记录，可能不是受支持的 CCS 版本。")

    duration_epoch = 0
    previous_duration = 0
    for step in sorted(steps.values(), key=lambda item: item.offset):
        duration = step.start_ms
        if previous_duration - duration > 2**31:
            duration_epoch += 2**32
        step.start_ms = duration + duration_epoch
        previous_duration = duration

    for offset in range(0, len(raw) - 127, 128):
        if raw[offset] != 3:
            continue
        parent_offset = struct.unpack_from("<I", raw, offset + 4)[0]
        step = steps.get(parent_offset)
        if step is None:
            continue
        count = raw[offset + 1]
        if not 1 <= count <= 15:
            continue
        pairs = np.frombuffer(raw, dtype=RAW_DTYPE, count=15, offset=offset + 8)
        selected = pairs[:count]
        is_raw = (
            np.all(selected["flags"] == 0)
            and np.isfinite(selected["voltage"]).all()
            and np.all(np.abs(selected["voltage"]) <= 100)
            and np.any(selected["voltage"] != 0)
        )
        if is_raw:
            voltages = selected["voltage"].copy()
            step.entries.append((offset, "raw", voltages))
            continue
        # 普通记录每块最多 6 条；count 之后的槽位可能保留其他工步的旧值，
        # 即使数值看起来合理也不能作为有效数据（或计入容量）。
        if count > 6:
            continue
        rows = np.frombuffer(raw, dtype=ROW_DTYPE, count=count, offset=offset + 8)
        rows = valid_rows(rows)
        if len(rows):
            step.entries.append((offset, "rows", rows))

    ordered_steps = sorted(steps.values(), key=lambda item: (item.start_ms, item.offset))
    arrays = {name: [] for name in [
        "cycle", "step", "test_time_s", "voltage_V", "current_A",
        "capacity_Ah", "energy_Wh"
    ]}
    step_counters = {}
    total_points = 0

    for index, step in enumerate(ordered_steps):
        step.entries.sort(key=lambda item: item[0])
        # 工步头中的电压和两条缓存记录不一定属于当前工步；
        # 有逐点子记录时只使用子记录，避免把旧缓存插在曲线开头。
        detail_entries = [entry for entry in step.entries if entry[1] in ("rows", "raw")]
        entries = detail_entries if detail_entries else step.entries
        voltage_parts = []
        current_parts = []
        capacity_parts = []
        energy_parts = []
        if not detail_entries and 0 < abs(step.start_voltage) <= 100:
            voltage_parts.append(np.array([step.start_voltage], dtype=np.float32))
            current_parts.append(np.array([np.nan], dtype=np.float32))
            capacity_parts.append(np.array([np.nan], dtype=np.float32))
            energy_parts.append(np.array([np.nan], dtype=np.float32))
        for _, kind, values in entries:
            if kind == "raw":
                size = len(values)
                voltage_parts.append(values)
                current_parts.append(np.full(size, np.nan, dtype=np.float32))
                capacity_parts.append(np.full(size, np.nan, dtype=np.float32))
                energy_parts.append(np.full(size, np.nan, dtype=np.float32))
            else:
                voltage_parts.append(values["voltage"])
                current_parts.append(values["current"])
                capacity_parts.append(values["capacity"])
                energy_parts.append(values["energy"])
        if not voltage_parts:
            continue
        voltage = np.concatenate(voltage_parts)
        size = len(voltage)
        if not size:
            continue
        next_start = ordered_steps[index + 1].start_ms if index + 1 < len(ordered_steps) else 0
        if next_start <= step.start_ms:
            next_start = step.start_ms + max(int(step.period_ms * size), 1)
        times_ms = np.linspace(step.start_ms, next_start, size, endpoint=False)
        step_counters[step.cycle] = step_counters.get(step.cycle, 0) + 1
        arrays["cycle"].append(np.full(size, step.cycle + 1, dtype=np.int32))
        arrays["step"].append(np.full(size, step_counters[step.cycle], dtype=np.int32))
        arrays["test_time_s"].append(times_ms / 1000.0)
        arrays["voltage_V"].append(voltage)
        arrays["current_A"].append(np.concatenate(current_parts))
        arrays["capacity_Ah"].append(np.concatenate(capacity_parts))
        arrays["energy_Wh"].append(np.concatenate(energy_parts))
        total_points += size

    if not total_points:
        raise ValueError("找到了文件头，但没有可显示的电压记录。")

    columns = {name: np.concatenate(parts) for name, parts in arrays.items()}
    time_order = np.argsort(columns["test_time_s"], kind="stable")
    columns = {name: values[time_order] for name, values in columns.items()}
    capacity_curves = build_capacity_curves(steps)
    cycle_capacities = {}
    for curve in capacity_curves:
        charge, discharge = cycle_capacities.get(curve.cycle, (0.0, 0.0))
        if curve.direction > 0:
            charge += curve.capacity_mah[-1]
        else:
            discharge += curve.capacity_mah[-1]
        cycle_capacities[curve.cycle] = (charge, discharge)
    cycle_numbers = sorted(cycle for cycle, (_, discharge) in cycle_capacities.items() if discharge > 0)
    charge_mah = np.asarray([cycle_capacities[cycle][0] for cycle in cycle_numbers], dtype=float)
    discharge_mah = np.asarray([cycle_capacities[cycle][1] for cycle in cycle_numbers], dtype=float)
    coulombic_efficiency = np.full(len(cycle_numbers), np.nan)
    np.divide(
        discharge_mah * 100.0, charge_mah,
        out=coulombic_efficiency,
        where=charge_mah > 0,
    )

    duration_h = float(np.max(columns["test_time_s"]) / 3600.0)
    mass_mg, mass_note = read_active_material_mass(raw)
    return CCSData(
        path=path,
        columns=columns,
        capacity_curves=capacity_curves,
        cycle_numbers=np.asarray(cycle_numbers, dtype=np.int32),
        charge_mah=charge_mah,
        discharge_mah=discharge_mah,
        coulombic_efficiency=coulombic_efficiency,
        total_cycles=len(np.unique(columns["cycle"])),
        duration_h=duration_h,
        active_material_mass_mg=mass_mg,
        mass_metadata_note=mass_note,
    )


def decimate_min_max(x, y, limit=30_000):
    if len(x) <= limit:
        return np.arange(len(x))
    bins = max(limit // 2, 1)
    edges = np.linspace(0, len(x), bins + 1, dtype=np.int64)
    indices = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = y[start:end]
        indices.extend((start + int(np.argmin(segment)), start + int(np.argmax(segment))))
    indices.append(len(x) - 1)
    return np.unique(indices)


class CCSPreview(tk.Tk):
    def __init__(self, initial_path=None):
        super().__init__()
        self.title("蓝和 CCS 快速预览")
        self.geometry("1160x800")
        self.minsize(900, 650)
        self.configure(background="#F2F5FA")
        self.data = None
        self.loading = False
        self.exporting_capacity = False
        self.configure_styles()

        toolbar = ttk.Frame(self, style="Card.TFrame", padding=(10, 4))
        toolbar.pack(fill="x", padx=8, pady=(8, 6))
        toolbar.columnconfigure(0, weight=1)
        self.file_var = tk.StringVar(value="尚未打开数据文件")
        self.file_label = ttk.Label(
            toolbar, textvariable=self.file_var, style="File.TLabel", width=1,
        )
        self.file_label.grid(row=1, column=0, sticky="ew", padx=(0, 16), pady=(5, 0))
        actions = ttk.Frame(toolbar, style="Card.TFrame")
        actions.grid(row=0, column=0, sticky="w")
        self.open_button = ttk.Button(
            actions, text="打开 CCS", style="Accent.TButton", command=self.choose_file,
        )
        self.open_button.pack(side="left", padx=(0, 8))
        self.export_button = ttk.Button(actions, text="导出当前图 CSV", command=self.export_csv)
        self.export_button.pack(side="left")
        self.export_button.state(["disabled"])

        self.summary_var = tk.StringVar(value="选择 CCS 文件，开始查看测试数据")
        ttk.Label(
            toolbar, textvariable=self.summary_var, style="CardMuted.TLabel",
        ).grid(row=1, column=1, sticky="e", pady=(5, 0))
        mass_controls = ttk.Frame(toolbar, style="Card.TFrame")
        mass_controls.grid(row=0, column=1, sticky="e")
        ttk.Label(mass_controls, text="活性物质质量", style="Card.TLabel").pack(side="left", padx=(0, 8))
        self.mass_var = tk.StringVar()
        self.mass_entry = ttk.Entry(mass_controls, textvariable=self.mass_var, width=10)
        self.mass_entry.pack(side="left")
        self.mass_entry.bind("<Return>", self.apply_mass)
        self.mass_entry.bind("<FocusOut>", self.apply_mass)
        ttk.Label(mass_controls, text="mg", style="CardMuted.TLabel").pack(side="left", padx=(6, 10))
        self.mass_button = ttk.Button(mass_controls, text="应用", width=5, command=self.apply_mass)
        self.mass_button.pack(side="left")
        # 保留状态值供加载/导出流程使用，不再用常驻说明占用绘图空间。
        self.mass_note_var = tk.StringVar(value="输入当前电池的活性物质质量")
        self.status_var = tk.StringVar(value="就绪 · 原始文件只读，导出保存为独立 CSV")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.cycle_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.specific_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.time_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.capacity_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(self.cycle_tab, text="循环 · 放电容量 / 效率")
        self.notebook.add(self.specific_tab, text="循环 · 放电比容量 / 效率")
        self.notebook.add(self.time_tab, text="时间 · 电压")
        self.notebook.add(self.capacity_tab, text="容量 · 电压")
        (
            self.cycle_start_var, self.cycle_end_var,
            self.cycle_start_box, self.cycle_end_box,
        ) = self.make_range_controls(self.cycle_tab, self.plot_cycles)
        (
            self.specific_start_var, self.specific_end_var,
            self.specific_start_box, self.specific_end_box,
        ) = self.make_range_controls(self.specific_tab, self.plot_specific_capacity)
        (
            self.time_start_var, self.time_end_var,
            self.time_start_box, self.time_end_box,
        ) = self.make_range_controls(self.time_tab, self.plot_time_voltage)
        self.cycle_figure, self.cycle_canvas = self.make_canvas(self.cycle_tab)
        self.specific_figure, self.specific_canvas = self.make_canvas(self.specific_tab)
        self.time_figure, self.time_canvas = self.make_canvas(self.time_tab)
        capacity_controls = ttk.Frame(self.capacity_tab, style="Card.TFrame", padding=(12, 4, 12, 0))
        capacity_controls.pack(fill="x")
        ttk.Label(capacity_controls, text="循环次数", style="Card.TLabel").pack(side="left")
        self.capacity_cycle_var = tk.StringVar()
        self.capacity_cycle_box = ttk.Combobox(
            capacity_controls, textvariable=self.capacity_cycle_var,
            state="readonly", width=7,
        )
        self.capacity_cycle_box.pack(side="left", padx=(8, 12))
        self.capacity_cycle_box.bind("<<ComboboxSelected>>", lambda _: self.plot_capacity())
        self.capacity_specific_var = tk.BooleanVar(value=False)
        self.capacity_specific_check = ttk.Checkbutton(
            capacity_controls, text="比容量 (mAh/g)", variable=self.capacity_specific_var,
            command=self.plot_capacity, style="Card.TCheckbutton",
        )
        self.capacity_specific_check.pack(side="left")
        ttk.Separator(capacity_controls, orient="vertical").pack(side="left", fill="y", padx=12)
        ttk.Label(capacity_controls, text="导出区间", style="Card.TLabel").pack(side="left")
        self.capacity_start_var = tk.StringVar()
        self.capacity_end_var = tk.StringVar()
        self.capacity_start_box = ttk.Spinbox(
            capacity_controls, textvariable=self.capacity_start_var, width=5,
        )
        self.capacity_start_box.pack(side="left", padx=(8, 6))
        ttk.Label(capacity_controls, text="至", style="CardMuted.TLabel").pack(side="left")
        self.capacity_end_box = ttk.Spinbox(
            capacity_controls, textvariable=self.capacity_end_var, width=5,
        )
        self.capacity_end_box.pack(side="left", padx=(6, 10))
        self.capacity_export_button = ttk.Button(
            capacity_controls, text="合并导出 CSV", command=self.export_capacity_batch,
        )
        self.capacity_export_button.pack(side="left")
        self.capacity_export_button.state(["disabled"])
        self.capacity_figure, self.capacity_canvas = self.make_canvas(self.capacity_tab)
        self.show_empty(self.cycle_figure, self.cycle_canvas, "打开 CCS 后显示每次循环的放电容量")
        self.show_empty(
            self.specific_figure, self.specific_canvas,
            "输入活性物质质量 (mg)，显示每次循环的放电比容量",
        )
        self.show_empty(self.time_figure, self.time_canvas, "打开 CCS 后显示测试时间与电压曲线")
        self.show_empty(self.capacity_figure, self.capacity_canvas, "打开 CCS 后显示容量与电压曲线")

        if initial_path:
            self.after(100, lambda: self.load_file(Path(initial_path)))

    def configure_styles(self):
        self.option_add("*TCombobox*Listbox.font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), foreground="#22324C")
        style.configure("TFrame", background="#F2F5FA")
        style.configure("TLabel", background="#F2F5FA", padding=0)
        style.configure("Muted.TLabel", foreground="#72829A", font=("Microsoft YaHei UI", 9))
        style.configure("Card.TFrame", background="white")
        style.configure("Card.TLabel", background="white")
        style.configure("CardMuted.TLabel", background="white", foreground="#72829A", font=("Microsoft YaHei UI", 9))
        style.configure("File.TLabel", background="white", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure(
            "TButton", padding=(12, 5), background="white", bordercolor="#D7E0EB",
            lightcolor="white", darkcolor="white", borderwidth=1, focusthickness=0,
        )
        style.map("TButton", background=[("active", "#E8F0FB"), ("pressed", "#DCE8FA")], foreground=[("disabled", "#A4AFC0")])
        style.configure("Accent.TButton", background="#2C6ECB", foreground="white", bordercolor="#2C6ECB")
        style.map("Accent.TButton", background=[("disabled", "#98B8E4"), ("pressed", "#1F529A"), ("active", "#245FB2")], foreground=[("disabled", "white"), ("!disabled", "white")])
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(name, padding=(7, 4), fieldbackground="white", bordercolor="#D5DFEB", lightcolor="white", darkcolor="white", arrowsize=14)
            style.map(name, bordercolor=[("focus", "#2C6ECB")], fieldbackground=[("readonly", "white")])
        style.configure("TNotebook", background="#F2F5FA", borderwidth=0, bordercolor="#E2E9F2", lightcolor="white", darkcolor="white", tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background="#E8EEF6", padding=(12, 4), borderwidth=0, bordercolor="#F2F5FA", lightcolor="#F2F5FA", darkcolor="#F2F5FA", foreground="#63738B")
        style.map("TNotebook.Tab", background=[("selected", "white"), ("active", "#E0EAF8")], foreground=[("selected", "#2264BB")], padding=[("selected", (12, 4)), ("!selected", (12, 4))], expand=[("selected", (0, 0, 0, 0))])
        style.configure("Card.TCheckbutton", background="white", padding=(4, 4), indicatorcolor="white")
        style.map("Card.TCheckbutton", background=[("active", "white")], indicatorcolor=[("selected", "#2C6ECB")])
        style.configure("TSeparator", background="#E2E9F2")

    def apply_mass(self, event=None):
        mass_mg = self.active_mass_mg()
        embedded = self.data.active_material_mass_mg if self.data is not None else None
        if mass_mg is None:
            note = self.data.mass_metadata_note if self.data is not None and not self.mass_var.get().strip() else ""
            self.mass_note_var.set(note or "请输入大于 0 的质量，单位 mg")
        elif embedded is not None and math.isclose(mass_mg, embedded, rel_tol=1e-6):
            self.mass_note_var.set(f"来自 CCS · {mass_mg:g} mg")
        else:
            self.mass_note_var.set(f"手动设置 · {mass_mg:g} mg")
        self.plot_specific_capacity()
        self.plot_capacity()

    def make_range_controls(self, parent, command):
        controls = ttk.Frame(parent, style="Card.TFrame", padding=(12, 4, 12, 0))
        controls.pack(fill="x")
        start_var = tk.StringVar()
        end_var = tk.StringVar()
        callback = lambda *_: command()
        ttk.Label(controls, text="循环区间", style="Card.TLabel").pack(side="left")
        start_box = ttk.Spinbox(controls, textvariable=start_var, width=7, command=callback)
        start_box.pack(side="left", padx=(10, 8))
        ttk.Label(controls, text="至", style="CardMuted.TLabel").pack(side="left")
        end_box = ttk.Spinbox(controls, textvariable=end_var, width=7, command=callback)
        end_box.pack(side="left", padx=(8, 4))
        for widget in (start_box, end_box):
            widget.bind("<Return>", callback)
            widget.bind("<FocusOut>", callback)
        return start_var, end_var, start_box, end_box

    def normalized_range(self, start_var, end_var, start_box, end_box):
        minimum = int(float(start_box.cget("from")))
        maximum = int(float(start_box.cget("to")))
        try:
            start = int(start_var.get())
        except ValueError:
            start = minimum
        try:
            end = int(end_var.get())
        except ValueError:
            end = maximum
        start = max(minimum, min(maximum, start))
        end = max(minimum, min(maximum, end))
        if start > end:
            start, end = end, start
        start_var.set(str(start))
        end_var.set(str(end))
        return start, end

    def cycle_range(self):
        return self.normalized_range(
            self.cycle_start_var, self.cycle_end_var,
            self.cycle_start_box, self.cycle_end_box,
        )

    def specific_cycle_range(self):
        return self.normalized_range(
            self.specific_start_var, self.specific_end_var,
            self.specific_start_box, self.specific_end_box,
        )

    def time_range(self):
        return self.normalized_range(
            self.time_start_var, self.time_end_var,
            self.time_start_box, self.time_end_box,
        )

    def active_mass_mg(self):
        text = self.mass_var.get().strip()
        if not text:
            return None
        try:
            mass = float(text)
        except ValueError:
            return None
        return mass if math.isfinite(mass) and mass > 0 else None

    def make_canvas(self, parent):
        figure = Figure(figsize=(8, 5), dpi=100, facecolor="white", layout="constrained")
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().configure(background="white", highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, canvas

    def finish_plot(self, figure, canvas):
        for index, axis in enumerate(figure.axes):
            # 图类型和循环范围已在选项卡/控制条中显示，图内不再重复大标题。
            axis.set_title("", pad=0)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(len(figure.axes) > 1 and index == 1)
            axis.tick_params(length=3, pad=6)
            if index == 0:
                axis.grid(True, alpha=0.55)
            else:
                axis.spines["left"].set_visible(False)
                axis.grid(False)
        canvas.draw_idle()

    def show_empty(self, figure, canvas, text):
        figure.clear()
        axis = figure.add_subplot(111)
        axis.text(0.5, 0.5, text, ha="center", va="center", color="#72829A", fontsize=11, linespacing=1.8)
        axis.set_axis_off()
        canvas.draw_idle()

    def choose_file(self):
        selected = filedialog.askopenfilename(
            title="选择蓝和 CCS 数据",
            filetypes=[("蓝和 CCS 数据", "*.ccs"), ("所有文件", "*.*")],
        )
        if selected:
            self.load_file(Path(selected))

    def load_file(self, path):
        if self.loading:
            return
        if not path.exists():
            messagebox.showerror("文件不存在", str(path), parent=self)
            return
        self.loading = True
        self.open_button.state(["disabled"])
        self.export_button.state(["disabled"])
        self.update_capacity_export_state()
        self.status_var.set(f"正在解析：{path.name}")

        def worker():
            try:
                result = parse_ccs(path)
            except Exception as error:
                self.after(0, lambda message=str(error): self.load_failed(path, message))
            else:
                self.after(0, lambda: self.load_succeeded(result))

        threading.Thread(target=worker, daemon=True).start()

    def load_failed(self, path, error):
        self.loading = False
        self.open_button.state(["!disabled"])
        self.export_button.state(["!disabled"] if self.data is not None else ["disabled"])
        self.update_capacity_export_state()
        self.status_var.set("解析失败 · 请检查文件格式，当前已载入数据保持不变")
        messagebox.showerror("无法解析 CCS", f"{path.name}\n\n{error}", parent=self)

    def load_succeeded(self, data):
        if data.active_material_mass_mg is not None:
            self.mass_var.set(f"{data.active_material_mass_mg:.7g}")
        elif self.data is not None and self.data.path != data.path:
            self.mass_var.set("")
        self.data = data
        self.loading = False
        self.open_button.state(["!disabled"])
        self.export_button.state(["!disabled"])
        point_count = len(data.columns["test_time_s"])
        self.title(f"蓝和 CCS 快速预览 - {data.path.name}")
        self.file_var.set(data.path.name)
        self.summary_var.set(
            f"{data.total_cycles} 次循环    /    {point_count:,} 个有效数据点    /    {data.duration_h:.2f} h"
        )
        self.status_var.set("已载入 · 仅使用有效记录，原始 CCS 文件不作修改")
        self.mass_note_var.set("请核对当前电池的质量 · 单位 mg")
        available_cycles = np.unique(data.columns["cycle"])
        first_cycle = int(available_cycles[0])
        last_cycle = int(available_cycles[-1])
        for start_box, end_box, start_var, end_var in [
            (self.cycle_start_box, self.cycle_end_box, self.cycle_start_var, self.cycle_end_var),
            (self.specific_start_box, self.specific_end_box, self.specific_start_var, self.specific_end_var),
            (self.time_start_box, self.time_end_box, self.time_start_var, self.time_end_var),
        ]:
            start_box.configure(from_=first_cycle, to=last_cycle)
            end_box.configure(from_=first_cycle, to=last_cycle)
            start_var.set(str(first_cycle))
            end_var.set(str(last_cycle))
        cycles = sorted({curve.cycle for curve in data.capacity_curves})
        self.capacity_cycle_box["values"] = [str(cycle) for cycle in cycles]
        self.capacity_cycle_var.set(str(cycles[0]) if cycles else "")
        self.capacity_start_box.configure(from_=cycles[0] if cycles else first_cycle, to=cycles[-1] if cycles else last_cycle)
        self.capacity_end_box.configure(from_=cycles[0] if cycles else first_cycle, to=cycles[-1] if cycles else last_cycle)
        self.capacity_start_var.set(str(cycles[0]) if cycles else "")
        self.capacity_end_var.set(str(cycles[-1]) if cycles else "")
        self.update_capacity_export_state()
        self.plot_cycles()
        self.plot_time_voltage()
        self.apply_mass()

    def plot_cycles(self):
        if self.data is None:
            return
        self.cycle_figure.clear()
        axis = self.cycle_figure.add_subplot(111)
        efficiency_axis = axis.twinx()
        first_cycle, last_cycle = self.cycle_range()
        mask = (
            (self.data.cycle_numbers >= first_cycle)
            & (self.data.cycle_numbers <= last_cycle)
        )
        cycles = self.data.cycle_numbers[mask]
        discharge = self.data.discharge_mah[mask]
        efficiency = self.data.coulombic_efficiency[mask]
        capacity_line = None
        efficiency_line = None
        if len(cycles):
            capacity_line, = axis.plot(
                cycles, discharge,
                color="#3278D3", marker="o", markersize=3.5,
                linewidth=1.3, label="放电容量",
            )
            valid_efficiency = np.isfinite(efficiency)
            if valid_efficiency.any():
                efficiency_line, = efficiency_axis.plot(
                    cycles[valid_efficiency], efficiency[valid_efficiency],
                    color="#159B86", marker="s", markersize=3,
                    linewidth=1.1, label="库伦效率",
                )
        else:
            axis.text(0.5, 0.5, "所选区间没有放电容量数据", ha="center", va="center", color="#777777")
        axis.set_title("次数 - 放电容量与库伦效率")
        axis.set_xlabel("循环次数")
        axis.set_ylabel("放电容量 (mAh)", color="#3278D3")
        efficiency_axis.set_ylabel("库伦效率 (%)", color="#159B86")
        axis.tick_params(axis="y", colors="#3278D3")
        efficiency_axis.tick_params(axis="y", colors="#159B86")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.03, y=0.18)
        efficiency_axis.margins(y=0.18)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        lines = [line for line in (capacity_line, efficiency_line) if line is not None]
        if lines:
            axis.legend(lines, [line.get_label() for line in lines], loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, ncol=2, fontsize=9)
            axis.set_title(axis.get_title(), pad=38)
        if len(cycles) == 1:
            cycle = int(cycles[0])
            axis.set_xlim(cycle - 0.5, cycle + 0.5)
            axis.set_xticks([cycle])
        self.finish_plot(self.cycle_figure, self.cycle_canvas)

    def plot_specific_capacity(self):
        if self.data is None:
            return
        mass_mg = self.active_mass_mg()
        if mass_mg is None:
            self.show_empty(
                self.specific_figure, self.specific_canvas,
                "请输入大于 0 的活性物质质量 (mg)\n放电比容量 = 放电容量 × 1000 / 质量",
            )
            return
        self.specific_figure.clear()
        axis = self.specific_figure.add_subplot(111)
        efficiency_axis = axis.twinx()
        first_cycle, last_cycle = self.specific_cycle_range()
        mask = (
            (self.data.cycle_numbers >= first_cycle)
            & (self.data.cycle_numbers <= last_cycle)
        )
        cycles = self.data.cycle_numbers[mask]
        specific_capacity = self.data.discharge_mah[mask] * 1000.0 / mass_mg
        efficiency = self.data.coulombic_efficiency[mask]
        capacity_line = None
        efficiency_line = None
        if len(cycles):
            capacity_line, = axis.plot(
                cycles, specific_capacity,
                color="#7865C8", marker="o", markersize=3.5,
                linewidth=1.3, label="放电比容量",
            )
            valid_efficiency = np.isfinite(efficiency)
            if valid_efficiency.any():
                efficiency_line, = efficiency_axis.plot(
                    cycles[valid_efficiency], efficiency[valid_efficiency],
                    color="#159B86", marker="s", markersize=3,
                    linewidth=1.1, label="库伦效率",
                )
        else:
            axis.text(0.5, 0.5, "所选区间没有放电容量数据", ha="center", va="center", color="#777777")
        axis.set_title(f"次数 - 放电比容量与库伦效率（活性物质 {mass_mg:g} mg）")
        axis.set_xlabel("循环次数")
        axis.set_ylabel("放电比容量 (mAh/g)", color="#7865C8")
        efficiency_axis.set_ylabel("库伦效率 (%)", color="#159B86")
        axis.tick_params(axis="y", colors="#7865C8")
        efficiency_axis.tick_params(axis="y", colors="#159B86")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.03, y=0.18)
        efficiency_axis.margins(y=0.18)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        lines = [line for line in (capacity_line, efficiency_line) if line is not None]
        if lines:
            axis.legend(lines, [line.get_label() for line in lines], loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, ncol=2, fontsize=9)
            axis.set_title(axis.get_title(), pad=38)
        if len(cycles) == 1:
            cycle = int(cycles[0])
            axis.set_xlim(cycle - 0.5, cycle + 0.5)
            axis.set_xticks([cycle])
        self.finish_plot(self.specific_figure, self.specific_canvas)

    def plot_time_voltage(self):
        if self.data is None:
            return
        self.time_figure.clear()
        axis = self.time_figure.add_subplot(111)
        first_cycle, last_cycle = self.time_range()
        mask = (
            (self.data.columns["cycle"] >= first_cycle)
            & (self.data.columns["cycle"] <= last_cycle)
        )
        time_h = self.data.columns["test_time_s"][mask] / 3600.0
        voltage = self.data.columns["voltage_V"][mask]
        cycle_values = self.data.columns["cycle"][mask]
        step_values = self.data.columns["step"][mask]
        if not len(time_h):
            axis.text(0.5, 0.5, "所选循环区间没有电压记录", ha="center", va="center", color="#777777")
        indices = decimate_min_max(time_h, voltage)
        selected_cycles = cycle_values[indices]
        selected_steps = step_values[indices]
        breaks = np.where(
            (selected_cycles[1:] != selected_cycles[:-1])
            | (selected_steps[1:] != selected_steps[:-1])
        )[0] + 1
        plot_time = np.insert(time_h[indices], breaks, np.nan)
        plot_voltage = np.insert(voltage[indices], breaks, np.nan)
        axis.plot(plot_time, plot_voltage, color="#DC626C", linewidth=0.9)
        axis.set_title(f"时间 - 电压（循环 {first_cycle}–{last_cycle}）")
        axis.set_xlabel("测试时间 (h)")
        axis.set_ylabel("电压 (V)")
        axis.grid(True, alpha=0.25)
        axis.margins(x=0.01)
        self.finish_plot(self.time_figure, self.time_canvas)

    def plot_capacity(self):
        specific = self.capacity_specific_var.get()
        self.notebook.tab(self.capacity_tab, text="比容量 · 电压" if specific else "容量 · 电压")
        if self.data is None:
            return
        if not self.capacity_cycle_var.get():
            self.show_empty(self.capacity_figure, self.capacity_canvas, "此文件没有可确认的逐点容量记录")
            return
        mass_mg = self.active_mass_mg()
        if specific and mass_mg is None:
            self.show_empty(
                self.capacity_figure, self.capacity_canvas,
                "请先输入活性物质质量 (mg) 并应用\n比容量 = 容量 × 1000 / 质量",
            )
            return
        scale = 1000.0 / mass_mg if specific else 1.0
        self.capacity_figure.clear()
        axis = self.capacity_figure.add_subplot(111)
        cycle = int(self.capacity_cycle_var.get())
        curves = primary_capacity_curves(self.data.capacity_curves, cycle)
        labelled = set()
        for curve in curves:
            direction = "充电" if curve.direction > 0 else "放电"
            color = "#DC626C" if curve.direction > 0 else "#3278D3"
            indices = decimate_min_max(curve.capacity_mah, curve.voltage_v, limit=10_000)
            axis.plot(
                curve.capacity_mah[indices] * scale, curve.voltage_v[indices],
                linewidth=1.5, color=color,
                label=direction if direction not in labelled else None,
            )
            labelled.add(direction)
        quantity = "比容量" if specific else "容量"
        mass_label = f" · {mass_mg:g} mg" if specific else ""
        axis.set_title(f"{quantity} - 电压（第 {cycle} 次循环{mass_label}）")
        axis.set_xlabel("比容量 (mAh/g)" if specific else "容量 (mAh)")
        axis.set_ylabel("电压 (V)")
        if labelled:
            axis.legend(loc="best", fontsize=9)
        self.finish_plot(self.capacity_figure, self.capacity_canvas)

    def update_capacity_export_state(self):
        enabled = (
            self.data is not None and bool(self.data.capacity_curves)
            and not self.loading and not self.exporting_capacity
        )
        self.capacity_export_button.state(["!disabled"] if enabled else ["disabled"])

    def export_capacity_batch(self):
        if self.data is None or self.loading or self.exporting_capacity:
            return
        specific = self.capacity_specific_var.get()
        mass_mg = self.active_mass_mg()
        if specific and mass_mg is None:
            messagebox.showinfo("需要活性物质质量", "请先输入大于 0 的活性物质质量 (mg)。", parent=self)
            return
        if not self.data.capacity_curves:
            messagebox.showinfo("没有容量数据", "此文件没有可导出的逐点容量记录。", parent=self)
            return
        first_cycle, last_cycle = self.normalized_range(
            self.capacity_start_var, self.capacity_end_var,
            self.capacity_start_box, self.capacity_end_box,
        )
        grouped = {}
        for curve in self.data.capacity_curves:
            if first_cycle <= curve.cycle <= last_cycle:
                grouped.setdefault(curve.cycle, []).append(curve)
        if not grouped:
            messagebox.showinfo("没有容量数据", "所选循环区间没有可导出的逐点容量记录。", parent=self)
            return
        quantity = "specific_capacity_voltage" if specific else "capacity_voltage"
        stem = self.data.path.stem
        selected = filedialog.asksaveasfilename(
            title="导出多循环容量－电压曲线到一个 CSV", parent=self,
            defaultextension=".csv",
            initialfile=f"{stem}_{quantity}_cycles_{first_cycle}-{last_cycle}.csv",
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not selected:
            return
        output_path = Path(selected)
        self.exporting_capacity = True
        self.capacity_export_button.configure(text="正在导出…")
        self.update_capacity_export_state()

        def worker():
            error_message = None
            temporary_path = None
            try:
                # 完整写入后才替换目标；写入失败时保留用户已有 CSV。
                with tempfile.NamedTemporaryFile(
                    mode="w", newline="", encoding="utf-8-sig", delete=False,
                    dir=output_path.parent, prefix=".ccs_export_", suffix=".tmp",
                ) as handle:
                    temporary_path = Path(handle.name)
                    write_capacity_range_csv(csv.writer(handle), grouped, specific, mass_mg)
                temporary_path.replace(output_path)
            except Exception as error:
                error_message = str(error)
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            self.after(0, lambda: self.capacity_batch_finished(
                output_path, len(grouped), last_cycle - first_cycle + 1 - len(grouped), error_message,
            ))

        threading.Thread(target=worker, daemon=True).start()

    def capacity_batch_finished(self, output_path, completed, skipped, error_message):
        self.exporting_capacity = False
        self.capacity_export_button.configure(text="合并导出 CSV")
        self.update_capacity_export_state()
        if error_message is not None:
            self.status_var.set(f"合并导出失败：{output_path}")
            messagebox.showerror(
                "合并导出失败", f"未保存合并结果：\n{output_path}\n\n{error_message}", parent=self,
            )
            return
        self.status_var.set(f"已合并导出 {completed} 个循环：{output_path}")
        note = f"\n跳过 {skipped} 个没有容量记录的循环。" if skipped else ""
        messagebox.showinfo(
            "合并导出完成",
            f"已将 {completed} 个循环合并到一个 CSV。{note}\n\n{output_path}\n\n"
            "Origin：导入后全选数据列 → 设置为 XY → 点击折线图。\n"
            "每个循环一组容量(X)、电压(Y)列，充放电之间以空行分隔。",
            parent=self,
        )

    def export_csv(self):
        if self.data is None:
            return
        tab_index = self.notebook.index(self.notebook.select())
        mass_mg = self.active_mass_mg()
        specific_curve = tab_index == 3 and self.capacity_specific_var.get()
        if (tab_index == 1 or specific_curve) and mass_mg is None:
            messagebox.showinfo(
                "需要活性物质质量", "请先输入大于 0 的活性物质质量 (mg)。", parent=self,
            )
            return
        if tab_index == 3 and not self.capacity_cycle_var.get():
            messagebox.showinfo("没有容量数据", "此文件没有可导出的逐点容量记录。", parent=self)
            return
        suffixes = [
            "_cycle_discharge.csv", "_cycle_specific_discharge.csv",
            "_time_voltage.csv", "_specific_capacity_voltage.csv" if specific_curve else "_capacity_voltage.csv",
        ]
        default_name = self.data.path.stem + suffixes[tab_index]
        selected = filedialog.asksaveasfilename(
            title="导出 CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV 文件", "*.csv")],
        )
        if not selected:
            return
        try:
            with open(selected, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                if tab_index == 0:
                    first_cycle, last_cycle = self.cycle_range()
                    mask = (
                        (self.data.cycle_numbers >= first_cycle)
                        & (self.data.cycle_numbers <= last_cycle)
                    )
                    writer.writerow([
                        "cycle", "charge_capacity_mAh", "discharge_capacity_mAh",
                        "coulombic_efficiency_percent",
                    ])
                    for index in np.flatnonzero(mask):
                        efficiency = self.data.coulombic_efficiency[index]
                        writer.writerow([
                            int(self.data.cycle_numbers[index]),
                            f"{self.data.charge_mah[index]:.8g}",
                            f"{self.data.discharge_mah[index]:.8g}",
                            "" if np.isnan(efficiency) else f"{efficiency:.8g}",
                        ])
                elif tab_index == 1:
                    first_cycle, last_cycle = self.specific_cycle_range()
                    mask = (
                        (self.data.cycle_numbers >= first_cycle)
                        & (self.data.cycle_numbers <= last_cycle)
                    )
                    writer.writerow([
                        "cycle", "active_material_mass_mg",
                        "discharge_specific_capacity_mAh_per_g",
                        "coulombic_efficiency_percent",
                    ])
                    for index in np.flatnonzero(mask):
                        efficiency = self.data.coulombic_efficiency[index]
                        specific_capacity = self.data.discharge_mah[index] * 1000.0 / mass_mg
                        writer.writerow([
                            int(self.data.cycle_numbers[index]),
                            f"{mass_mg:.8g}",
                            f"{specific_capacity:.8g}",
                            "" if np.isnan(efficiency) else f"{efficiency:.8g}",
                        ])
                elif tab_index == 2:
                    columns = self.data.columns
                    writer.writerow([
                        "cycle", "step", "test_time_s", "test_time_h", "voltage_V",
                        "current_A", "capacity_Ah", "energy_Wh"
                    ])
                    first_cycle, last_cycle = self.time_range()
                    indices = np.flatnonzero(
                        (columns["cycle"] >= first_cycle)
                        & (columns["cycle"] <= last_cycle)
                    )
                    for start in range(0, len(indices), 100_000):
                        end = min(start + 100_000, len(indices))
                        for index in indices[start:end]:
                            writer.writerow([
                                int(columns["cycle"][index]),
                                int(columns["step"][index]),
                                f"{columns['test_time_s'][index]:.6f}",
                                f"{columns['test_time_s'][index] / 3600.0:.9f}",
                                f"{columns['voltage_V'][index]:.8g}",
                                "" if np.isnan(columns["current_A"][index]) else f"{columns['current_A'][index]:.8g}",
                                "" if np.isnan(columns["capacity_Ah"][index]) else f"{columns['capacity_Ah'][index]:.8g}",
                                "" if np.isnan(columns["energy_Wh"][index]) else f"{columns['energy_Wh'][index]:.8g}",
                            ])
                else:
                    cycle = int(self.capacity_cycle_var.get())
                    curves = primary_capacity_curves(self.data.capacity_curves, cycle)
                    write_capacity_csv(writer, curves, specific_curve, mass_mg)
        except OSError as error:
            messagebox.showerror("导出失败", str(error), parent=self)
            return
        self.status_var.set(f"已导出：{selected}")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    app = CCSPreview(sys.argv[1] if len(sys.argv) > 1 else None)
    app.mainloop()
