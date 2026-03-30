import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter
from qtpy import QtCore, QtGui, QtWidgets
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt

APP_NAME = "MyScope"
APP_VERSION = "0.3.8"

MAX_PLOT_POINTS_MAIN = 20000
MAX_PLOT_POINTS_BAND = 8000


class SignalDataset:
    def __init__(self):
        self.root_props = {}
        self.groups = {}

    def clear(self):
        self.root_props = {}
        self.groups = {}

    def has_groups(self):
        return bool(self.groups)

    def ensure_unique_group_name(self, base_name):
        name = str(base_name).strip() or "Group"
        if name not in self.groups:
            return name
        idx = 2
        while f"{name} ({idx})" in self.groups:
            idx += 1
        return f"{name} ({idx})"

    def add_group(self, group_name, group_props=None, channels=None):
        unique_name = self.ensure_unique_group_name(group_name)
        self.groups[unique_name] = {
            "props": dict(group_props or {}),
            "channels": dict(channels or {}),
        }
        return unique_name

    def get_group_names(self):
        return list(self.groups.keys())

    def get_group(self, group_name):
        return self.groups.get(group_name, {"props": {}, "channels": {}})

    def total_channel_count(self):
        return sum(len(g["channels"]) for g in self.groups.values())


class CustomViewBox(pg.ViewBox):
    MODE_RECT = "rect"
    MODE_PAN = "pan"
    MODE_X = "x"
    MODE_Y = "y"

    def __init__(self):
        super().__init__()
        self.mode = self.MODE_RECT
        self.setMouseMode(pg.ViewBox.RectMode)

    def set_mode(self, mode):
        self.mode = mode
        if mode in (self.MODE_RECT, self.MODE_X, self.MODE_Y):
            self.setMouseMode(pg.ViewBox.RectMode)
        elif mode == self.MODE_PAN:
            self.setMouseMode(pg.ViewBox.PanMode)

    def mouseDragEvent(self, ev, axis=None):
        if self.mode == self.MODE_X:
            super().mouseDragEvent(ev, axis=0)
            return
        if self.mode == self.MODE_Y:
            super().mouseDragEvent(ev, axis=1)
            return
        super().mouseDragEvent(ev, axis=axis)

    def wheelEvent(self, ev, axis=None):
        if self.mode == self.MODE_X:
            super().wheelEvent(ev, axis=0)
            return
        if self.mode == self.MODE_Y:
            super().wheelEvent(ev, axis=1)
            return
        super().wheelEvent(ev, axis=axis)


def _new_single_group_dataset(group_name, group_props=None, channels=None, root_props=None):
    dataset = SignalDataset()
    dataset.root_props = dict(root_props or {})
    dataset.add_group(group_name, group_props=group_props, channels=channels)
    return dataset


def dataset_from_tdms(tdms_file):
    dataset = SignalDataset()
    dataset.root_props = dict(tdms_file.properties)

    for group in tdms_file.groups():
        channels = {}
        for ch in group.channels():
            y = np.asarray(ch[:], dtype=float)
            if "wf_increment" in ch.properties:
                dt = float(ch.properties["wf_increment"])
                t0 = float(ch.properties.get("wf_start_offset", 0.0))
                x = t0 + np.arange(len(y), dtype=float) * dt
            else:
                x = np.arange(len(y), dtype=float)

            channels[ch.name] = {
                "x": x,
                "y": y,
                "unit": str(ch.properties.get("unit_string", "")),
            }

        dataset.add_group(group.name, group_props=dict(group.properties), channels=channels)

    return dataset


def dataset_from_slab(file_path):
    with open(file_path, "r", encoding="cp1250") as f:
        lines = f.readlines()

    header_line_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("t") and "ch" in s.lower():
            header_line_idx = i
            break

    if header_line_idx is None:
        raise RuntimeError("Measurement header line not found (line starting with 't' and containing 'ch').")

    header_lines = lines[:header_line_idx]

    def find_value(label):
        pat = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$")
        for ln in header_lines:
            m = pat.match(ln)
            if m:
                val = m.group(1).strip()
                return val if val else None
        return None

    object_name = find_value("Object")
    measurement = find_value("Measurement")
    measured_by = find_value("Measured by")
    time_str = find_value("Time")
    date_str = find_value("Date")
    file_name_str = find_value("File Name")

    group_name = (measurement or "Measured Data").strip()
    group_props = {
        "Author": measured_by,
        "Project": object_name,
        "Date": date_str,
        "Time": time_str,
        "SourceFormat": "sLAB",
    }
    group_props = {k: v for k, v in group_props.items() if v is not None}
    root_props = {"SourceFile": file_name_str or file_path}

    desc_re = re.compile(r"^\s*Channel\s+(\d+)\s*-\s*(.*?)\s*\[(.*?)\]\s*$")
    channel_num_to_name = {}
    channel_name_to_unit = {}
    for ln in header_lines:
        m = desc_re.match(ln.strip())
        if m:
            ch_num = int(m.group(1))
            name = m.group(2).strip()
            unit = m.group(3).strip()
            channel_num_to_name[ch_num] = name
            channel_name_to_unit[name] = unit

    header_cols = lines[header_line_idx].rstrip("\r\n").split("\t")
    while header_cols and header_cols[-1] == "":
        header_cols.pop()

    col_names = []
    time_col_index = None
    for idx, col in enumerate(header_cols):
        c = col.strip()
        if c.lower().startswith("t"):
            col_names.append("Time")
            time_col_index = idx
            continue

        m = re.match(r"^ch\s*(\d+)$", c, flags=re.IGNORECASE)
        if m:
            ch_num = int(m.group(1))
            name = channel_num_to_name.get(ch_num, f"ch{ch_num}")
            col_names.append(name)
        else:
            col_names.append(c if c else f"col_{idx}")

    if time_col_index is None:
        raise RuntimeError("No Time column found")

    data_cols = [[] for _ in range(len(col_names))]
    for line in lines[header_line_idx + 1:]:
        raw = line.rstrip("\r\n")
        if not raw.strip():
            continue

        parts = raw.split("\t")
        while parts and parts[-1] == "":
            parts.pop()

        if len(parts) <= time_col_index:
            continue

        if len(parts) < len(col_names):
            parts = parts + [""] * (len(col_names) - len(parts))
        elif len(parts) > len(col_names):
            parts = parts[:len(col_names)]

        parsed = []
        row_ok = True
        for val in parts:
            val = val.strip()
            if val == "":
                parsed.append(np.nan)
                continue
            try:
                parsed.append(float(val.replace(",", ".")))
            except ValueError:
                row_ok = False
                break

        if not row_ok:
            continue

        for i, val in enumerate(parsed):
            data_cols[i].append(val)

    data_cols = [np.asarray(col, dtype=float) for col in data_cols]
    t_all = data_cols[time_col_index]
    valid_t = ~np.isnan(t_all)
    t = t_all[valid_t]

    if len(t) < 2:
        raise RuntimeError("Not enough samples")

    dt = float(np.median(np.diff(t)))
    diffs = np.diff(t)
    max_dev = float(np.max(np.abs(diffs - dt)))
    tol = max(1e-9, abs(dt) * 1e-6)
    if max_dev > tol:
        raise RuntimeError("Time is not uniformly sampled; cannot prepare waveform export safely.")

    channels = {}
    for idx, name in enumerate(col_names):
        if idx == time_col_index:
            continue
        y = data_cols[idx][valid_t]
        if len(y) == 0 or np.all(np.isnan(y)):
            continue

        channels[name] = {
            "x": t.copy(),
            "y": y,
            "unit": channel_name_to_unit.get(name, ""),
        }

    return _new_single_group_dataset(group_name, group_props=group_props, channels=channels, root_props=root_props)


def dataset_from_srm(file_path):
    with open(file_path, "r", encoding="cp1250") as f:
        lines = f.readlines()

    def find_value(label):
        pat = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$")
        for ln in lines:
            m = pat.match(ln.strip())
            if m:
                val = m.group(1).strip()
                return val if val else None
        return None

    object_name = find_value("Object")
    measurement = find_value("Measurement")
    measured_by = find_value("Measured by")
    time_str = find_value("Time")
    date_str = find_value("Date")
    scan_per_s = find_value("scan/s")

    group_name = (measurement or "SRM Data").strip()
    group_props = {
        "Author": measured_by,
        "Project": object_name,
        "Date": date_str,
        "Time": time_str,
        "SourceFormat": "SRM",
    }
    group_props = {k: v for k, v in group_props.items() if v is not None}
    root_props = {"SourceFile": file_path}

    data_start_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        parts = s.split("\t")
        ok = True
        for p in parts:
            try:
                float(p.replace(",", "."))
            except ValueError:
                ok = False
                break
        if ok and len(parts) >= 2:
            data_start_idx = i
            break

    if data_start_idx is None:
        raise RuntimeError("No numeric SRM data rows found.")

    rows = []
    max_cols = 0
    for line in lines[data_start_idx:]:
        s = line.strip()
        if not s:
            continue
        parts = s.split("\t")
        vals = []
        row_ok = True
        for p in parts:
            p = p.strip()
            if p == "":
                vals.append(np.nan)
                continue
            try:
                vals.append(float(p.replace(",", ".")))
            except ValueError:
                row_ok = False
                break
        if not row_ok:
            continue

        rows.append(vals)
        max_cols = max(max_cols, len(vals))

    if not rows:
        raise RuntimeError("No valid SRM samples found.")

    padded = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [np.nan] * (max_cols - len(r))
        padded.append(r)
    data = np.asarray(padded, dtype=float)

    if data.shape[1] < 2:
        raise RuntimeError("SRM file must contain time column and at least one signal column.")

    t = data[:, 0]
    valid_t = ~np.isnan(t)
    t = t[valid_t]

    if len(t) < 2:
        raise RuntimeError("Not enough samples")

    dt = float(np.median(np.diff(t)))
    diffs = np.diff(t)
    max_dev = float(np.max(np.abs(diffs - dt)))
    tol = max(1e-9, abs(dt) * 1e-6)
    if max_dev > tol:
        raise RuntimeError("Time is not uniformly sampled; cannot prepare waveform export safely.")

    if scan_per_s is not None:
        try:
            group_props["SamplingRate"] = float(scan_per_s.replace(",", "."))
        except ValueError:
            pass

    channel_names = ["X1", "Y1", "X2", "Y2", "X3", "Y3", "Pgen", "KP"]
    channels = {}
    for col_idx in range(1, min(data.shape[1], len(channel_names) + 1)):
        y = data[:, col_idx][valid_t]
        if len(y) == 0 or np.all(np.isnan(y)):
            continue
        channels[channel_names[col_idx - 1]] = {
            "x": t.copy(),
            "y": y,
            "unit": "",
        }

    return _new_single_group_dataset(group_name, group_props=group_props, channels=channels, root_props=root_props)


def dataset_from_vib(file_path):
    with open(file_path, "r", encoding="cp1250") as f:
        lines = f.readlines()

    def line_value(index):
        if 0 <= index < len(lines):
            s = lines[index].strip()
            return s if s else None
        return None

    def find_prefixed_value(prefix):
        for ln in lines:
            s = ln.strip()
            if s.startswith(prefix):
                parts = s.split(":", 1)
                if len(parts) == 2:
                    val = parts[1].strip()
                    return val if val else None
        return None

    date_time_line = line_value(2)
    object_name = line_value(3)
    measurement = line_value(4)
    extra_1 = line_value(5)
    extra_2 = line_value(6)

    presample_rate = find_prefixed_value("Presample scan rate [S/s]")
    scan_rate = find_prefixed_value("Scan rate [S/s]")
    highpass = find_prefixed_value("Highpass filter [Hz]")
    lowpass = find_prefixed_value("Lowpass filter [Hz]")

    channel_desc = {}
    channel_desc_re = re.compile(r"^\s*Channel\s+(\d+)\s*:\s*(.*?)\s*$")
    header_line_idx = None

    for i, ln in enumerate(lines):
        s = ln.strip()
        m = channel_desc_re.match(s)
        if m:
            channel_desc[int(m.group(1))] = m.group(2).strip()
        if "\t" in s and re.search(r"\[[^\]]+\]", s):
            header_line_idx = i
            break

    if header_line_idx is None:
        raise RuntimeError("VIB header line not found.")

    group_name = measurement or "VIB Data"
    group_props = {
        "Project": object_name,
        "Measurement": measurement,
        "State1": extra_1,
        "State2": extra_2,
        "DateTime": date_time_line,
        "PresampleRate": presample_rate,
        "SamplingRate": scan_rate,
        "HighpassFilter": highpass,
        "LowpassFilter": lowpass,
        "SourceFormat": "VIB",
    }
    group_props = {k: v for k, v in group_props.items() if v is not None}
    root_props = {"SourceFile": file_path}

    header_cols = [c.strip() for c in lines[header_line_idx].rstrip("\r\n").split("\t")]
    while header_cols and header_cols[-1] == "":
        header_cols.pop()
    if not header_cols:
        raise RuntimeError("VIB header columns not found.")

    col_defs = []
    for idx, col in enumerate(header_cols):
        m = re.match(r"^(.*?)\s*\[(.*?)\]\s*$", col)
        if m:
            name = m.group(1).strip()
            unit = m.group(2).strip()
        else:
            name = col.strip()
            unit = ""
        desc = channel_desc.get(idx + 1)
        final_name = desc if desc else name if name else f"ch{idx + 1}"
        col_defs.append({"name": final_name, "unit": unit})

    rows = []
    max_cols = len(col_defs)
    for line in lines[header_line_idx + 1:]:
        s = line.strip()
        if not s or s == "-":
            continue

        parts = line.rstrip("\r\n").split("\t")
        while parts and parts[-1].strip() == "":
            parts.pop()

        if len(parts) < max_cols:
            parts = parts + [""] * (max_cols - len(parts))
        elif len(parts) > max_cols:
            parts = parts[:max_cols]

        vals = []
        row_ok = True
        for p in parts:
            p = p.strip()
            if p == "":
                vals.append(np.nan)
                continue
            try:
                vals.append(float(p.replace(",", ".")))
            except ValueError:
                row_ok = False
                break

        if row_ok:
            rows.append(vals)

    if not rows:
        raise RuntimeError("No valid VIB samples found.")

    data = np.asarray(rows, dtype=float)
    samples = data.shape[0]

    fs = None
    if scan_rate is not None:
        try:
            fs = float(scan_rate.replace(",", "."))
        except ValueError:
            fs = None

    if fs is None or fs <= 0:
        raise RuntimeError("Invalid or missing VIB scan rate.")

    dt = 1.0 / fs
    t = np.arange(samples, dtype=float) * dt

    channels = {}
    for col_idx, coldef in enumerate(col_defs):
        y = data[:, col_idx]
        if len(y) == 0 or np.all(np.isnan(y)):
            continue
        channels[coldef["name"]] = {
            "x": t.copy(),
            "y": y,
            "unit": coldef["unit"],
        }

    return _new_single_group_dataset(group_name, group_props=group_props, channels=channels, root_props=root_props)


def merge_datasets(base_dataset, incoming_dataset):
    if incoming_dataset.root_props and not base_dataset.root_props:
        base_dataset.root_props = dict(incoming_dataset.root_props)

    for group_name, group_data in incoming_dataset.groups.items():
        unique_name = base_dataset.ensure_unique_group_name(group_name)
        base_dataset.groups[unique_name] = {
            "props": copy.deepcopy(group_data["props"]),
            "channels": copy.deepcopy(group_data["channels"]),
        }


def dataset_to_tdms(dataset, out_tdms):
    if not dataset.has_groups():
        raise RuntimeError("Dataset contains no groups.")

    objects = [RootObject(properties=dataset.root_props)]

    for group_name, group_data in dataset.groups.items():
        group_props = dict(group_data["props"])
        channels = group_data["channels"]

        objects.append(GroupObject(group_name, properties=group_props))

        for name, data in channels.items():
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)

            if len(x) < 2 or len(y) < 2:
                raise RuntimeError(f"Channel '{group_name}/{name}' has fewer than 2 samples.")
            if len(x) != len(y):
                raise RuntimeError(f"Channel '{group_name}/{name}' has mismatched X/Y length.")

            diffs = np.diff(x)
            dt = float(np.median(diffs))
            t0 = float(x[0])
            max_dev = float(np.max(np.abs(diffs - dt)))
            tol = max(1e-9, abs(dt) * 1e-6)
            if max_dev > tol:
                raise RuntimeError(f"Channel '{group_name}/{name}' is not uniformly sampled.")

            props = {
                "wf_increment": np.float64(dt),
                "wf_start_offset": np.float64(t0),
                "wf_samples": np.int32(len(y)),
                "wf_xname": "Time",
                "wf_xunit_string": "s",
            }

            unit = str(data.get("unit", "")).strip()
            if unit:
                props["unit_string"] = unit

            objects.append(ChannelObject(group_name, name, y, properties=props))

    with TdmsWriter(out_tdms) as writer:
        writer.write_segment(objects)


class TdmsPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1650, 1000)

        self.dataset = SignalDataset()
        self.channel_map = {}
        self.plotted_data = {}
        self.undo_stack = []
        self.max_undo_steps = 20
        self.group_selection_state = {}
        self._last_group_name = ""

        self.filter_settings = {
            "moving_average": {"window_samples": 5},
            "lowpass": {"cutoff_hz": 10.0, "order": 4},
            "highpass": {"cutoff_hz": 1.0, "order": 4},
            "bandpass": {"low_cutoff_hz": 1.0, "high_cutoff_hz": 10.0, "order": 4},
            "bandpass_sos": {"low_cutoff_hz": 1.0, "high_cutoff_hz": 10.0, "order": 4},
            "moving_pkpk": {
                "window_sec": 1.0,
                "use_marker": False,
                "marker_channel": "",
                "threshold": 0.0,
                "use_instantaneous": False,
            },
            "moving_rms": {
                "window_sec": 1.0,
                "use_marker": False,
                "marker_channel": "",
                "threshold": 0.0,
                "use_instantaneous": False,
            },
            "subtract_mean": {},
        }

        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._connect_signals()
        self.update_bottom_y_controls_visibility()

    def current_group_name(self):
        return self.group_combo.currentText().strip()

    def current_group(self):
        return self.dataset.get_group(self.current_group_name())

    def _reset_workspace(self):
        self.dataset = SignalDataset()
        self.channel_map.clear()
        self.plotted_data.clear()
        self.undo_stack.clear()
        self.group_selection_state = {}
        self._last_group_name = ""

        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.blockSignals(False)

        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_list.blockSignals(False)

        self.file_label.setText("No file loaded")
        self.info_box.clear()
        self.band_table.setRowCount(0)
        self.band_label.setText("X1: -, X2: -, dX: -")

        self.band_checkbox.setChecked(True)
        self.xy_mode_checkbox.setChecked(False)
        self.bottom_enable_autoscale_y_checkbox.setChecked(False)
        self.update_bottom_y_controls_visibility()

        self.plot.clear()
        self.legend = self.plot.addLegend()
        self.legend.anchor((1, 0), (1, 0))
        self.legend.setOffset((-10, 10))
        self.plot.addItem(self.region)
        self.region.setRegion((0.0, 1.0))
        self.region.setVisible(True)

        self.band_plot.clear()
        self.band_legend = self.band_plot.addLegend()
        self.band_legend.anchor((1, 0), (1, 0))
        self.band_legend.setOffset((-10, 10))
        self.band_plot.setVisible(True)

        for idx, pair in enumerate(self.xy_pairs):
            pair["enable"].setChecked(idx == 0)
            pair["x_combo"].blockSignals(True)
            pair["y_combo"].blockSignals(True)
            pair["x_combo"].clear()
            pair["y_combo"].clear()
            pair["x_combo"].blockSignals(False)
            pair["y_combo"].blockSignals(False)

    def new_project(self):
        try:
            has_workspace_data = self.dataset.has_groups() or bool(self.undo_stack)
            if has_workspace_data:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "New Project",
                    "Start a new project and clear the current workspace?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return

            self._reset_workspace()
            self.statusBar().showMessage("New project created")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "New Project Error", str(e))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()

        self.open_tdms_button = QtWidgets.QPushButton("Open TDMS")
        self.open_slab_button = QtWidgets.QPushButton("Open sLAB")
        self.open_srm_button = QtWidgets.QPushButton("Open SRM")
        self.open_vib_button = QtWidgets.QPushButton("Open VIB")

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("file_label")

        self.group_combo = QtWidgets.QComboBox()
        self.group_combo.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.group_combo.view().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.group_combo.setObjectName("group_combo")

        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.channel_list.setObjectName("channel_list")

        left.addWidget(self.open_tdms_button)
        left.addWidget(self.open_slab_button)
        left.addWidget(self.open_srm_button)
        left.addWidget(self.open_vib_button)
        left.addWidget(self.file_label)
        left.addWidget(QtWidgets.QLabel("Group"))
        left.addWidget(self.group_combo)
        left.addWidget(QtWidgets.QLabel("Channels"))
        left.addWidget(self.channel_list)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)

        self.viewbox = CustomViewBox()

        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget(viewBox=self.viewbox)
        self.plot.showGrid(x=True, y=True)
        self.plot.setBackground("w")
        for axis_name in ("left", "bottom"):
            axis = self.plot.getAxis(axis_name)
            axis.setTextPen("k")
            axis.setPen("k")

        self.legend = self.plot.addLegend()
        self.legend.anchor((1, 0), (1, 0))
        self.legend.setOffset((-10, 10))

        self.region = pg.LinearRegionItem()
        self.region.sigRegionChanged.connect(self.update_band_views)
        self.plot.addItem(self.region)

        self.band_plot = pg.PlotWidget()
        self.band_plot.showGrid(x=True, y=True)
        self.band_plot.setBackground("w")
        self.band_plot.setMinimumHeight(180)
        self.band_plot.setLabel("bottom", "Time / X (Band Range)")
        self.band_plot.setLabel("left", "Value")
        for axis_name in ("left", "bottom"):
            axis = self.band_plot.getAxis(axis_name)
            axis.setTextPen("k")
            axis.setPen("k")

        self.band_legend = self.band_plot.addLegend()
        self.band_legend.anchor((1, 0), (1, 0))
        self.band_legend.setOffset((-10, 10))

        self.xy_mode_checkbox = QtWidgets.QCheckBox("XY plot")
        self.xy_mode_checkbox.setObjectName("xy_mode_checkbox")

        self.bottom_autoscale_y_once_button = QtWidgets.QPushButton("Auto-scale Y once")

        self.bottom_enable_autoscale_y_checkbox = QtWidgets.QCheckBox("Enable autoscale Y")
        self.bottom_enable_autoscale_y_checkbox.setChecked(False)
        self.bottom_enable_autoscale_y_checkbox.setObjectName("bottom_enable_autoscale_y_checkbox")

        self.xy_pairs = []
        for i in range(4):
            enable_cb = QtWidgets.QCheckBox()
            enable_cb.setChecked(i == 0)

            x_label = QtWidgets.QLabel(f"X{i + 1}:")
            y_label = QtWidgets.QLabel(f"Y{i + 1}:")
            x_combo = QtWidgets.QComboBox()
            y_combo = QtWidgets.QComboBox()

            x_label.setMinimumWidth(24)
            y_label.setMinimumWidth(24)
            x_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            y_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            x_combo.setMinimumWidth(140)
            y_combo.setMinimumWidth(140)

            enable_cb.setVisible(False)
            x_label.setVisible(False)
            y_label.setVisible(False)
            x_combo.setVisible(False)
            y_combo.setVisible(False)

            enable_cb.setObjectName(f"xy_enable_{i}")
            x_combo.setObjectName(f"xy_x_combo_{i}")
            y_combo.setObjectName(f"xy_y_combo_{i}")

            self.xy_pairs.append({
                "enable": enable_cb,
                "x_label": x_label,
                "y_label": y_label,
                "x_combo": x_combo,
                "y_combo": y_combo,
            })

        band_bottom_widget = QtWidgets.QWidget()
        band_bottom_layout = QtWidgets.QHBoxLayout(band_bottom_widget)
        band_bottom_layout.setContentsMargins(0, 0, 0, 0)

        xy_controls_widget = QtWidgets.QWidget()
        xy_controls_layout = QtWidgets.QVBoxLayout(xy_controls_widget)
        xy_controls_layout.setContentsMargins(4, 4, 4, 4)

        top_controls_row = QtWidgets.QHBoxLayout()
        top_controls_row.addWidget(self.xy_mode_checkbox)
        top_controls_row.addStretch()
        top_controls_row.addWidget(self.bottom_autoscale_y_once_button)
        top_controls_row.addWidget(self.bottom_enable_autoscale_y_checkbox)
        xy_controls_layout.addLayout(top_controls_row)

        for pair in self.xy_pairs:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(pair["enable"])
            row.addWidget(pair["x_label"])
            row.addWidget(pair["x_combo"], 1)
            row.addWidget(pair["y_label"])
            row.addWidget(pair["y_combo"], 1)
            xy_controls_layout.addLayout(row)

        xy_controls_layout.addStretch()

        self.xy_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.xy_splitter.addWidget(self.band_plot)
        self.xy_splitter.addWidget(xy_controls_widget)
        self.xy_splitter.setSizes([800, 420])
        self.xy_splitter.setChildrenCollapsible(False)
        self.xy_splitter.setObjectName("xy_splitter")

        band_bottom_layout.addWidget(self.xy_splitter)

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_splitter.addWidget(self.plot)
        self.plot_splitter.addWidget(band_bottom_widget)
        self.plot_splitter.setSizes([720, 280])
        self.plot_splitter.setChildrenCollapsible(False)
        self.plot_splitter.setObjectName("plot_splitter")

        plot_layout.addWidget(self.plot_splitter)

        right = QtWidgets.QVBoxLayout()

        self.band_checkbox = QtWidgets.QCheckBox("Enable band")
        self.band_checkbox.setChecked(True)
        self.band_checkbox.setObjectName("band_checkbox")

        self.band_label = QtWidgets.QLabel("X1: -, X2: -, dX: -")
        self.band_label.setObjectName("band_label")

        self.band_table = QtWidgets.QTableWidget(0, 12)
        self.band_table.setObjectName("band_table")
        self.band_table.setHorizontalHeaderLabels([
            "Channel", "Unit", "Y@X1", "Y@X2", "Delta Y", "Mean",
            "Min", "Max", "PkPk", "StdDev", "RMS", "AC RMS",
        ])
        self.band_table.verticalHeader().setVisible(False)

        info_group = QtWidgets.QGroupBox("Info")
        info_layout = QtWidgets.QVBoxLayout()
        self.info_box = QtWidgets.QPlainTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(190)
        self.info_box.setObjectName("info_box")
        info_layout.addWidget(self.info_box)
        info_group.setLayout(info_layout)

        right.addWidget(self.band_checkbox)
        right.addWidget(self.band_label)
        right.addWidget(self.band_table)
        right.addWidget(info_group)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(plot_container)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setSizes([340, 980, 440])
        self.main_splitter.setObjectName("main_splitter")

        layout.addWidget(self.main_splitter)
        self.statusBar().showMessage(f"{APP_NAME} {APP_VERSION} ready")

    def _build_actions(self):
        self.action_new_project = QtWidgets.QAction("New Project", self)
        self.action_open_tdms = QtWidgets.QAction("Open TDMS", self)
        self.action_open_slab = QtWidgets.QAction("Open sLAB File", self)
        self.action_open_srm = QtWidgets.QAction("Open SRM File", self)
        self.action_open_vib = QtWidgets.QAction("Open VIB File", self)
        self.action_open_project = QtWidgets.QAction("Open Project...", self)
        self.action_save_project = QtWidgets.QAction("Save Project...", self)
        self.action_export_dataset = QtWidgets.QAction("Export Dataset TDMS", self)

        self.action_undo = QtWidgets.QAction("Undo", self)
        self.action_undo.setShortcut(QtGui.QKeySequence.Undo)

        self.zoom_rect = QtWidgets.QAction("Box Zoom", self)
        self.zoom_rect.setCheckable(True)
        self.zoom_x = QtWidgets.QAction("Zoom X", self)
        self.zoom_x.setCheckable(True)
        self.zoom_y = QtWidgets.QAction("Zoom Y", self)
        self.zoom_y.setCheckable(True)
        self.pan = QtWidgets.QAction("Pan", self)
        self.pan.setCheckable(True)

        self.zoom_band = QtWidgets.QAction("Zoom to Band", self)
        self.reset = QtWidgets.QAction("Reset View", self)
        self.autorange = QtWidgets.QAction("Auto Range", self)
        self.action_bottom_y_autoscale = QtWidgets.QAction("Auto-scale Bottom Y Once", self)

        self.action_filter_moving_average = QtWidgets.QAction("Moving Average...", self)
        self.action_filter_lowpass = QtWidgets.QAction("Low-pass...", self)
        self.action_filter_highpass = QtWidgets.QAction("High-pass...", self)
        self.action_filter_bandpass = QtWidgets.QAction("Band-pass...", self)
        self.action_filter_bandpass_sos = QtWidgets.QAction("Band-pass (Stable SOS)...", self)
        self.action_moving_pkpk = QtWidgets.QAction("Moving Window Pk-Pk...", self)
        self.action_moving_rms = QtWidgets.QAction("Moving Window RMS...", self)
        self.action_subtract_mean = QtWidgets.QAction("Subtract Mean", self)

    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.action_new_project)
        file_menu.addSeparator()
        file_menu.addAction(self.action_open_tdms)
        file_menu.addAction(self.action_open_slab)
        file_menu.addAction(self.action_open_srm)
        file_menu.addAction(self.action_open_vib)
        file_menu.addSeparator()
        file_menu.addAction(self.action_open_project)
        file_menu.addAction(self.action_save_project)
        file_menu.addSeparator()
        file_menu.addAction(self.action_export_dataset)

        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self.action_undo)

        zoom_menu = menubar.addMenu("Zoom")
        zoom_menu.addAction(self.zoom_rect)
        zoom_menu.addAction(self.zoom_x)
        zoom_menu.addAction(self.zoom_y)
        zoom_menu.addAction(self.pan)
        zoom_menu.addSeparator()
        zoom_menu.addAction(self.zoom_band)
        zoom_menu.addSeparator()
        zoom_menu.addAction(self.reset)
        zoom_menu.addAction(self.autorange)
        zoom_menu.addSeparator()
        zoom_menu.addAction(self.action_bottom_y_autoscale)

        filter_menu = menubar.addMenu("Filters")
        filter_menu.addAction(self.action_filter_moving_average)
        filter_menu.addAction(self.action_filter_lowpass)
        filter_menu.addAction(self.action_filter_highpass)
        filter_menu.addAction(self.action_filter_bandpass)
        filter_menu.addAction(self.action_filter_bandpass_sos)
        filter_menu.addAction(self.action_moving_pkpk)
        filter_menu.addAction(self.action_moving_rms)
        filter_menu.addAction(self.action_subtract_mean)

    def _connect_signals(self):
        self.open_tdms_button.clicked.connect(self.open_tdms)
        self.open_slab_button.clicked.connect(self.open_slab)
        self.open_srm_button.clicked.connect(self.open_srm)
        self.open_vib_button.clicked.connect(self.open_vib)

        self.action_new_project.triggered.connect(self.new_project)
        self.action_open_tdms.triggered.connect(self.open_tdms)
        self.action_open_slab.triggered.connect(self.open_slab)
        self.action_open_srm.triggered.connect(self.open_srm)
        self.action_open_vib.triggered.connect(self.open_vib)
        self.action_open_project.triggered.connect(self.open_project)
        self.action_save_project.triggered.connect(self.save_project)
        self.action_export_dataset.triggered.connect(self.export_dataset)
        self.action_undo.triggered.connect(self.undo_last_action)

        self.action_filter_moving_average.triggered.connect(self.filter_moving_average)
        self.action_filter_lowpass.triggered.connect(self.filter_lowpass)
        self.action_filter_highpass.triggered.connect(self.filter_highpass)
        self.action_filter_bandpass.triggered.connect(self.filter_bandpass)
        self.action_filter_bandpass_sos.triggered.connect(self.filter_bandpass_sos)
        self.action_moving_pkpk.triggered.connect(self.filter_moving_window_pkpk)
        self.action_moving_rms.triggered.connect(self.filter_moving_window_rms)
        self.action_subtract_mean.triggered.connect(self.filter_subtract_mean)

        self.group_combo.currentIndexChanged.connect(self._handle_group_change)
        self.group_combo.customContextMenuRequested.connect(self.show_group_context_menu)
        self.group_combo.view().customContextMenuRequested.connect(self.show_group_view_context_menu)

        self.channel_list.itemSelectionChanged.connect(self._on_channel_selection_changed)
        self.channel_list.customContextMenuRequested.connect(self.show_channel_context_menu)

        self.zoom_rect.triggered.connect(lambda: self.viewbox.set_mode("rect"))
        self.zoom_x.triggered.connect(lambda: self.viewbox.set_mode("x"))
        self.zoom_y.triggered.connect(lambda: self.viewbox.set_mode("y"))
        self.pan.triggered.connect(lambda: self.viewbox.set_mode("pan"))

        self.zoom_band.triggered.connect(self.zoom_to_band)
        self.reset.triggered.connect(lambda: self.plot.enableAutoRange())
        self.autorange.triggered.connect(lambda: self.plot.autoRange())
        self.action_bottom_y_autoscale.triggered.connect(self.auto_scale_bottom_y_once)

        self.band_checkbox.toggled.connect(self.toggle_band)
        self.xy_mode_checkbox.toggled.connect(self.toggle_xy_mode)
        self.bottom_autoscale_y_once_button.clicked.connect(self.auto_scale_bottom_y_once)
        self.bottom_enable_autoscale_y_checkbox.toggled.connect(self.on_bottom_enable_autoscale_y_toggled)

        for pair in self.xy_pairs:
            pair["enable"].toggled.connect(self._update_band_plot)
            pair["x_combo"].currentIndexChanged.connect(self._update_band_plot)
            pair["y_combo"].currentIndexChanged.connect(self._update_band_plot)

    def update_bottom_y_controls_visibility(self):
        autoscale_enabled = self.bottom_enable_autoscale_y_checkbox.isChecked()
        self.bottom_autoscale_y_once_button.setVisible(not autoscale_enabled)

    def on_bottom_enable_autoscale_y_toggled(self, checked):
        self.update_bottom_y_controls_visibility()
        self._update_band_plot()

    def _push_undo_state(self, description=""):
        state = {
            "dataset": copy.deepcopy(self.dataset),
            "selected_group": self.current_group_name(),
            "group_selection_state": copy.deepcopy(self.group_selection_state),
            "description": description,
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    def undo_last_action(self):
        try:
            if not self.undo_stack:
                self.statusBar().showMessage("Nothing to undo")
                return

            state = self.undo_stack.pop()
            self.dataset = state["dataset"]
            self.group_selection_state = state.get("group_selection_state", {})
            self.populate_groups()

            prev_group = state.get("selected_group", "")
            idx = self.group_combo.findText(prev_group)
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)
            else:
                self.populate_channels()
                self.refresh_group_channel_highlight()
                self.plot_channels()

            self.update_info_panel()
            self._last_group_name = self.current_group_name()
            self.statusBar().showMessage(f"Undo: {state.get('description', 'last action')}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Undo Error", f"Could not undo last action.\n\n{e}")

    def _minmax_envelope_downsample(self, x, y, max_points):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        n = len(x)
        if n <= max_points or max_points <= 0:
            return x, y

        if n < 3 or max_points < 4:
            idx = np.linspace(0, n - 1, num=min(max_points, n), dtype=int)
            idx = np.unique(idx)
            return x[idx], y[idx]

        buckets = max(1, max_points // 2)
        edges = np.linspace(0, n, buckets + 1, dtype=int)

        xs = [x[0]]
        ys = [y[0]]

        for i in range(buckets):
            start = edges[i]
            end = edges[i + 1]
            if end <= start:
                continue

            xb = x[start:end]
            yb = y[start:end]

            if len(yb) == 1:
                xs.append(xb[0])
                ys.append(yb[0])
                continue

            i_min = int(np.argmin(yb))
            i_max = int(np.argmax(yb))
            if i_min <= i_max:
                xs.extend([xb[i_min], xb[i_max]])
                ys.extend([yb[i_min], yb[i_max]])
            else:
                xs.extend([xb[i_max], xb[i_min]])
                ys.extend([yb[i_max], yb[i_min]])

        if xs[-1] != x[-1] or ys[-1] != y[-1]:
            xs.append(x[-1])
            ys.append(y[-1])

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)

        if len(xs) > max_points:
            idx = np.linspace(0, len(xs) - 1, num=max_points, dtype=int)
            idx = np.unique(idx)
            xs = xs[idx]
            ys = ys[idx]

        return xs, ys

    def _downsample_band_xy(self, x, y, lo, hi, max_points):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = (x >= lo) & (x <= hi)
        xb = x[mask]
        yb = y[mask]
        if len(xb) == 0:
            return xb, yb
        return self._minmax_envelope_downsample(xb, yb, max_points)

    def _preserve_band_plot_y_range(self):
        try:
            vb = self.band_plot.getViewBox()
            y_range = vb.viewRange()[1]
            return float(y_range[0]), float(y_range[1])
        except Exception:
            return None

    def _restore_band_plot_y_range(self, y_range):
        if y_range is None:
            return
        y0, y1 = y_range
        self.band_plot.setYRange(y0, y1, padding=0)

    def auto_scale_bottom_y_once(self):
        try:
            if not self.band_checkbox.isChecked() or not self.plotted_data:
                return
            self.band_plot.enableAutoRange(axis="y", enable=True)
            self.band_plot.autoRange()
            self.band_plot.enableAutoRange(axis="y", enable=False)
            self.statusBar().showMessage("Bottom graph Y auto-scaled once")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Bottom Y Auto-scale Error", str(e))

    def toggle_band(self, checked):
        try:
            enabled = bool(checked)
            self.region.setVisible(enabled)
            self.band_plot.setVisible(enabled)
            self.xy_mode_checkbox.setEnabled(enabled)
            self.bottom_autoscale_y_once_button.setEnabled(enabled)
            self.bottom_enable_autoscale_y_checkbox.setEnabled(enabled)

            for pair in self.xy_pairs:
                pair["enable"].setEnabled(enabled)
                pair["x_combo"].setEnabled(enabled)
                pair["y_combo"].setEnabled(enabled)

            if enabled:
                self.update_band_views()
                self.statusBar().showMessage("Band enabled")
            else:
                self.band_label.setText("Band disabled")
                self.band_table.setRowCount(0)
                self.band_plot.clear()
                self.band_legend = self.band_plot.addLegend()
                self.band_legend.anchor((1, 0), (1, 0))
                self.band_legend.setOffset((-10, 10))
                self.statusBar().showMessage("Band disabled")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Band Error", str(e))

    def update_info_panel(self):
        group_name = self.current_group_name()
        group = self.current_group()

        if not group_name or group_name not in self.dataset.groups:
            self.info_box.clear()
            return

        lines = []
        lines.append("Root properties:")
        if self.dataset.root_props:
            for k, v in self.dataset.root_props.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  -")

        lines.append("")
        lines.append(f"Group: {group_name}")

        if group["props"]:
            for k, v in group["props"].items():
                lines.append(f"{k}: {v}")

        lines.append("")

        channels = list(group["channels"].values())
        if channels:
            first = channels[0]
            x = np.asarray(first["x"], dtype=float)
            samples = len(x)

            if samples > 1:
                dt = float(np.median(np.diff(x)))
                sampling_rate = 1.0 / dt if dt != 0 else 0.0
                duration = float(x[-1] - x[0])
            else:
                dt = 0.0
                sampling_rate = 0.0
                duration = 0.0

            lines.append(f"Sampling rate: {sampling_rate:.6g} Hz")
            lines.append(f"dt: {dt:.6g} s")
            lines.append(f"Number of samples: {samples}")
            lines.append(f"Number of channels in group: {len(group['channels'])}")
            lines.append(f"Total channels in dataset: {self.dataset.total_channel_count()}")
            lines.append(f"Selected channels in this group: {len(self.group_selection_state.get(group_name, []))}")
            lines.append(f"Duration: {duration:.6g} s")

        self.info_box.setPlainText("\n".join(lines))

    def populate_groups(self):
        current = self.current_group_name()

        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for group_name in self.dataset.get_group_names():
            self.group_combo.addItem(group_name)
        self.group_combo.blockSignals(False)

        idx = self.group_combo.findText(current)
        if idx >= 0:
            self.group_combo.setCurrentIndex(idx)
        elif self.group_combo.count() > 0:
            self.group_combo.setCurrentIndex(0)

    def _load_dataset(self, dataset, file_label):
        self.dataset = dataset
        self.group_selection_state = {g: [] for g in self.dataset.get_group_names()}
        self.file_label.setText(file_label)

        self.populate_groups()
        self.populate_channels()
        self.refresh_group_channel_highlight()
        self.update_info_panel()
        self.plot_channels()

        self._last_group_name = self.current_group_name()
        self.statusBar().showMessage(f"Loaded: {file_label}")

    def _append_dataset(self, incoming_dataset, file_path):
        if not self.dataset.has_groups():
            self.dataset = SignalDataset()

        old_group_names = set(self.dataset.get_group_names())
        merge_datasets(self.dataset, incoming_dataset)
        new_group_names = self.dataset.get_group_names()

        for group_name in new_group_names:
            self.group_selection_state.setdefault(group_name, [])

        self.file_label.setText(f"{len(new_group_names)} groups loaded")
        self.populate_groups()

        added_groups = [g for g in new_group_names if g not in old_group_names]
        if added_groups:
            idx = self.group_combo.findText(added_groups[-1])
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)

        self.populate_channels()
        self.refresh_group_channel_highlight()
        self.update_info_panel()
        self.plot_channels()
        self._last_group_name = self.current_group_name()
        self.statusBar().showMessage(f"Added file: {file_path}")

    def open_tdms(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open TDMS", "", "TDMS files (*.tdms)")
        if not path:
            return
        try:
            dataset = dataset_from_tdms(TdmsFile.read(path))
            if self.dataset.has_groups():
                self._append_dataset(dataset, path)
            else:
                self._load_dataset(dataset, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Error", str(e))

    def open_slab(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open sLAB File", "", "sLAB files (*.xls *.txt *.dat);;All files (*.*)"
        )
        if not path:
            return
        try:
            dataset = dataset_from_slab(path)
            if self.dataset.has_groups():
                self._append_dataset(dataset, path)
            else:
                self._load_dataset(dataset, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "sLAB Import Error", str(e))

    def open_srm(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SRM File", "", "SRM files (*.xls *.txt *.dat);;All files (*.*)"
        )
        if not path:
            return
        try:
            dataset = dataset_from_srm(path)
            if self.dataset.has_groups():
                self._append_dataset(dataset, path)
            else:
                self._load_dataset(dataset, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SRM Import Error", str(e))

    def open_vib(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open VIB File", "", "VIB files (*.*)")
        if not path:
            return
        try:
            dataset = dataset_from_vib(path)
            if self.dataset.has_groups():
                self._append_dataset(dataset, path)
            else:
                self._load_dataset(dataset, path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "VIB Import Error", str(e))

    def export_dataset(self):
        try:
            if not self.dataset.has_groups():
                QtWidgets.QMessageBox.information(self, "Export", "No dataset loaded.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export Dataset to TDMS", "", "TDMS files (*.tdms)"
            )
            if not path:
                return
            dataset_to_tdms(self.dataset, path)
            self.statusBar().showMessage(f"Exported dataset Ă„â€šĂ˘â‚¬ĹľÄ‚ËĂ˘â€šÂ¬ÄąË‡Ă„â€šĂ˘â‚¬Ä…Ä‚â€šĂ‚ÂÄ‚â€žĂ˘â‚¬ĹˇÄ‚â€ąĂ‚ÂĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąË‡Ä‚â€šĂ‚Â¬Ă„â€šĂ˘â‚¬ĹˇÄ‚â€šĂ‚Â Ä‚â€žĂ˘â‚¬ĹˇÄ‚â€ąĂ‚ÂĂ„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąË‡Ä‚â€šĂ‚Â¬Ă„â€šĂ‹ÂÄ‚ËĂ˘â€šÂ¬ÄąÄľÄ‚â€ąĂ‚Â {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def populate_channels(self):
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_map.clear()

        group = self.current_group()
        for name, data in group["channels"].items():
            unit = str(data.get("unit", "")).strip()
            display_name = f"{name} [{unit}]" if unit else name
            self.channel_list.addItem(display_name)
            self.channel_map[display_name] = name

        self.channel_list.blockSignals(False)

    def refresh_group_channel_highlight(self):
        group_name = self.current_group_name().strip()
        if not group_name:
            return

        selected_names = set(self.group_selection_state.get(group_name, []))

        self.channel_list.blockSignals(True)
        self.channel_list.clearSelection()

        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            display_name = item.text()
            if display_name not in self.channel_map:
                continue
            base_name = self.channel_map[display_name]
            if base_name in selected_names:
                item.setSelected(True)

        self.channel_list.blockSignals(False)

    def _handle_group_change(self):
        if self._last_group_name:
            self._save_current_group_selection_for_name(self._last_group_name)

        current = self.current_group_name().strip()
        self.populate_channels()
        self.refresh_group_channel_highlight()
        self.update_info_panel()
        self.plot_channels()
        self._last_group_name = current

    def _on_channel_selection_changed(self):
        if self.channel_list.signalsBlocked():
            return
        self._save_current_group_selection()
        self.plot_channels()

    def _save_current_group_selection(self):
        self._save_current_group_selection_for_name(self.current_group_name())

    def _save_current_group_selection_for_name(self, group_name):
        group_name = str(group_name).strip()
        if not group_name:
            return

        selected = []
        for item in self.channel_list.selectedItems():
            display_name = item.text()
            if display_name in self.channel_map:
                selected.append(self.channel_map[display_name])

        self.group_selection_state[group_name] = selected

    def update_xy_channel_selectors(self):
        plot_names = list(self.plotted_data.keys())

        for idx, pair in enumerate(self.xy_pairs):
            x_combo = pair["x_combo"]
            y_combo = pair["y_combo"]

            current_x = x_combo.currentText()
            current_y = y_combo.currentText()

            x_combo.blockSignals(True)
            y_combo.blockSignals(True)
            x_combo.clear()
            y_combo.clear()

            for name in plot_names:
                x_combo.addItem(name)
                y_combo.addItem(name)

            if current_x and x_combo.findText(current_x) >= 0:
                x_combo.setCurrentText(current_x)
            elif x_combo.count() > 0:
                x_combo.setCurrentIndex(0)

            if current_y and y_combo.findText(current_y) >= 0:
                y_combo.setCurrentText(current_y)
            elif y_combo.count() > 0:
                default_index = min(idx, y_combo.count() - 1)
                y_combo.setCurrentIndex(default_index)

            x_combo.blockSignals(False)
            y_combo.blockSignals(False)

        if not any(pair["enable"].isChecked() for pair in self.xy_pairs):
            self.xy_pairs[0]["enable"].setChecked(True)

    def toggle_xy_mode(self, checked):
        visible = bool(checked)
        for pair in self.xy_pairs:
            pair["enable"].setVisible(visible)
            pair["x_label"].setVisible(visible)
            pair["y_label"].setVisible(visible)
            pair["x_combo"].setVisible(visible)
            pair["y_combo"].setVisible(visible)
        self._update_band_plot()

    def plot_channels(self):
        self.plot.clear()

        self.legend = self.plot.addLegend()
        self.legend.anchor((1, 0), (1, 0))
        self.legend.setOffset((-10, 10))

        self.plot.addItem(self.region)
        self.region.setVisible(self.band_checkbox.isChecked())

        self.plotted_data.clear()
        color_index = 0

        for group_name, group_data in self.dataset.groups.items():
            selected_channels = self.group_selection_state.get(group_name, [])
            for ch_name in selected_channels:
                if ch_name not in group_data["channels"]:
                    continue

                ch = group_data["channels"][ch_name]
                x = np.asarray(ch["x"], dtype=float)
                y = np.asarray(ch["y"], dtype=float)

                x_plot, y_plot = self._minmax_envelope_downsample(x, y, MAX_PLOT_POINTS_MAIN)
                display_name = f"{group_name} | {ch_name}"

                self.plot.plot(x_plot, y_plot, pen=pg.intColor(color_index), name=display_name)
                self.plotted_data[display_name] = {
                    "x": x,
                    "y": y,
                    "unit": ch.get("unit", ""),
                }
                color_index += 1

        self.plot.autoRange()
        self.update_xy_channel_selectors()
        self.update_band_views()

    def _update_band_plot(self):
        autoscale_y_enabled = self.bottom_enable_autoscale_y_checkbox.isChecked()
        y_range_before = None if autoscale_y_enabled else self._preserve_band_plot_y_range()

        self.band_plot.clear()
        self.band_legend = self.band_plot.addLegend()
        self.band_legend.anchor((1, 0), (1, 0))
        self.band_legend.setOffset((-10, 10))

        if not self.band_checkbox.isChecked():
            self.band_plot.setVisible(False)
            return

        self.band_plot.setVisible(True)
        if not self.plotted_data:
            return

        x1, x2 = self.region.getRegion()
        lo, hi = sorted((float(x1), float(x2)))
        if hi <= lo:
            return

        if self.xy_mode_checkbox.isChecked():
            plotted_any = False
            first_x_label = None
            first_y_label = None

            for i, pair in enumerate(self.xy_pairs):
                if not pair["enable"].isChecked():
                    continue

                x_name = pair["x_combo"].currentText().strip()
                y_name = pair["y_combo"].currentText().strip()
                if not x_name or not y_name:
                    continue
                if x_name not in self.plotted_data or y_name not in self.plotted_data:
                    continue

                x_data = self.plotted_data[x_name]
                y_data = self.plotted_data[y_name]

                tx = np.asarray(x_data["x"], dtype=float)
                ty = np.asarray(y_data["x"], dtype=float)
                xv = np.asarray(x_data["y"], dtype=float)
                yv = np.asarray(y_data["y"], dtype=float)

                mask_x = (tx >= lo) & (tx <= hi)
                mask_y = (ty >= lo) & (ty <= hi)

                txb = tx[mask_x]
                tyb = ty[mask_y]
                xvb = xv[mask_x]
                yvb = yv[mask_y]

                if len(txb) == 0 or len(tyb) == 0:
                    continue

                common_lo = max(np.min(txb), np.min(tyb))
                common_hi = min(np.max(txb), np.max(tyb))
                common_mask = (txb >= common_lo) & (txb <= common_hi)

                t_common = txb[common_mask]
                x_common = xvb[common_mask]
                if len(t_common) == 0:
                    continue

                y_common = np.interp(t_common, tyb, yvb)
                x_ds, y_ds = self._minmax_envelope_downsample(x_common, y_common, MAX_PLOT_POINTS_BAND)

                self.band_plot.plot(x_ds, y_ds, pen=pg.intColor(i), symbol=None, name=f"{y_name} vs {x_name}")

                if first_x_label is None:
                    x_unit = str(x_data.get("unit", "")).strip()
                    y_unit = str(y_data.get("unit", "")).strip()
                    first_x_label = x_name + (f" [{x_unit}]" if x_unit else "")
                    first_y_label = y_name + (f" [{y_unit}]" if y_unit else "")

                plotted_any = True

            if plotted_any:
                self.band_plot.setLabel("bottom", first_x_label or "X")
                self.band_plot.setLabel("left", first_y_label or "Y")
                self.band_plot.enableAutoRange(axis="x", enable=True)
                if autoscale_y_enabled:
                    self.band_plot.enableAutoRange(axis="y", enable=True)
                    self.band_plot.autoRange()
                else:
                    self.band_plot.enableAutoRange(axis="y", enable=False)
                    self._restore_band_plot_y_range(y_range_before)
            else:
                self.band_plot.setLabel("bottom", "X")
                self.band_plot.setLabel("left", "Y")
            return

        self.band_plot.setLabel("bottom", "Time / X (Band Range)")
        self.band_plot.setLabel("left", "Value")

        plotted_any = False
        for i, (name, data) in enumerate(self.plotted_data.items()):
            x = np.asarray(data["x"], dtype=float)
            y = np.asarray(data["y"], dtype=float)
            xb, yb = self._downsample_band_xy(x, y, lo, hi, MAX_PLOT_POINTS_BAND)
            if len(xb) == 0:
                continue
            self.band_plot.plot(xb, yb, pen=pg.intColor(i), name=name)
            plotted_any = True

        if plotted_any:
            self.band_plot.enableAutoRange(axis="x", enable=True)
            if autoscale_y_enabled:
                self.band_plot.enableAutoRange(axis="y", enable=True)
                self.band_plot.autoRange()
            else:
                self.band_plot.enableAutoRange(axis="y", enable=False)
                self._restore_band_plot_y_range(y_range_before)

    def update_band_views(self):
        self.update_band_table()
        self._update_band_plot()

    def zoom_to_band(self):
        try:
            if not self.band_checkbox.isChecked():
                return
            x1, x2 = self.region.getRegion()
            lo, hi = sorted((float(x1), float(x2)))
            if hi <= lo:
                return
            self.plot.setXRange(lo, hi, padding=0)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Zoom Error", str(e))

    def update_band_table(self):
        try:
            if not self.band_checkbox.isChecked():
                return

            if not self.plotted_data:
                self.band_label.setText("X1: -, X2: -, dX: -")
                self.band_table.setRowCount(0)
                return

            x1, x2 = self.region.getRegion()
            self.band_label.setText(f"X1: {x1:.4g}   X2: {x2:.4g}   dX: {x2 - x1:.4g}")
            self.band_table.setRowCount(len(self.plotted_data))

            for row, (name, data) in enumerate(self.plotted_data.items()):
                x = np.asarray(data["x"], dtype=float)
                y = np.asarray(data["y"], dtype=float)
                unit = str(data["unit"])

                y1 = np.interp(x1, x, y)
                y2 = np.interp(x2, x, y)

                lo, hi = sorted((x1, x2))
                mask = (x >= lo) & (x <= hi)
                samples = y[mask]

                if len(samples):
                    mean = np.mean(samples)
                    minv = np.min(samples)
                    maxv = np.max(samples)
                    pkpk = maxv - minv
                    std = np.std(samples)
                    rms = np.sqrt(np.mean(samples ** 2))
                    ac_rms = np.sqrt(np.mean((samples - mean) ** 2))
                else:
                    mean = minv = maxv = pkpk = std = rms = ac_rms = np.nan

                values = [name, unit, y1, y2, y2 - y1, mean, minv, maxv, pkpk, std, rms, ac_rms]
                for col, val in enumerate(values):
                    text = f"{val:.6g}" if isinstance(val, (float, np.floating)) else str(val)
                    self.band_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

            self.band_table.resizeColumnsToContents()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Band Error", str(e))

    def _selected_base_channel_names(self):
        names = []
        for item in self.channel_list.selectedItems():
            display_name = item.text()
            if display_name in self.channel_map:
                names.append(self.channel_map[display_name])
        return names

    def _refresh_channels_after_processing(self, new_channel_names):
        group_name = self.current_group_name()
        self.group_selection_state[group_name] = list(new_channel_names)
        self.populate_channels()
        self.refresh_group_channel_highlight()
        self.plot_channels()

    def _get_dt_for_channel(self, channel_name):
        group = self.current_group()
        x = np.asarray(group["channels"][channel_name]["x"], dtype=float)
        if len(x) < 2:
            raise RuntimeError(f"Channel '{channel_name}' has fewer than 2 samples.")
        dt = float(np.median(np.diff(x)))
        if dt <= 0:
            raise RuntimeError(f"Channel '{channel_name}' has invalid time step.")
        return dt

    def _create_filtered_channel(self, source_name, new_name, y_new):
        group = self.current_group()
        src = group["channels"][source_name]
        group["channels"][new_name] = {
            "x": np.asarray(src["x"], dtype=float).copy(),
            "y": np.asarray(y_new, dtype=float),
            "unit": src.get("unit", ""),
        }

    def filter_moving_average(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["moving_average"]
            window, ok = QtWidgets.QInputDialog.getInt(
                self, "Moving Average", "Window length (samples):",
                int(cfg.get("window_samples", 5)), 2, 100000, 1
            )
            if not ok:
                return

            cfg["window_samples"] = int(window)

            self._push_undo_state(f"moving average filter ({window})")
            kernel = np.ones(window, dtype=float) / float(window)
            new_names = []

            for ch_name in selected:
                y = np.asarray(self.current_group()["channels"][ch_name]["y"], dtype=float)
                y_filt = np.convolve(y, kernel, mode="same")
                new_name = f"{ch_name}_MA{window}"
                self._create_filtered_channel(ch_name, new_name, y_filt)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} moving-average channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_lowpass(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["lowpass"]
            cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "Low-pass Filter", "Cutoff frequency [Hz]:",
                float(cfg.get("cutoff_hz", 10.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            order, ok = QtWidgets.QInputDialog.getInt(
                self, "Low-pass Filter", "Filter order:",
                int(cfg.get("order", 4)), 1, 20, 1
            )
            if not ok:
                return

            cfg["cutoff_hz"] = float(cutoff)
            cfg["order"] = int(order)

            self._push_undo_state(f"low-pass filter ({cutoff:g} Hz)")
            new_names = []

            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                fs = 1.0 / dt
                nyq = 0.5 * fs
                if cutoff >= nyq:
                    raise RuntimeError(f"Cutoff frequency for '{ch_name}' must be below Nyquist ({nyq:.6g} Hz).")

                y = np.asarray(self.current_group()["channels"][ch_name]["y"], dtype=float)
                b, a = butter(order, cutoff / nyq, btype="low")
                y_filt = filtfilt(b, a, y)

                new_name = f"{ch_name}_LP_{cutoff:g}Hz"
                self._create_filtered_channel(ch_name, new_name, y_filt)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} low-pass channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_highpass(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["highpass"]
            cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "High-pass Filter", "Cutoff frequency [Hz]:",
                float(cfg.get("cutoff_hz", 1.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            order, ok = QtWidgets.QInputDialog.getInt(
                self, "High-pass Filter", "Filter order:",
                int(cfg.get("order", 4)), 1, 20, 1
            )
            if not ok:
                return

            cfg["cutoff_hz"] = float(cutoff)
            cfg["order"] = int(order)

            self._push_undo_state(f"high-pass filter ({cutoff:g} Hz)")
            new_names = []

            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                fs = 1.0 / dt
                nyq = 0.5 * fs
                if cutoff >= nyq:
                    raise RuntimeError(f"Cutoff frequency for '{ch_name}' must be below Nyquist ({nyq:.6g} Hz).")

                y = np.asarray(self.current_group()["channels"][ch_name]["y"], dtype=float)
                b, a = butter(order, cutoff / nyq, btype="high")
                y_filt = filtfilt(b, a, y)

                new_name = f"{ch_name}_HP_{cutoff:g}Hz"
                self._create_filtered_channel(ch_name, new_name, y_filt)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} high-pass channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_bandpass(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["bandpass"]
            low_cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "Band-pass Filter", "Low cutoff frequency [Hz]:",
                float(cfg.get("low_cutoff_hz", 1.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            high_cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "Band-pass Filter", "High cutoff frequency [Hz]:",
                float(cfg.get("high_cutoff_hz", 10.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            order, ok = QtWidgets.QInputDialog.getInt(
                self, "Band-pass Filter", "Filter order:",
                int(cfg.get("order", 4)), 1, 20, 1
            )
            if not ok:
                return

            low_cutoff = float(low_cutoff)
            high_cutoff = float(high_cutoff)
            if low_cutoff >= high_cutoff:
                raise RuntimeError("Low cutoff frequency must be below high cutoff frequency.")

            cfg["low_cutoff_hz"] = low_cutoff
            cfg["high_cutoff_hz"] = high_cutoff
            cfg["order"] = int(order)

            self._push_undo_state(f"band-pass filter ({low_cutoff:g} Hz - {high_cutoff:g} Hz)")
            new_names = []

            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                fs = 1.0 / dt
                nyq = 0.5 * fs
                if high_cutoff >= nyq:
                    raise RuntimeError(
                        f"High cutoff frequency for '{ch_name}' must be below Nyquist ({nyq:.6g} Hz)."
                    )

                y = np.asarray(self.current_group()["channels"][ch_name]["y"], dtype=float)
                b, a = butter(order, [low_cutoff / nyq, high_cutoff / nyq], btype="bandpass")
                y_filt = filtfilt(b, a, y)

                new_name = f"{ch_name}_BP_{low_cutoff:g}-{high_cutoff:g}Hz"
                self._create_filtered_channel(ch_name, new_name, y_filt)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} band-pass channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_bandpass_sos(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["bandpass_sos"]
            low_cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "Band-pass Filter (Stable SOS)", "Low cutoff frequency [Hz]:",
                float(cfg.get("low_cutoff_hz", 1.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            high_cutoff, ok = QtWidgets.QInputDialog.getDouble(
                self, "Band-pass Filter (Stable SOS)", "High cutoff frequency [Hz]:",
                float(cfg.get("high_cutoff_hz", 10.0)), 1e-9, 1e12, 6
            )
            if not ok:
                return

            order, ok = QtWidgets.QInputDialog.getInt(
                self, "Band-pass Filter (Stable SOS)", "Filter order:",
                int(cfg.get("order", 4)), 1, 20, 1
            )
            if not ok:
                return

            low_cutoff = float(low_cutoff)
            high_cutoff = float(high_cutoff)
            if low_cutoff >= high_cutoff:
                raise RuntimeError("Low cutoff frequency must be below high cutoff frequency.")

            cfg["low_cutoff_hz"] = low_cutoff
            cfg["high_cutoff_hz"] = high_cutoff
            cfg["order"] = int(order)

            self._push_undo_state(f"band-pass stable sos filter ({low_cutoff:g} Hz - {high_cutoff:g} Hz)")
            new_names = []

            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                fs = 1.0 / dt
                nyq = 0.5 * fs
                if high_cutoff >= nyq:
                    raise RuntimeError(
                        f"High cutoff frequency for '{ch_name}' must be below Nyquist ({nyq:.6g} Hz)."
                    )

                y = np.asarray(self.current_group()["channels"][ch_name]["y"], dtype=float)
                sos = butter(order, [low_cutoff / nyq, high_cutoff / nyq], btype="bandpass", output="sos")
                y_filt = sosfiltfilt(sos, y)

                new_name = f"{ch_name}_BPsos_{low_cutoff:g}-{high_cutoff:g}Hz"
                self._create_filtered_channel(ch_name, new_name, y_filt)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} stable SOS band-pass channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def _marker_window_dialog(self, title, description, cfg_key):
        cfg = self.filter_settings[cfg_key]

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(620, 280)

        layout = QtWidgets.QVBoxLayout(dialog)

        desc_label = QtWidgets.QLabel(description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        window_layout = QtWidgets.QHBoxLayout()
        window_label = QtWidgets.QLabel("Window length [s]:")
        window_input = QtWidgets.QDoubleSpinBox()
        window_input.setRange(1e-9, 1e12)
        window_input.setDecimals(6)
        window_input.setValue(float(cfg.get("window_sec", 1.0)))
        window_layout.addWidget(window_label)
        window_layout.addWidget(window_input)
        layout.addLayout(window_layout)

        marker_checkbox = QtWidgets.QCheckBox("Time window from marker")
        marker_checkbox.setChecked(bool(cfg.get("use_marker", False)))
        layout.addWidget(marker_checkbox)

        marker_layout = QtWidgets.QHBoxLayout()
        marker_label = QtWidgets.QLabel("Marker channel:")
        marker_channel_combo = QtWidgets.QComboBox()
        threshold_label = QtWidgets.QLabel("Threshold:")
        threshold_input = QtWidgets.QDoubleSpinBox()
        threshold_input.setDecimals(6)
        threshold_input.setRange(-1e12, 1e12)
        threshold_input.setValue(float(cfg.get("threshold", 0.0)))
        marker_layout.addWidget(marker_label)
        marker_layout.addWidget(marker_channel_combo, 1)
        marker_layout.addWidget(threshold_label)
        marker_layout.addWidget(threshold_input)
        layout.addLayout(marker_layout)

        inst_checkbox = QtWidgets.QCheckBox("Instantaneous window per revolution")
        inst_checkbox.setChecked(bool(cfg.get("use_instantaneous", False)))
        layout.addWidget(inst_checkbox)

        group = self.current_group()
        for ch_name in group["channels"].keys():
            marker_channel_combo.addItem(ch_name)

        saved_marker = str(cfg.get("marker_channel", "")).strip()
        if saved_marker:
            idx = marker_channel_combo.findText(saved_marker)
            if idx >= 0:
                marker_channel_combo.setCurrentIndex(idx)

        def toggle_marker_ui(checked):
            marker_label.setVisible(checked)
            marker_channel_combo.setVisible(checked)
            threshold_label.setVisible(checked)
            threshold_input.setVisible(checked)
            inst_checkbox.setVisible(checked)
            window_input.setEnabled(not checked)

        toggle_marker_ui(marker_checkbox.isChecked())
        marker_checkbox.toggled.connect(toggle_marker_ui)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None

        cfg["window_sec"] = float(window_input.value())
        cfg["use_marker"] = bool(marker_checkbox.isChecked())
        cfg["marker_channel"] = marker_channel_combo.currentText().strip()
        cfg["threshold"] = float(threshold_input.value())
        cfg["use_instantaneous"] = bool(inst_checkbox.isChecked())
        return copy.deepcopy(cfg)

    def _prepare_marker_window(self, group, cfg):
        use_marker = bool(cfg["use_marker"])
        use_instantaneous = bool(cfg["use_instantaneous"])
        marker_edges_idx = None
        marker_window_sec = None
        mt = None

        if use_marker:
            marker_name = cfg["marker_channel"]
            threshold = cfg["threshold"]
            if not marker_name:
                raise RuntimeError("Marker channel not selected.")
            if marker_name not in group["channels"]:
                raise RuntimeError(f"Marker channel '{marker_name}' not found in group.")

            marker_data = group["channels"][marker_name]
            mt = np.asarray(marker_data["x"], dtype=float)
            my = np.asarray(marker_data["y"], dtype=float)

            if len(mt) < 3:
                raise RuntimeError("Marker channel is too short.")

            rising = (my[:-1] < threshold) & (my[1:] >= threshold)
            marker_edges_idx = np.where(rising)[0] + 1
            if len(marker_edges_idx) < 2:
                raise RuntimeError("Not enough marker pulse rising edges detected.")

            marker_dt = np.diff(mt[marker_edges_idx])
            marker_window_sec = float(np.mean(marker_dt))
        else:
            marker_window_sec = float(cfg["window_sec"])

        return use_marker, use_instantaneous, marker_edges_idx, marker_window_sec, mt

    def filter_moving_window_pkpk(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self._marker_window_dialog(
                "Moving Window Peak-to-Peak",
                "Moving Window Pk-Pk calculates local peak-to-peak value.\n"
                "Manual mode: uses a fixed time window [s].\n"
                "Marker mode: window is derived from marker pulse rising edges.\n"
                "Instantaneous per revolution: each revolution uses its own pulse-to-pulse duration.",
                "moving_pkpk",
            )
            if cfg is None:
                return

            group = self.current_group()
            use_marker, use_instantaneous, marker_edges_idx, marker_window_sec, mt = self._prepare_marker_window(group, cfg)

            self._push_undo_state(
                f"moving window pk-pk ({marker_window_sec:g} s"
                + (", marker" if use_marker else "")
                + (", instantaneous" if use_instantaneous else "")
                + ")"
            )

            new_names = []
            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                ch = group["channels"][ch_name]
                x = np.asarray(ch["x"], dtype=float)
                y = np.asarray(ch["y"], dtype=float)

                if use_marker and use_instantaneous:
                    y_pkpk = np.full_like(y, np.nan, dtype=float)
                    marker_times = mt[marker_edges_idx]
                    ch_edge_idx = np.searchsorted(x, marker_times)
                    valid_edge_idx = ch_edge_idx[(ch_edge_idx >= 0) & (ch_edge_idx < len(x))]

                    if len(valid_edge_idx) < 2:
                        raise RuntimeError(f"Marker edges could not be mapped to channel '{ch_name}'.")

                    for i in range(len(valid_edge_idx) - 1):
                        i0 = int(valid_edge_idx[i])
                        i1 = int(valid_edge_idx[i + 1])
                        if i1 <= i0:
                            continue
                        seg = y[i0:i1]
                        if len(seg) == 0:
                            continue
                        y_pkpk[i0:i1] = float(np.max(seg) - np.min(seg))

                    finite_mask = np.isfinite(y_pkpk)
                    if np.any(finite_mask):
                        first_valid = int(np.argmax(finite_mask))
                        last_valid = len(y_pkpk) - 1 - int(np.argmax(finite_mask[::-1]))
                        y_pkpk[:first_valid] = y_pkpk[first_valid]
                        y_pkpk[last_valid + 1:] = y_pkpk[last_valid]
                    else:
                        raise RuntimeError(f"Could not compute instantaneous revolution pk-pk for '{ch_name}'.")

                    new_name = f"{ch_name}_PkPk_1rev_inst"
                else:
                    window_samples = max(1, int(round(marker_window_sec / dt)))
                    if window_samples % 2 == 0:
                        window_samples += 1
                    y_max = maximum_filter1d(y, size=window_samples, mode="nearest")
                    y_min = minimum_filter1d(y, size=window_samples, mode="nearest")
                    y_pkpk = y_max - y_min
                    new_name = f"{ch_name}_PkPk_1rev_avg" if use_marker else f"{ch_name}_PkPk_{marker_window_sec:.4g}s"

                self._create_filtered_channel(ch_name, new_name, y_pkpk)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()

            if use_marker and use_instantaneous:
                self.statusBar().showMessage(f"Created {len(new_names)} instantaneous per-revolution pk-pk channel(s)")
            elif use_marker:
                self.statusBar().showMessage(f"Created {len(new_names)} marker-based average-revolution pk-pk channel(s)")
            else:
                self.statusBar().showMessage(f"Created {len(new_names)} moving pk-pk channel(s), window={marker_window_sec:.4g}s")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_moving_window_rms(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self._marker_window_dialog(
                "Moving Window RMS",
                "Moving Window RMS calculates local RMS value.\n"
                "Manual mode: uses a fixed time window [s].\n"
                "Marker mode: window is derived from marker pulse rising edges.\n"
                "Instantaneous per revolution: each revolution uses its own pulse-to-pulse duration.",
                "moving_rms",
            )
            if cfg is None:
                return

            group = self.current_group()
            use_marker, use_instantaneous, marker_edges_idx, marker_window_sec, mt = self._prepare_marker_window(group, cfg)

            self._push_undo_state(
                f"moving window rms ({marker_window_sec:g} s"
                + (", marker" if use_marker else "")
                + (", instantaneous" if use_instantaneous else "")
                + ")"
            )

            new_names = []
            for ch_name in selected:
                dt = self._get_dt_for_channel(ch_name)
                ch = group["channels"][ch_name]
                x = np.asarray(ch["x"], dtype=float)
                y = np.asarray(ch["y"], dtype=float)

                if use_marker and use_instantaneous:
                    y_rms = np.full_like(y, np.nan, dtype=float)
                    marker_times = mt[marker_edges_idx]
                    ch_edge_idx = np.searchsorted(x, marker_times)
                    valid_edge_idx = ch_edge_idx[(ch_edge_idx >= 0) & (ch_edge_idx < len(x))]

                    if len(valid_edge_idx) < 2:
                        raise RuntimeError(f"Marker edges could not be mapped to channel '{ch_name}'.")

                    for i in range(len(valid_edge_idx) - 1):
                        i0 = int(valid_edge_idx[i])
                        i1 = int(valid_edge_idx[i + 1])
                        if i1 <= i0:
                            continue
                        seg = y[i0:i1]
                        if len(seg) == 0:
                            continue
                        y_rms[i0:i1] = float(np.sqrt(np.mean(seg ** 2)))

                    finite_mask = np.isfinite(y_rms)
                    if np.any(finite_mask):
                        first_valid = int(np.argmax(finite_mask))
                        last_valid = len(y_rms) - 1 - int(np.argmax(finite_mask[::-1]))
                        y_rms[:first_valid] = y_rms[first_valid]
                        y_rms[last_valid + 1:] = y_rms[last_valid]
                    else:
                        raise RuntimeError(f"Could not compute instantaneous revolution RMS for '{ch_name}'.")

                    new_name = f"{ch_name}_RMS_1rev_inst"
                else:
                    window_samples = max(1, int(round(marker_window_sec / dt)))
                    if window_samples % 2 == 0:
                        window_samples += 1
                    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
                    y_rms = np.sqrt(np.convolve(y ** 2, kernel, mode="same"))
                    new_name = f"{ch_name}_RMS_1rev_avg" if use_marker else f"{ch_name}_RMS_{marker_window_sec:.4g}s"

                self._create_filtered_channel(ch_name, new_name, y_rms)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()

            if use_marker and use_instantaneous:
                self.statusBar().showMessage(f"Created {len(new_names)} instantaneous per-revolution RMS channel(s)")
            elif use_marker:
                self.statusBar().showMessage(f"Created {len(new_names)} marker-based average-revolution RMS channel(s)")
            else:
                self.statusBar().showMessage(f"Created {len(new_names)} moving RMS channel(s), window={marker_window_sec:.4g}s")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def filter_subtract_mean(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            self._push_undo_state("subtract mean")
            new_names = []
            group = self.current_group()

            for ch_name in selected:
                y = np.asarray(group["channels"][ch_name]["y"], dtype=float)
                mean_val = float(np.mean(y))
                y_new = y - mean_val
                new_name = f"{ch_name}_mean0"
                self._create_filtered_channel(ch_name, new_name, y_new)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} mean-subtracted channel(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Filter Error", str(e))

    def show_channel_context_menu(self, pos):
        item = self.channel_list.itemAt(pos)
        if item is None:
            return

        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction("Rename channel")
        delete_action = menu.addAction("Delete channel")
        action = menu.exec_(self.channel_list.viewport().mapToGlobal(pos))

        if action == rename_action:
            self.rename_channel(item)
        elif action == delete_action:
            self.delete_channel(item)

    def rename_channel(self, item):
        try:
            display_name = item.text()
            if display_name not in self.channel_map:
                return

            old_name = self.channel_map[display_name]
            group_name = self.current_group_name()
            group = self.current_group()

            new_name, ok = QtWidgets.QInputDialog.getText(
                self, "Rename Channel", f"New name for '{old_name}':", text=old_name
            )
            if not ok:
                return

            new_name = new_name.strip()
            if not new_name or new_name == old_name:
                return

            if new_name in group["channels"]:
                QtWidgets.QMessageBox.warning(self, "Rename Error", "Channel name already exists.")
                return

            self._push_undo_state(f"rename channel {old_name} -> {new_name}")
            group["channels"][new_name] = group["channels"].pop(old_name)

            selected = self.group_selection_state.get(group_name, [])
            self.group_selection_state[group_name] = [new_name if n == old_name else n for n in selected]

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.plot_channels()
            self.update_info_panel()
            self.statusBar().showMessage(f"Renamed channel: {old_name} -> {new_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename Channel Error", f"Could not rename channel.\n\n{e}")

    def delete_channel(self, item):
        try:
            display_name = item.text()
            if display_name not in self.channel_map:
                return

            base_name = self.channel_map[display_name]
            group_name = self.current_group_name()

            reply = QtWidgets.QMessageBox.question(
                self,
                "Delete Channel",
                f"Delete channel '{display_name}' from group '{group_name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

            self._push_undo_state(f"delete channel {display_name}")

            group = self.current_group()
            if base_name in group["channels"]:
                del group["channels"][base_name]

            selected = self.group_selection_state.get(group_name, [])
            self.group_selection_state[group_name] = [n for n in selected if n != base_name]

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.plot_channels()
            self.update_info_panel()
            self.statusBar().showMessage(f"Deleted channel: {display_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete Channel Error", f"Could not delete channel.\n\n{e}")

    def _group_name_at_pos(self, pos):
        view = self.group_combo.view()
        if view is None:
            return None

        global_pos = self.group_combo.mapToGlobal(pos)
        view_pos = view.viewport().mapFromGlobal(global_pos)
        index = view.indexAt(view_pos)
        if index.isValid():
            return index.data()

        return self.current_group_name().strip() or None

    def show_group_context_menu(self, pos):
        group_name = self._group_name_at_pos(pos)
        if not group_name:
            return

        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction(f"Rename group '{group_name}'")
        delete_action = menu.addAction(f"Delete group '{group_name}'")

        action = menu.exec_(self.group_combo.mapToGlobal(pos))
        if action == rename_action:
            self.rename_group(group_name)
        elif action == delete_action:
            self.delete_group(group_name)

    def show_group_view_context_menu(self, pos):
        view = self.group_combo.view()
        index = view.indexAt(pos)
        if not index.isValid():
            return

        group_name = index.data()
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction(f"Rename group '{group_name}'")
        delete_action = menu.addAction(f"Delete group '{group_name}'")

        action = menu.exec_(view.viewport().mapToGlobal(pos))
        if action == rename_action:
            self.rename_group(group_name)
        elif action == delete_action:
            self.delete_group(group_name)

    def rename_group(self, old_group_name):
        try:
            old_group_name = str(old_group_name).strip()
            if not old_group_name or old_group_name not in self.dataset.groups:
                return

            new_group_name, ok = QtWidgets.QInputDialog.getText(
                self,
                "Rename Group",
                f"New name for '{old_group_name}':",
                text=old_group_name,
            )
            if not ok:
                return

            new_group_name = new_group_name.strip()
            if not new_group_name or new_group_name == old_group_name:
                return

            if new_group_name in self.dataset.groups:
                QtWidgets.QMessageBox.warning(self, "Rename Error", "Group name already exists.")
                return

            self._push_undo_state(f"rename group {old_group_name} -> {new_group_name}")

            items = list(self.dataset.groups.items())
            new_groups = {}
            for key, value in items:
                if key == old_group_name:
                    new_groups[new_group_name] = value
                else:
                    new_groups[key] = value
            self.dataset.groups = new_groups

            if old_group_name in self.group_selection_state:
                self.group_selection_state[new_group_name] = self.group_selection_state.pop(old_group_name)

            if self._last_group_name == old_group_name:
                self._last_group_name = new_group_name

            self.populate_groups()
            idx = self.group_combo.findText(new_group_name)
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels()
            self.statusBar().showMessage(f"Renamed group: {old_group_name} -> {new_group_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Rename Group Error", f"Could not rename group.\n\n{e}")

    def delete_group(self, group_name):
        try:
            group_name = str(group_name).strip()
            if not group_name or group_name not in self.dataset.groups:
                return

            reply = QtWidgets.QMessageBox.question(
                self,
                "Delete Group",
                f"Delete group '{group_name}' and all its channels?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

            self._push_undo_state(f"delete group {group_name}")
            current_group = self.current_group_name().strip()

            del self.dataset.groups[group_name]
            self.group_selection_state.pop(group_name, None)

            if not self.dataset.has_groups():
                self.group_combo.blockSignals(True)
                self.group_combo.clear()
                self.group_combo.blockSignals(False)

                self.channel_list.blockSignals(True)
                self.channel_list.clear()
                self.channel_list.blockSignals(False)
                self.channel_map.clear()
                self.plotted_data.clear()

                self.plot.clear()
                self.legend = self.plot.addLegend()
                self.legend.anchor((1, 0), (1, 0))
                self.legend.setOffset((-10, 10))
                self.plot.addItem(self.region)

                self.band_table.setRowCount(0)
                self.band_label.setText("X1: -, X2: -, dX: -")
                self.info_box.clear()
                self.file_label.setText("No file loaded")
                self._last_group_name = ""
                self.statusBar().showMessage(f"Deleted group: {group_name}")
                return

            remaining_groups = self.dataset.get_group_names()
            self.populate_groups()

            if current_group == group_name:
                new_group = remaining_groups[0]
                idx = self.group_combo.findText(new_group)
                if idx >= 0:
                    self.group_combo.setCurrentIndex(idx)
            else:
                idx = self.group_combo.findText(current_group)
                if idx >= 0:
                    self.group_combo.setCurrentIndex(idx)

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels()
            self._last_group_name = self.current_group_name().strip()
            self.statusBar().showMessage(f"Deleted group: {group_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete Group Error", f"Could not delete group.\n\n{e}")

    # ----------------------------
    # Project save / open
    # ----------------------------
    def _serialize_widget(self, widget):
        name = widget.objectName().strip() if widget.objectName() else ""
        if not name:
            return None

        data = {"class": widget.__class__.__name__}
        if isinstance(widget, QtWidgets.QCheckBox):
            data["checked"] = widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox):
            data["current_text"] = widget.currentText()
            data["current_index"] = widget.currentIndex()
        elif isinstance(widget, QtWidgets.QSplitter):
            data["sizes"] = widget.sizes()
        elif isinstance(widget, QtWidgets.QTableWidget):
            data["column_widths"] = [widget.columnWidth(i) for i in range(widget.columnCount())]
        else:
            return None

        return name, data

    def _apply_widget_state(self, widget, data):
        try:
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(data.get("checked", False)))
            elif isinstance(widget, QtWidgets.QComboBox):
                text = str(data.get("current_text", ""))
                idx = widget.findText(text)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                elif widget.count() > 0:
                    saved_idx = int(data.get("current_index", 0))
                    if 0 <= saved_idx < widget.count():
                        widget.setCurrentIndex(saved_idx)
            elif isinstance(widget, QtWidgets.QSplitter):
                sizes = data.get("sizes", [])
                if sizes:
                    widget.setSizes([int(x) for x in sizes])
            elif isinstance(widget, QtWidgets.QTableWidget):
                widths = data.get("column_widths", [])
                for i, w in enumerate(widths):
                    if i < widget.columnCount():
                        widget.setColumnWidth(i, int(w))
        except Exception:
            pass

    def _collect_project_state(self, tdms_filename):
        state = {
            "project_version": APP_VERSION,
            "tdms_file": tdms_filename,
            "current_group": self.current_group_name(),
            "last_group_name": self._last_group_name,
            "group_selection_state": self.group_selection_state,
            "filter_settings": self.filter_settings,
            "band": {
                "enabled": self.band_checkbox.isChecked(),
                "region": list(self.region.getRegion()),
            },
            "plots": {},
            "widgets": {},
            "xy_pairs": [],
        }

        try:
            main_range = self.plot.getViewBox().viewRange()
            state["plots"]["main"] = {
                "x_min": float(main_range[0][0]),
                "x_max": float(main_range[0][1]),
                "y_min": float(main_range[1][0]),
                "y_max": float(main_range[1][1]),
            }
        except Exception:
            state["plots"]["main"] = None

        try:
            band_range = self.band_plot.getViewBox().viewRange()
            state["plots"]["band"] = {
                "x_min": float(band_range[0][0]),
                "x_max": float(band_range[0][1]),
                "y_min": float(band_range[1][0]),
                "y_max": float(band_range[1][1]),
            }
        except Exception:
            state["plots"]["band"] = None

        for pair in self.xy_pairs:
            state["xy_pairs"].append({
                "enabled": pair["enable"].isChecked(),
                "x": pair["x_combo"].currentText(),
                "y": pair["y_combo"].currentText(),
            })

        for widget in self.findChildren(QtWidgets.QWidget):
            result = self._serialize_widget(widget)
            if result is not None:
                name, data = result
                state["widgets"][name] = data

        return state

    def _apply_project_state(self, state):
        self.group_selection_state = copy.deepcopy(state.get("group_selection_state", {}))
        self.filter_settings = copy.deepcopy(state.get("filter_settings", self.filter_settings))

        current_group = str(state.get("current_group", "")).strip()
        if current_group:
            idx = self.group_combo.findText(current_group)
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)

        widgets_state = state.get("widgets", {})
        for widget in self.findChildren(QtWidgets.QWidget):
            name = widget.objectName().strip() if widget.objectName() else ""
            if not name or name not in widgets_state:
                continue
            self._apply_widget_state(widget, widgets_state[name])

        self.populate_channels()
        self.refresh_group_channel_highlight()

        xy_pairs_state = state.get("xy_pairs", [])
        for i, pair_state in enumerate(xy_pairs_state):
            if i >= len(self.xy_pairs):
                break

            pair = self.xy_pairs[i]
            pair["enable"].setChecked(bool(pair_state.get("enabled", False)))

            x_text = str(pair_state.get("x", ""))
            y_text = str(pair_state.get("y", ""))

            x_idx = pair["x_combo"].findText(x_text)
            if x_idx >= 0:
                pair["x_combo"].setCurrentIndex(x_idx)

            y_idx = pair["y_combo"].findText(y_text)
            if y_idx >= 0:
                pair["y_combo"].setCurrentIndex(y_idx)

        band_state = state.get("band", {})
        self.band_checkbox.setChecked(bool(band_state.get("enabled", True)))

        region = band_state.get("region", None)
        if isinstance(region, (list, tuple)) and len(region) == 2:
            try:
                self.region.setRegion((float(region[0]), float(region[1])))
            except Exception:
                pass

        self.plot_channels()
        self.update_info_panel()

        main_plot = state.get("plots", {}).get("main", None)
        if main_plot:
            try:
                self.plot.setXRange(float(main_plot["x_min"]), float(main_plot["x_max"]), padding=0)
                self.plot.setYRange(float(main_plot["y_min"]), float(main_plot["y_max"]), padding=0)
            except Exception:
                pass

        band_plot = state.get("plots", {}).get("band", None)
        if band_plot:
            try:
                self.band_plot.setXRange(float(band_plot["x_min"]), float(band_plot["x_max"]), padding=0)
                self.band_plot.setYRange(float(band_plot["y_min"]), float(band_plot["y_max"]), padding=0)
            except Exception:
                pass

        self._last_group_name = self.current_group_name()
        self.update_info_panel()

    def save_project(self):
        try:
            if not self.dataset.has_groups():
                QtWidgets.QMessageBox.information(self, "Save Project", "No dataset loaded.")
                return

            prj_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Project", "", "Project files (*.prj)"
            )
            if not prj_path:
                return

            prj_path = str(Path(prj_path).with_suffix(".prj"))
            tdms_path = str(Path(prj_path).with_suffix(".tdms"))

            dataset_to_tdms(self.dataset, tdms_path)

            state = self._collect_project_state(os.path.basename(tdms_path))
            with open(prj_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            self.statusBar().showMessage(f"Project saved: {prj_path}")
            QtWidgets.QMessageBox.information(
                self,
                "Save Project",
                f"Project saved successfully.\n\nPRJ: {prj_path}\nTDMS: {tdms_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Project Error", str(e))

    def open_project(self):
        try:
            prj_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Project", "", "Project files (*.prj)"
            )
            if not prj_path:
                return

            with open(prj_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            tdms_file = str(state.get("tdms_file", "")).strip()
            if not tdms_file:
                raise RuntimeError("Project file does not contain TDMS reference.")

            tdms_path = str(Path(prj_path).with_name(tdms_file))
            if not os.path.exists(tdms_path):
                raise RuntimeError(f"Referenced TDMS file not found:\n{tdms_path}")

            dataset = dataset_from_tdms(TdmsFile.read(tdms_path))
            self._load_dataset(dataset, tdms_path)
            self._apply_project_state(state)
            self.update_info_panel()

            self.statusBar().showMessage(f"Project opened: {prj_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Project Error", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    win = TdmsPlotter()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
