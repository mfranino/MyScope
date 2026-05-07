import copy
import csv
import ctypes
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from ctypes import wintypes

import numpy as np
import pyqtgraph as pg
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter
from qtpy import QtCore, QtGui, QtWidgets
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, filtfilt, sosfiltfilt

APP_NAME = "MyScope"
APP_VERSION = "0.3.9"

MAX_PLOT_POINTS_MAIN = 20000
MAX_PLOT_POINTS_BAND = 8000



class NonUniformSamplingError(RuntimeError):
    def __init__(self, message, *, dt_expected=None, missing_points=0):
        super().__init__(message)
        self.dt_expected = dt_expected
        self.missing_points = int(missing_points)

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
        self.setMouseMode(pg.ViewBox.PanMode)

    def set_mode(self, mode):
        self.mode = mode
        if mode == self.MODE_RECT:
            self.setMouseMode(pg.ViewBox.PanMode)
        elif mode in (self.MODE_X, self.MODE_Y):
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
    group_props = dict(group_props or {})
    source_file = dataset.root_props.get("SourceFile")
    if source_file is not None and "SourceFile" not in group_props:
        group_props["SourceFile"] = source_file
    dataset.add_group(group_name, group_props=group_props, channels=channels)
    return dataset


def _build_uniform_timebase(t, data, *, dt_hint=None, rounding_resolution=None):
    t = np.asarray(t, dtype=float)
    data = np.asarray(data, dtype=float)
    if t.ndim != 1:
        raise RuntimeError("Time vector must be one-dimensional.")
    if len(t) < 2:
        raise RuntimeError("Not enough samples")

    diffs = np.diff(t)
    if dt_hint is not None:
        dt = float(dt_hint)
        if dt <= 0:
            raise RuntimeError("Invalid time step.")

        residual_tol = max(1e-9, abs(dt) * 0.05)
        if rounding_resolution is not None and rounding_resolution > 0:
            residual_tol = max(residual_tol, float(rounding_resolution) * 0.5 + 1e-12)

        # Some SRM exports round the time column more coarsely than the
        # sampling interval in the header, so parsed timestamps can repeat.
        # In that case, accept sample order and rebuild the uniform timebase
        # from the header-provided sampling rate.
        sample_idx = np.arange(len(t), dtype=int)
        aligned_t = t[0] + sample_idx.astype(float) * dt
        max_residual = float(np.max(np.abs(t - aligned_t)))
        if np.all(diffs >= 0) and max_residual <= residual_tol:
            return aligned_t, data.copy(), dt, 0

        sample_idx = np.rint((t - t[0]) / dt).astype(int)
        idx_diffs = np.diff(sample_idx)
        if np.any(idx_diffs <= 0):
            raise NonUniformSamplingError(
                "Sampling rate from the SRM header does not match the time column.",
                dt_expected=dt,
                missing_points=0,
            )

        aligned_t = t[0] + sample_idx.astype(float) * dt
        max_residual = float(np.max(np.abs(t - aligned_t)))
        if max_residual > residual_tol:
            raise NonUniformSamplingError(
                "Sampling rate from the SRM header does not match the time column.",
                dt_expected=dt,
                missing_points=0,
            )

        missing_points = int(np.sum(idx_diffs - 1))
        t_uniform = t[0] + np.arange(sample_idx[-1] + 1, dtype=float) * dt
    else:
        if np.any(diffs <= 0):
            raise RuntimeError("Time values must be strictly increasing.")
        dt = float(np.median(diffs))
        if dt <= 0:
            raise RuntimeError("Invalid time step.")
        tol = max(1e-9, abs(dt) * 1e-6)
        multiples = np.rint(diffs / dt).astype(int)
        multiples = np.maximum(multiples, 1)
        reconstructed = multiples * dt
        max_dev = float(np.max(np.abs(diffs - reconstructed)))
        if max_dev > max(tol, abs(dt) * 0.25):
            raise NonUniformSamplingError(
                "Time is not uniformly sampled; cannot prepare waveform export safely.",
                dt_expected=dt,
                missing_points=0,
            )

        missing_points = int(np.sum(multiples - 1))
        if missing_points == 0:
            return t.copy(), data.copy(), dt, 0
        t_uniform = t[0] + np.arange(int(np.sum(multiples)) + 1, dtype=float) * dt

    repaired_cols = []
    for col_idx in range(data.shape[1]):
        y = np.asarray(data[:, col_idx], dtype=float)
        valid = ~np.isnan(y)
        if not np.any(valid):
            repaired_cols.append(np.full_like(t_uniform, np.nan, dtype=float))
            continue
        if np.count_nonzero(valid) == 1:
            repaired_cols.append(np.full_like(t_uniform, y[valid][0], dtype=float))
            continue
        repaired_cols.append(np.interp(t_uniform, t[valid], y[valid]))

    repaired = np.column_stack(repaired_cols)
    return t_uniform, repaired, dt, missing_points


def _decimal_resolution(token):
    token = str(token).strip()
    if not token:
        return None
    token = token.replace(",", ".")
    if "." not in token:
        return 1.0
    decimals = len(token.rsplit(".", 1)[1])
    return 10.0 ** (-decimals)



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

    group_name = os.path.splitext(os.path.basename(str(file_name_str or "").strip()))[0]
    group_props = {
        "Author": measured_by,
        "Project": object_name,
        "Measurement": measurement,
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


def dataset_from_srm(file_path, repair_missing=False):
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
        "Measurement": measurement,
        "Date": date_str,
        "Time": time_str,
        "SourceFormat": "SRM",
    }
    group_props = {k: v for k, v in group_props.items() if v is not None}
    root_props = {"SourceFile": file_path}

    sampling_rate = None
    if scan_per_s is not None:
        try:
            sampling_rate = float(scan_per_s.replace(",", "."))
            group_props["SamplingRate"] = sampling_rate
        except ValueError:
            sampling_rate = None

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
    time_resolutions = []
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

        time_resolution = _decimal_resolution(parts[0]) if parts else None
        if time_resolution is not None:
            time_resolutions.append(time_resolution)

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

    t_all = data[:, 0]
    valid_t = ~np.isnan(t_all)
    data_valid = data[valid_t]
    t = data_valid[:, 0]

    if len(t) < 2:
        raise RuntimeError("Not enough samples")

    rounding_resolution = min(time_resolutions) if time_resolutions else None
    dt_hint = (1.0 / sampling_rate) if sampling_rate not in (None, 0) else None
    t_uniform, data_uniform, dt_used, missing_points = _build_uniform_timebase(
        t,
        data_valid,
        dt_hint=dt_hint,
        rounding_resolution=rounding_resolution,
    )


    if missing_points > 0:
        if not repair_missing:
            raise NonUniformSamplingError(
                f"Detected {missing_points} missing sample(s) in SRM time data.",
                dt_expected=dt_used,
                missing_points=missing_points,
            )
        group_props["InterpolatedMissingSamples"] = missing_points
        group_props["InterpolationMethod"] = "Linear"

    channel_names = ["X1", "Y1", "X2", "Y2", "X3", "Y3", "Pgen", "KP"]
    channels = {}
    for col_idx in range(1, min(data_uniform.shape[1], len(channel_names) + 1)):
        y = data_uniform[:, col_idx]
        if len(y) == 0 or np.all(np.isnan(y)):
            continue
        channels[channel_names[col_idx - 1]] = {
            "x": t_uniform.copy(),
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


def _group_start_datetime(group_props):
    group_props = dict(group_props or {})

    date_str = str(group_props.get("Date", "")).strip()
    time_str = str(group_props.get("Time", "")).strip()
    if date_str and time_str:
        combined = f"{date_str} {time_str}"
        for fmt in (
            "%d. %m. %Y %H:%M:%S",
            "%d. %m. %Y %H:%M",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                pass

    date_time_str = str(group_props.get("DateTime", "")).strip()
    if date_time_str:
        for fmt in (
            "%d. %m. %Y %H:%M:%S",
            "%d. %m. %Y %H:%M",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
        ):
            try:
                return datetime.strptime(date_time_str, fmt)
            except ValueError:
                pass

    return None


def dataset_to_tdms(dataset, out_tdms):
    if not dataset.has_groups():
        raise RuntimeError("Dataset contains no groups.")

    objects = [RootObject(properties=dataset.root_props)]

    for group_name, group_data in dataset.groups.items():
        group_props = dict(group_data["props"])
        channels = group_data["channels"]
        group_start_time = _group_start_datetime(group_props)

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
            if group_start_time is not None:
                props["wf_start_time"] = group_start_time

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
            "scale_linear": {"gain": 1.0, "offset": 0.0},
            "subtract_mean": {},
        }

        self._build_ui()
        self._build_actions()
        self._build_menus()
        self._connect_signals()
        self.update_bottom_y_controls_visibility()
        self._memory_timer = QtCore.QTimer(self)
        self._memory_timer.setInterval(2000)
        self._memory_timer.timeout.connect(self.update_memory_usage_label)
        self._memory_timer.start()
        self.update_memory_usage_label()

    def _get_memory_usage_mb(self):
        try:
            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)

            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            process = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            return None

    def update_memory_usage_label(self):
        usage_mb = self._get_memory_usage_mb()
        if usage_mb is None:
            self.memory_usage_label.setText("Memory: n/a")
        else:
            self.memory_usage_label.setText(f"Memory: {usage_mb:.1f} MB")

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
        self.notes_box.clear()
        self.band_table.setRowCount(0)
        self.band_label.setText("X1: -, X2: -, dX: -")

        self.band_checkbox.setChecked(True)
        self.xy_mode_checkbox.setChecked(False)
        self.xy_pair_count_spin.setValue(2)
        self._set_xy_pair_count(self.xy_pair_count_spin.value())
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

        self.channel_list = QtWidgets.QTreeWidget()
        self.channel_list.setHeaderHidden(True)
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.channel_list.setObjectName("channel_list")

        self.clear_channel_selection_button = QtWidgets.QPushButton("Clear Ch selection")

        channels_header = QtWidgets.QHBoxLayout()
        channels_header.setContentsMargins(0, 0, 0, 0)
        channels_header.addWidget(QtWidgets.QLabel("Channels"))
        channels_header.addStretch(1)
        channels_header.addWidget(self.clear_channel_selection_button)

        left.addWidget(self.open_tdms_button)
        left.addWidget(self.open_slab_button)
        left.addWidget(self.open_srm_button)
        left.addWidget(self.open_vib_button)
        left.addWidget(self.file_label)
        left.addLayout(channels_header)
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

        self.xy_pair_count_spin = QtWidgets.QSpinBox()
        self.xy_pair_count_spin.setObjectName("xy_pair_count_spin")
        self.xy_pair_count_spin.setMinimum(1)
        self.xy_pair_count_spin.setMaximum(32)
        self.xy_pair_count_spin.setValue(2)
        self.xy_pair_count_spin.setToolTip("Number of XY channel pairs")

        self.bottom_autoscale_y_once_button = QtWidgets.QPushButton("Auto-scale Y once")

        self.bottom_enable_autoscale_y_checkbox = QtWidgets.QCheckBox("Enable autoscale Y")
        self.bottom_enable_autoscale_y_checkbox.setChecked(False)
        self.bottom_enable_autoscale_y_checkbox.setObjectName("bottom_enable_autoscale_y_checkbox")

        self.xy_pairs = []
        band_bottom_widget = QtWidgets.QWidget()
        band_bottom_layout = QtWidgets.QHBoxLayout(band_bottom_widget)
        band_bottom_layout.setContentsMargins(0, 0, 0, 0)

        xy_controls_widget = QtWidgets.QWidget()
        xy_controls_layout = QtWidgets.QVBoxLayout(xy_controls_widget)
        xy_controls_layout.setContentsMargins(4, 4, 4, 4)

        top_controls_row = QtWidgets.QHBoxLayout()
        top_controls_row.addWidget(self.xy_mode_checkbox)
        top_controls_row.addWidget(self.xy_pair_count_spin)
        top_controls_row.addStretch()
        top_controls_row.addWidget(self.bottom_autoscale_y_once_button)
        top_controls_row.addWidget(self.bottom_enable_autoscale_y_checkbox)
        xy_controls_layout.addLayout(top_controls_row)

        self.xy_pairs_layout = QtWidgets.QVBoxLayout()
        self.xy_pairs_layout.setContentsMargins(0, 0, 0, 0)
        xy_controls_layout.addLayout(self.xy_pairs_layout)
        self._set_xy_pair_count(self.xy_pair_count_spin.value())

        x_axis_range_row = QtWidgets.QHBoxLayout()
        y_axis_range_row = QtWidgets.QHBoxLayout()
        self.band_x_min_spin = QtWidgets.QDoubleSpinBox()
        self.band_x_max_spin = QtWidgets.QDoubleSpinBox()
        self.band_y_min_spin = QtWidgets.QDoubleSpinBox()
        self.band_y_max_spin = QtWidgets.QDoubleSpinBox()
        self.band_axis_apply_button = QtWidgets.QPushButton("Apply")
        self.band_axis_apply_button.setObjectName("band_axis_apply_button")

        for widget, name in (
            (self.band_x_min_spin, "band_x_min_spin"),
            (self.band_x_max_spin, "band_x_max_spin"),
            (self.band_y_min_spin, "band_y_min_spin"),
            (self.band_y_max_spin, "band_y_max_spin"),
        ):
            widget.setObjectName(name)
            widget.setDecimals(1)
            widget.setRange(-1e12, 1e12)
            widget.setMinimumWidth(42)

        x_axis_range_row.addWidget(QtWidgets.QLabel("X axis range:"))
        x_axis_range_row.addWidget(self.band_x_min_spin)
        x_axis_range_row.addWidget(self.band_x_max_spin)
        x_axis_range_row.addSpacing(self.band_axis_apply_button.sizeHint().width())
        x_axis_range_row.addStretch()
        xy_controls_layout.addLayout(x_axis_range_row)

        y_axis_range_row.addWidget(QtWidgets.QLabel("Y axis range:"))
        y_axis_range_row.addWidget(self.band_y_min_spin)
        y_axis_range_row.addWidget(self.band_y_max_spin)
        y_axis_range_row.addWidget(self.band_axis_apply_button)
        y_axis_range_row.addStretch()
        xy_controls_layout.addLayout(y_axis_range_row)

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

        self.band_table = QtWidgets.QTableWidget(0, 13)
        self.band_table.setObjectName("band_table")
        self.band_table.setHorizontalHeaderLabels([
            "Unit", "Y@X1", "Y@X2", "Delta Y", "Mean",
            "Min", "Max", "PkPk", "PkPk98", "PkPk95", "StdDev", "RMS", "AC RMS",
        ])
        self.band_table.verticalHeader().setVisible(True)
        self.band_table_copy_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Copy, self.band_table)
        self.band_table_copy_shortcut.activated.connect(self.copy_band_table_selection)

        info_group = QtWidgets.QGroupBox("Info")
        info_layout = QtWidgets.QVBoxLayout()
        self.info_box = QtWidgets.QPlainTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(190)
        self.info_box.setObjectName("info_box")
        info_layout.addWidget(self.info_box)
        info_group.setLayout(info_layout)

        notes_group = QtWidgets.QGroupBox("Notes")
        notes_layout = QtWidgets.QVBoxLayout()
        self.notes_box = QtWidgets.QPlainTextEdit()
        self.notes_box.setPlaceholderText("Project notes")
        self.notes_box.setMinimumHeight(120)
        self.notes_box.setObjectName("notes_box")
        notes_layout.addWidget(self.notes_box)
        notes_group.setLayout(notes_layout)

        memory_row = QtWidgets.QHBoxLayout()
        memory_row.addStretch()
        self.memory_usage_label = QtWidgets.QLabel("Memory: --")
        self.memory_usage_label.setObjectName("memory_usage_label")
        memory_row.addWidget(self.memory_usage_label)

        right.addWidget(self.band_checkbox)
        right.addWidget(self.band_label)
        right.addWidget(self.band_table)
        right.addWidget(info_group)
        right.addWidget(notes_group)
        right.addLayout(memory_row)

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
        self.action_save_project.setShortcut(QtGui.QKeySequence.Save)
        self.action_export_dataset = QtWidgets.QAction("Export Dataset TDMS", self)
        self.action_export_statistics = QtWidgets.QAction("Export Statistics Table...", self)

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
        self.action_scale_linear = QtWidgets.QAction("Scale...", self)
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
        file_menu.addAction(self.action_export_statistics)

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
        filter_menu.addAction(self.action_scale_linear)
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
        self.action_export_statistics.triggered.connect(self.export_statistics_table)
        self.action_undo.triggered.connect(self.undo_last_action)

        self.action_filter_moving_average.triggered.connect(self.filter_moving_average)
        self.action_filter_lowpass.triggered.connect(self.filter_lowpass)
        self.action_filter_highpass.triggered.connect(self.filter_highpass)
        self.action_filter_bandpass.triggered.connect(self.filter_bandpass)
        self.action_filter_bandpass_sos.triggered.connect(self.filter_bandpass_sos)
        self.action_moving_pkpk.triggered.connect(self.filter_moving_window_pkpk)
        self.action_moving_rms.triggered.connect(self.filter_moving_window_rms)
        self.action_scale_linear.triggered.connect(self.filter_scale_linear)
        self.action_subtract_mean.triggered.connect(self.filter_subtract_mean)

        self.group_combo.currentIndexChanged.connect(self._handle_group_change)
        self.group_combo.customContextMenuRequested.connect(self.show_group_context_menu)
        self.group_combo.view().customContextMenuRequested.connect(self.show_group_view_context_menu)

        self.channel_list.itemSelectionChanged.connect(self._on_channel_selection_changed)
        self.channel_list.itemClicked.connect(self.on_channel_tree_item_clicked)
        self.clear_channel_selection_button.clicked.connect(self.clear_channel_selection_all_groups)
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
        self.xy_pair_count_spin.valueChanged.connect(self._on_xy_pair_count_changed)
        self.bottom_autoscale_y_once_button.clicked.connect(self.auto_scale_bottom_y_once)
        self.bottom_enable_autoscale_y_checkbox.toggled.connect(self.on_bottom_enable_autoscale_y_toggled)
        self.band_axis_apply_button.clicked.connect(self.apply_band_axis_ranges)

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

    def apply_band_axis_ranges(self):
        try:
            if not self.band_checkbox.isChecked():
                return

            x_min = float(self.band_x_min_spin.value())
            x_max = float(self.band_x_max_spin.value())
            y_min = float(self.band_y_min_spin.value())
            y_max = float(self.band_y_max_spin.value())

            if x_max <= x_min:
                raise RuntimeError("X axis max must be greater than X axis min.")
            if y_max <= y_min:
                raise RuntimeError("Y axis max must be greater than Y axis min.")

            self.band_plot.enableAutoRange(axis="x", enable=False)
            self.band_plot.enableAutoRange(axis="y", enable=False)
            self.band_plot.setXRange(x_min, x_max, padding=0)
            self.band_plot.setYRange(y_min, y_max, padding=0)
            self.statusBar().showMessage("Band plot axis ranges applied")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Band Axis Range Error", str(e))

    def toggle_band(self, checked):
        try:
            enabled = bool(checked)
            self.region.setVisible(enabled)
            self.band_plot.setVisible(enabled)
            self.xy_mode_checkbox.setEnabled(enabled)
            self.xy_pair_count_spin.setEnabled(enabled)
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
        source_file = group["props"].get("SourceFile")
        root_props = dict(self.dataset.root_props)
        if source_file is not None:
            root_props["SourceFile"] = source_file

        lines.append("Root properties:")
        if root_props:
            for k, v in root_props.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  -")

        lines.append(f"Measurement: {group['props'].get('Measurement', '')}")


        if group["props"]:
            for k, v in group["props"].items():
                if k in ("SourceFile", "SamplingRate", "Measurement"):
                    continue
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
        self.group_selection_state = {
            g: list(self.dataset.get_group(g)["channels"].keys())
            for g in self.dataset.get_group_names()
        }
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
            self.group_selection_state.setdefault(group_name, list(self.dataset.get_group(group_name)["channels"].keys()))

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

    def _add_loaded_dataset(self, dataset, path):
        if self.dataset.has_groups():
            self._append_dataset(dataset, path)
        else:
            self._load_dataset(dataset, path)

    def _import_tdms_file(self, path):
        dataset = dataset_from_tdms(TdmsFile.read(path))
        self._add_loaded_dataset(dataset, path)


    def _open_files_with_progress(self, paths, label, import_func):
        progress = QtWidgets.QProgressDialog(f"Opening {label} files...", "Cancel", 0, len(paths), self)
        progress.setWindowTitle(f"Open {label}")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        for idx, path in enumerate(paths, start=1):
            progress.setLabelText(f"Opening {label} file {idx} of {len(paths)}\n{path}")
            progress.setValue(idx - 1)
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                self.statusBar().showMessage(f"Open {label} canceled")
                return

            result = import_func(path)
            if result is False:
                progress.setValue(idx)
                return

            progress.setValue(idx)
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                self.statusBar().showMessage(f"Open {label} canceled")
                return
    def open_tdms(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open TDMS", "", "TDMS files (*.tdms)")
        if not paths:
            return
        try:
            self._open_files_with_progress(paths, "TDMS", self._import_tdms_file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Error", str(e))
    def _import_slab_file(self, path):
        dataset = dataset_from_slab(path)
        self._add_loaded_dataset(dataset, path)

    def open_slab(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open sLAB File", "", "sLAB files (*.xls *.txt *.dat);;All files (*.*)"
        )
        if not paths:
            return
        try:
            self._open_files_with_progress(paths, "sLAB", self._import_slab_file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "sLAB Import Error", str(e))

    def _import_srm_file(self, path):
        try:
            dataset = dataset_from_srm(path)
        except NonUniformSamplingError as e:
            if e.missing_points <= 0:
                QtWidgets.QMessageBox.critical(self, "SRM Import Error", str(e))
                return False
            dt_text = f"{e.dt_expected:.9g} s" if e.dt_expected else "the inferred time step"
            reply = QtWidgets.QMessageBox.question(
                self,
                "Repair SRM Sampling",
                (
                    f"Detected {e.missing_points} missing sample(s) in the SRM time column.\n\n"
                    f"MyScope can insert the missing time points using a uniform step of {dt_text} "
                    "and linearly interpolate the measurement values.\n\n"
                    "Do you want to continue with this repair?"
                ),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return False
            try:
                dataset = dataset_from_srm(path, repair_missing=True)
            except Exception as inner:
                QtWidgets.QMessageBox.critical(self, "SRM Import Error", str(inner))
                return False
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SRM Import Error", str(e))
            return False

        self._add_loaded_dataset(dataset, path)
        return True

    def open_srm(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open SRM File", "", "SRM files (*.xls *.txt *.dat);;All files (*.*)"
        )
        if not paths:
            return
        self._open_files_with_progress(paths, "SRM", self._import_srm_file)

    def _import_vib_file(self, path):
        dataset = dataset_from_vib(path)
        self._add_loaded_dataset(dataset, path)

    def open_vib(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open VIB File", "", "VIB files (*.*)")
        if not paths:
            return
        try:
            self._open_files_with_progress(paths, "VIB", self._import_vib_file)
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
            self.statusBar().showMessage(f"Exported dataset -> {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def copy_band_table_selection(self):
        selected_indexes = self.band_table.selectedIndexes()
        if not selected_indexes:
            return

        rows = sorted({index.row() for index in selected_indexes})
        cols = sorted({index.column() for index in selected_indexes})
        row_lookup = {row: i for i, row in enumerate(rows)}
        col_lookup = {col: i for i, col in enumerate(cols)}

        grid = [["" for _ in cols] for _ in rows]
        for index in selected_indexes:
            item = self.band_table.item(index.row(), index.column())
            grid[row_lookup[index.row()]][col_lookup[index.column()]] = item.text() if item is not None else ""

        text = "\n".join("\t".join(row) for row in grid)
        QtWidgets.QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Selected statistics copied to clipboard")

    def export_statistics_table(self):
        try:
            if self.band_table.rowCount() == 0 or self.band_table.columnCount() == 0:
                QtWidgets.QMessageBox.information(self, "Export Statistics", "No statistics table data to export.")
                return

            path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Statistics Table",
                "",
                "CSV files (*.csv);;Excel files (*.xlsx)"
            )
            if not path:
                return

            suffix = Path(path).suffix.lower()
            if not suffix:
                if "xlsx" in selected_filter.lower():
                    path = str(Path(path).with_suffix(".xlsx"))
                    suffix = ".xlsx"
                else:
                    path = str(Path(path).with_suffix(".csv"))
                    suffix = ".csv"

            vertical_header_title = self.band_table.verticalHeaderItem(0)
            leading_header = vertical_header_title.text() if vertical_header_title and vertical_header_title.text().strip() else "Channel"
            headers = [leading_header] + [self.band_table.horizontalHeaderItem(col).text() for col in range(self.band_table.columnCount())]
            rows = []
            for row in range(self.band_table.rowCount()):
                vertical_item = self.band_table.verticalHeaderItem(row)
                row_values = [vertical_item.text() if vertical_item is not None else ""]
                for col in range(self.band_table.columnCount()):
                    item = self.band_table.item(row, col)
                    row_values.append(item.text() if item is not None else "")
                rows.append(row_values)

            def _excel_value(text):
                text = str(text).strip()
                if text == "":
                    return ""
                try:
                    value = float(text)
                    if np.isfinite(value) and value.is_integer():
                        return int(value)
                    return value
                except ValueError:
                    return text

            if suffix == ".csv":
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(rows)
            elif suffix == ".xlsx":
                try:
                    from openpyxl import Workbook
                except ImportError as e:
                    raise RuntimeError(
                        "Excel export requires the openpyxl package. Install it or export as CSV instead."
                    ) from e

                wb = Workbook()
                ws = wb.active
                ws.title = "Statistics"
                ws.append(headers)
                for row_values in rows:
                    ws.append([_excel_value(value) for value in row_values])
                wb.save(path)
            else:
                raise RuntimeError("Unsupported export file type. Use .csv or .xlsx.")

            self.statusBar().showMessage(f"Statistics table exported: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Statistics Error", str(e))
    def _tree_item_kind(self, item):
        return str(item.data(0, QtCore.Qt.UserRole) or "") if item is not None else ""

    def _is_channel_tree_item(self, item):
        return self._tree_item_kind(item) == "channel"

    def _is_group_tree_item(self, item):
        return self._tree_item_kind(item) == "group"

    def _is_settings_tree_item(self, item):
        return self._tree_item_kind(item) in ("settings", "time_offset")

    def _tree_top_group_item(self, item):
        current = item
        while current is not None and current.parent() is not None:
            current = current.parent()
        return current

    def _channel_tree_group_name(self, item):
        top_item = self._tree_top_group_item(item)
        return top_item.text(0) if top_item is not None else ""

    def _channel_tree_name(self, item):
        if not self._is_channel_tree_item(item):
            return ""
        return str(item.data(0, QtCore.Qt.UserRole + 1) or "")

    def _group_time_offset(self, group_name):
        group = self.dataset.get_group(group_name)
        props = group.setdefault("props", {})
        try:
            return float(props.get("TimeOffset", 0.0))
        except Exception:
            return 0.0

    def _iter_channel_tree_items(self):
        for i in range(self.channel_list.topLevelItemCount()):
            group_item = self.channel_list.topLevelItem(i)
            for j in range(group_item.childCount()):
                child = group_item.child(j)
                if self._is_channel_tree_item(child):
                    yield child

    def _capture_group_tree_expansion_state(self):
        state = {}
        for i in range(self.channel_list.topLevelItemCount()):
            item = self.channel_list.topLevelItem(i)
            if item is not None:
                state[item.text(0)] = bool(item.isExpanded())
        return state

    def populate_channels(self):
        expanded_state = self._capture_group_tree_expansion_state()
        self.channel_list.blockSignals(True)
        self.channel_list.clear()
        self.channel_map.clear()

        for group_name in self.dataset.get_group_names():
            group = self.dataset.get_group(group_name)
            group.setdefault("props", {})
            group["props"].setdefault("TimeOffset", 0.0)

            group_item = QtWidgets.QTreeWidgetItem([group_name])
            group_item.setData(0, QtCore.Qt.UserRole, "group")
            group_item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.channel_list.addTopLevelItem(group_item)

            settings_item = QtWidgets.QTreeWidgetItem(["Settings"])
            settings_item.setData(0, QtCore.Qt.UserRole, "settings")
            settings_item.setFlags(QtCore.Qt.ItemIsEnabled)
            group_item.addChild(settings_item)

            offset_value = self._group_time_offset(group_name)
            offset_item = QtWidgets.QTreeWidgetItem([f"Time offset: {offset_value:g}"])
            offset_item.setData(0, QtCore.Qt.UserRole, "time_offset")
            offset_item.setFlags(QtCore.Qt.ItemIsEnabled)
            settings_item.addChild(offset_item)

            for name, data in group["channels"].items():
                unit = str(data.get("unit", "")).strip()
                display_name = f"{name} [{unit}]" if unit else name
                channel_item = QtWidgets.QTreeWidgetItem([display_name])
                channel_item.setData(0, QtCore.Qt.UserRole, "channel")
                channel_item.setData(0, QtCore.Qt.UserRole + 1, name)
                channel_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                group_item.addChild(channel_item)

            group_item.setExpanded(bool(expanded_state.get(group_name, False)))

        self.channel_list.blockSignals(False)

    def refresh_group_channel_highlight(self):
        selected_by_group = {
            group_name: set(self.group_selection_state.get(group_name, []))
            for group_name in self.dataset.get_group_names()
        }

        self.channel_list.blockSignals(True)
        self.channel_list.clearSelection()

        for item in self._iter_channel_tree_items():
            group_name = self._channel_tree_group_name(item)
            channel_name = self._channel_tree_name(item)
            if channel_name in selected_by_group.get(group_name, set()):
                item.setSelected(True)

        self.channel_list.blockSignals(False)

    def on_channel_tree_item_clicked(self, item, column):
        group_name = self._channel_tree_group_name(item)
        if not group_name:
            return
        idx = self.group_combo.findText(group_name)
        if idx >= 0 and idx != self.group_combo.currentIndex():
            self.group_combo.setCurrentIndex(idx)
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
        self.plot_channels(preserve_view=True)

    def clear_channel_selection_all_groups(self):
        self.group_selection_state = {g: [] for g in self.dataset.get_group_names()}
        self.refresh_group_channel_highlight()
        self.update_info_panel()
        self.plot_channels()

    def _save_current_group_selection(self):
        self._save_current_group_selection_for_name(self.current_group_name())

    def _save_current_group_selection_for_name(self, group_name):
        selected_by_group = {g: [] for g in self.dataset.get_group_names()}
        for item in self.channel_list.selectedItems():
            if not self._is_channel_tree_item(item):
                continue
            selected_group = self._channel_tree_group_name(item)
            channel_name = self._channel_tree_name(item)
            if selected_group and channel_name:
                selected_by_group.setdefault(selected_group, []).append(channel_name)

        for selected_group, selected_names in selected_by_group.items():
            ordered_names = list(self.dataset.get_group(selected_group)["channels"].keys())
            selected_set = set(selected_names)
            self.group_selection_state[selected_group] = [
                name for name in ordered_names if name in selected_set
            ]
    def _capture_xy_pair_state(self):
        pair_state = []
        for pair in self.xy_pairs:
            pair_state.append({
                "enabled": pair["enable"].isChecked(),
                "x": pair["x_combo"].currentText(),
                "y": pair["y_combo"].currentText(),
            })
        return pair_state

    def _set_xy_pair_visibility(self, visible):
        for pair in self.xy_pairs:
            pair["row_widget"].setVisible(bool(visible))

    def _create_xy_pair(self, index):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        enable_cb = QtWidgets.QCheckBox()
        enable_cb.setChecked(index == 0)
        enable_cb.setObjectName(f"xy_enable_{index}")

        x_label = QtWidgets.QLabel(f"X{index + 1}:")
        y_label = QtWidgets.QLabel(f"Y{index + 1}:")
        x_label.setMinimumWidth(24)
        y_label.setMinimumWidth(24)

        x_combo = QtWidgets.QComboBox()
        y_combo = QtWidgets.QComboBox()
        x_combo.setObjectName(f"xy_x_combo_{index}")
        y_combo.setObjectName(f"xy_y_combo_{index}")
        x_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        y_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        x_combo.setMinimumWidth(140)
        y_combo.setMinimumWidth(140)

        row_layout.addWidget(enable_cb)
        row_layout.addWidget(x_label)
        row_layout.addWidget(x_combo, 1)
        row_layout.addWidget(y_label)
        row_layout.addWidget(y_combo, 1)

        enable_cb.toggled.connect(self._update_band_plot)
        x_combo.currentIndexChanged.connect(self._update_band_plot)
        y_combo.currentIndexChanged.connect(self._update_band_plot)

        self.xy_pairs_layout.addWidget(row_widget)
        self.xy_pairs.append({
            "enable": enable_cb,
            "x_label": x_label,
            "y_label": y_label,
            "x_combo": x_combo,
            "y_combo": y_combo,
            "row_widget": row_widget,
        })

    def _collect_selected_xy_defaults(self):
        defaults = []
        plot_names = list(self.plotted_data.keys())

        for i in range(0, len(plot_names), 2):
            x_name = plot_names[i]
            y_name = plot_names[i + 1] if i + 1 < len(plot_names) else x_name
            defaults.append({"x": x_name, "y": y_name, "enabled": True})
        return defaults

    def _set_xy_pair_count(self, count, pair_state=None):
        count = max(1, int(count))
        while self.xy_pairs_layout.count():
            item = self.xy_pairs_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.xy_pairs = []
        for index in range(count):
            self._create_xy_pair(index)

        self.update_xy_channel_selectors(prefill=bool(self.xy_mode_checkbox.isChecked()), preferred_pairs=pair_state)
        self._set_xy_pair_visibility(self.xy_mode_checkbox.isChecked())

    def _on_xy_pair_count_changed(self, value):
        pair_state = self._capture_xy_pair_state()
        self._set_xy_pair_count(value, pair_state=pair_state)
        self._update_band_plot()

    def update_xy_channel_selectors(self, prefill=False, preferred_pairs=None):
        plot_names = list(self.plotted_data.keys())
        preferred_pairs = list(preferred_pairs or [])

        for idx, pair in enumerate(self.xy_pairs):
            x_combo = pair["x_combo"]
            y_combo = pair["y_combo"]

            current_x = x_combo.currentText()
            current_y = y_combo.currentText()
            preferred = preferred_pairs[idx] if idx < len(preferred_pairs) else {}
            target_x = str(preferred.get("x", "")).strip() if prefill else current_x
            target_y = str(preferred.get("y", "")).strip() if prefill else current_y

            x_combo.blockSignals(True)
            y_combo.blockSignals(True)
            x_combo.clear()
            y_combo.clear()

            for name in plot_names:
                x_combo.addItem(name)
                y_combo.addItem(name)

            if target_x and x_combo.findText(target_x) >= 0:
                x_combo.setCurrentText(target_x)
            elif current_x and x_combo.findText(current_x) >= 0:
                x_combo.setCurrentText(current_x)
            elif x_combo.count() > 0:
                default_index = min(idx * 2, x_combo.count() - 1)
                x_combo.setCurrentIndex(default_index)

            if target_y and y_combo.findText(target_y) >= 0:
                y_combo.setCurrentText(target_y)
            elif current_y and y_combo.findText(current_y) >= 0:
                y_combo.setCurrentText(current_y)
            elif y_combo.count() > 0:
                default_index = min(idx * 2 + 1, y_combo.count() - 1)
                y_combo.setCurrentIndex(default_index)

            pair["enable"].setChecked(bool(preferred.get("enabled", idx == 0)) if prefill else pair["enable"].isChecked())
            x_combo.blockSignals(False)
            y_combo.blockSignals(False)

        if self.xy_pairs and not any(pair["enable"].isChecked() for pair in self.xy_pairs):
            self.xy_pairs[0]["enable"].setChecked(True)

    def toggle_xy_mode(self, checked):
        visible = bool(checked)
        self._set_xy_pair_visibility(visible)
        if visible:
            self.update_xy_channel_selectors(prefill=True, preferred_pairs=self._collect_selected_xy_defaults())
        self._update_band_plot()
    def plot_channels(self, preserve_view=False):
        current_range = self.plot.getViewBox().viewRange() if preserve_view else None
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
                x = np.asarray(ch["x"], dtype=float) + self._group_time_offset(group_name)
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

        if preserve_view and current_range is not None:
            x_range, y_range = current_range
            self.plot.setXRange(float(x_range[0]), float(x_range[1]), padding=0)
            self.plot.setYRange(float(y_range[0]), float(y_range[1]), padding=0)
        else:
            self.plot.autoRange()
        self.update_xy_channel_selectors(prefill=False)
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
                self.band_plot.plot(x_common, y_common, pen=pg.intColor(i), symbol=None, name=f"{y_name} vs {x_name}")

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
            mask = (x >= lo) & (x <= hi)
            xb = x[mask]
            yb = y[mask]
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
                    pkpk98 = np.percentile(samples, 99) - np.percentile(samples, 1)
                    pkpk95 = np.percentile(samples, 97.5) - np.percentile(samples, 2.5)
                    std = np.std(samples)
                    rms = np.sqrt(np.mean(samples ** 2))
                    ac_rms = np.sqrt(np.mean((samples - mean) ** 2))
                else:
                    mean = minv = maxv = pkpk = pkpk98 = pkpk95 = std = rms = ac_rms = np.nan
                header_text = str(name)[:20]
                self.band_table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(header_text))
                values = [unit, y1, y2, y2 - y1, mean, minv, maxv, pkpk, pkpk98, pkpk95, std, rms, ac_rms]
                for col, val in enumerate(values):
                    text = f"{val:.6g}" if isinstance(val, (float, np.floating)) else str(val)
                    self.band_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))

            self.band_table.resizeColumnsToContents()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Band Error", str(e))

    def _selected_base_channel_names(self):
        names = []
        current_group = self.current_group_name().strip()
        for item in self.channel_list.selectedItems():
            if not self._is_channel_tree_item(item):
                continue
            if self._channel_tree_group_name(item) != current_group:
                continue
            channel_name = self._channel_tree_name(item)
            if channel_name:
                names.append(channel_name)
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
                x = np.asarray(ch["x"], dtype=float) + self._group_time_offset(group_name)
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
                x = np.asarray(ch["x"], dtype=float) + self._group_time_offset(group_name)
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


    def filter_scale_linear(self):
        try:
            selected = self._selected_base_channel_names()
            if not selected:
                QtWidgets.QMessageBox.information(self, "Filter", "Select at least one channel.")
                return

            cfg = self.filter_settings["scale_linear"]
            gain, ok = QtWidgets.QInputDialog.getDouble(
                self, "Scale Signal", "Gain:",
                float(cfg.get("gain", 1.0)), -1e12, 1e12, 6
            )
            if not ok:
                return

            offset, ok = QtWidgets.QInputDialog.getDouble(
                self, "Scale Signal", "Offset:",
                float(cfg.get("offset", 0.0)), -1e12, 1e12, 6
            )
            if not ok:
                return

            gain = float(gain)
            offset = float(offset)
            cfg["gain"] = gain
            cfg["offset"] = offset

            self._push_undo_state(f"scale linear ({gain:g}, {offset:g})")
            new_names = []
            group = self.current_group()

            for ch_name in selected:
                y = np.asarray(group["channels"][ch_name]["y"], dtype=float)
                y_new = gain * y + offset
                new_name = f"{ch_name}_scaled_{gain:g}_{offset:g}"
                self._create_filtered_channel(ch_name, new_name, y_new)
                new_names.append(new_name)

            self._refresh_channels_after_processing(new_names)
            self.update_info_panel()
            self.statusBar().showMessage(f"Created {len(new_names)} scaled channel(s)")
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

        global_pos = self.channel_list.viewport().mapToGlobal(pos)
        if self._is_group_tree_item(item):
            self._show_group_actions_menu(item.text(0), global_pos, include_tree_actions=True)
            return

        if self._is_settings_tree_item(item):
            group_name = self._channel_tree_group_name(item)
            menu = QtWidgets.QMenu(self)
            edit_offset_action = menu.addAction(f"Set time offset for '{group_name}'")
            reset_offset_action = menu.addAction(f"Reset time offset for '{group_name}'")
            action = menu.exec_(global_pos)
            if action == edit_offset_action:
                self.set_group_time_offset(group_name)
            elif action == reset_offset_action:
                self.set_group_time_offset(group_name, new_offset=0.0)
            return

        if not self._is_channel_tree_item(item):
            return

        group_name = self._channel_tree_group_name(item)
        channel_name = self._channel_tree_name(item)
        group = self.dataset.get_group(group_name)
        channel_names = list(group["channels"].keys())
        try:
            index = channel_names.index(channel_name)
        except ValueError:
            return

        menu = QtWidgets.QMenu(self)
        move_up_action = menu.addAction("Move up")
        move_down_action = menu.addAction("Move down")
        move_up_action.setEnabled(index > 0)
        move_down_action.setEnabled(index < len(channel_names) - 1)
        menu.addSeparator()
        rename_action = menu.addAction("Rename channel")
        delete_action = menu.addAction("Delete channel")
        action = menu.exec_(global_pos)

        if action == move_up_action:
            self.move_channel(item, -1)
        elif action == move_down_action:
            self.move_channel(item, 1)
        elif action == rename_action:
            self.rename_channel(item)
        elif action == delete_action:
            self.delete_channel(item)
    def set_group_time_offset(self, group_name, new_offset=None):
        try:
            group_name = str(group_name).strip()
            if not group_name or group_name not in self.dataset.groups:
                return

            current_offset = self._group_time_offset(group_name)
            if new_offset is None:
                new_offset, ok = QtWidgets.QInputDialog.getDouble(
                    self,
                    "Group Time Offset",
                    f"Time offset for '{group_name}':",
                    current_offset,
                    -1e12,
                    1e12,
                    6,
                )
                if not ok:
                    return

            new_offset = float(new_offset)
            if new_offset == current_offset:
                return

            self._push_undo_state(f"set time offset {group_name} -> {new_offset:g}")
            self.dataset.groups[group_name].setdefault("props", {})["TimeOffset"] = new_offset
            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels(preserve_view=True)
            self.statusBar().showMessage(f"Updated time offset: {group_name} -> {new_offset:g}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Time Offset Error", str(e))
    def move_channel(self, item, direction):
        try:
            if not self._is_channel_tree_item(item):
                return

            channel_name = self._channel_tree_name(item)
            group_name = self._channel_tree_group_name(item)
            group = self.dataset.get_group(group_name)
            channel_names = list(group["channels"].keys())
            idx = channel_names.index(channel_name)
            new_idx = idx + int(direction)
            if new_idx < 0 or new_idx >= len(channel_names):
                return

            self._push_undo_state(f"move channel {channel_name}")
            channel_names[idx], channel_names[new_idx] = channel_names[new_idx], channel_names[idx]
            group["channels"] = {name: group["channels"][name] for name in channel_names}

            selected_set = set(self.group_selection_state.get(group_name, []))
            self.group_selection_state[group_name] = [name for name in channel_names if name in selected_set]

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.plot_channels()
            self.update_info_panel()
            self.statusBar().showMessage(f"Moved channel: {channel_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Move Channel Error", f"Could not move channel.\n\n{e}")

    def rename_channel(self, item):
        try:
            if not self._is_channel_tree_item(item):
                return

            old_name = self._channel_tree_name(item)
            group_name = self._channel_tree_group_name(item)
            group = self.dataset.get_group(group_name)

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
            if not self._is_channel_tree_item(item):
                return

            base_name = self._channel_tree_name(item)
            group_name = self._channel_tree_group_name(item)
            display_name = item.text(0)

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

            group = self.dataset.get_group(group_name)
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


    def _show_group_actions_menu(self, group_name, global_pos, include_tree_actions=False):
        group_name = str(group_name).strip()
        if not group_name or group_name not in self.dataset.groups:
            return

        items = list(self.dataset.groups.keys())
        idx = items.index(group_name) if group_name in items else -1

        menu = QtWidgets.QMenu(self)
        move_up_action = menu.addAction("Move up")
        move_down_action = menu.addAction("Move down")
        move_top_action = menu.addAction("Move to top")
        move_bottom_action = menu.addAction("Move to bottom")
        sort_action = menu.addAction("Sort groups")
        collapse_all_action = menu.addAction("Collapse all groups") if include_tree_actions else None
        if idx == 0:
            move_up_action.setEnabled(False)
            move_top_action.setEnabled(False)
        if idx == len(items) - 1:
            move_down_action.setEnabled(False)
            move_bottom_action.setEnabled(False)
        menu.addSeparator()
        rename_action = menu.addAction(f"Rename group '{group_name}'")
        delete_action = menu.addAction(f"Delete group '{group_name}'")

        action = menu.exec_(global_pos)
        if action == move_up_action:
            self.move_group(group_name, -1)
        elif action == move_down_action:
            self.move_group(group_name, 1)
        elif action == move_top_action:
            self.move_group_to_edge(group_name, to_top=True)
        elif action == move_bottom_action:
            self.move_group_to_edge(group_name, to_top=False)
        elif action == collapse_all_action:
            self.collapse_all_group_nodes()
        elif action == sort_action:
            self.sort_groups_descending()
        elif action == rename_action:
            self.rename_group(group_name)
        elif action == delete_action:
            self.delete_group(group_name)

    def collapse_all_group_nodes(self):
        for i in range(self.channel_list.topLevelItemCount()):
            item = self.channel_list.topLevelItem(i)
            if item is not None:
                item.setExpanded(False)
        self.statusBar().showMessage("Collapsed all group nodes")
    def show_group_context_menu(self, pos):
        group_name = self._group_name_at_pos(pos)
        if not group_name:
            return
        self._show_group_actions_menu(group_name, self.group_combo.mapToGlobal(pos))
    def show_group_view_context_menu(self, pos):
        view = self.group_combo.view()
        index = view.indexAt(pos)
        if not index.isValid():
            return

        group_name = index.data()
        self._show_group_actions_menu(group_name, view.viewport().mapToGlobal(pos))
    def sort_groups_descending(self):
        try:
            if len(self.dataset.groups) < 2:
                return

            self._push_undo_state("sort groups descending")
            current_group = self.current_group_name().strip()
            items = sorted(self.dataset.groups.items(), key=lambda item: item[0], reverse=True)
            self.dataset.groups = dict(items)

            self.populate_groups()
            if current_group:
                idx = self.group_combo.findText(current_group)
                if idx >= 0:
                    self.group_combo.setCurrentIndex(idx)

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels()
            self.statusBar().showMessage("Groups sorted descending")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Sort Groups Error", f"Could not sort groups.\n\n{e}")


    def move_group_to_edge(self, group_name, to_top=True):
        try:
            group_name = str(group_name).strip()
            if not group_name or group_name not in self.dataset.groups:
                return

            items = list(self.dataset.groups.items())
            idx = next((i for i, (name, _) in enumerate(items) if name == group_name), -1)
            if idx < 0:
                return

            target_idx = 0 if to_top else len(items) - 1
            if idx == target_idx:
                return

            self._push_undo_state(f"move group {group_name}")
            item = items.pop(idx)
            items.insert(target_idx, item)
            self.dataset.groups = dict(items)

            current_group = self.current_group_name().strip() or group_name
            self.populate_groups()
            combo_idx = self.group_combo.findText(current_group)
            if combo_idx >= 0:
                self.group_combo.setCurrentIndex(combo_idx)

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels()
            edge_name = "top" if to_top else "bottom"
            self.statusBar().showMessage(f"Moved group to {edge_name}: {group_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Move Group Error", f"Could not move group.\n\n{e}")
    def move_group(self, group_name, direction):
        try:
            group_name = str(group_name).strip()
            if not group_name or group_name not in self.dataset.groups:
                return

            items = list(self.dataset.groups.items())
            idx = next((i for i, (name, _) in enumerate(items) if name == group_name), -1)
            if idx < 0:
                return

            new_idx = idx + int(direction)
            if new_idx < 0 or new_idx >= len(items):
                return

            self._push_undo_state(f"move group {group_name}")
            items[idx], items[new_idx] = items[new_idx], items[idx]
            self.dataset.groups = dict(items)

            current_group = self.current_group_name().strip() or group_name
            self.populate_groups()
            combo_idx = self.group_combo.findText(current_group)
            if combo_idx >= 0:
                self.group_combo.setCurrentIndex(combo_idx)

            self.populate_channels()
            self.refresh_group_channel_highlight()
            self.update_info_panel()
            self.plot_channels()
            self.statusBar().showMessage(f"Moved group: {group_name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Move Group Error", f"Could not move group.\n\n{e}")

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
        elif isinstance(widget, QtWidgets.QSpinBox):
            data["value"] = widget.value()
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
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(data.get("value", widget.value())))
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
            "notes": self.notes_box.toPlainText(),
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
        self.notes_box.setPlainText(str(state.get("notes", "")))

        widgets_state = state.get("widgets", {})
        if "xy_pair_count_spin" in widgets_state:
            self._apply_widget_state(self.xy_pair_count_spin, widgets_state["xy_pair_count_spin"])
        self._set_xy_pair_count(self.xy_pair_count_spin.value())

        current_group = str(state.get("current_group", "")).strip()
        if current_group:
            idx = self.group_combo.findText(current_group)
            if idx >= 0:
                self.group_combo.setCurrentIndex(idx)

        for widget in self.findChildren(QtWidgets.QWidget):
            name = widget.objectName().strip() if widget.objectName() else ""
            if (
                not name
                or name == "xy_pair_count_spin"
                or name.startswith("xy_enable_")
                or name.startswith("xy_x_combo_")
                or name.startswith("xy_y_combo_")
                or name not in widgets_state
            ):
                continue
            self._apply_widget_state(widget, widgets_state[name])

        self.populate_channels()
        self.refresh_group_channel_highlight()
        self.plot_channels()

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







































