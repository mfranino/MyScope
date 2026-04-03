# MyScope 0.3.9

## Overview

MyScope is a desktop signal viewer and signal-analysis tool built with QtPy and PyQtGraph for importing, organizing, plotting, filtering, annotating, and exporting measurement data.

Supported input formats:
- TDMS
- sLAB
- SRM
- VIB

## Main Capabilities

### File import
- Open TDMS, sLAB, SRM, and VIB files
- Batch open multiple files from the file picker
- Append imported files as new groups when a workspace is already loaded
- Open saved project files (`.prj` with companion `.tdms`)
- Start a new empty workspace with `New Project`

### Group management
- Select groups from the group dropdown
- Rename groups
- Delete groups
- Move groups up and down in the dropdown order
- Move groups directly to the top or bottom from the context menu
- Sort groups descending (`Z -> A`, numeric descending) from the context menu
- Preserve independent channel selections per group

### Channel management
- List all channels for the selected group
- Show units in channel names when available
- Multi-select channels for plotting
- Clear channel selections across all groups with `Clear Ch selection`
- Rename channels
- Delete channels
- Reorder channels within a group with `Move up` and `Move down` from the context menu

### Plotting
- Plot selected channels automatically
- Main plot with white background, grid, legend, and custom zoom/pan modes
- Min/max envelope downsampling for responsive display
- Bottom plot for the active band range
- Optional XY mode in the bottom plot
- Dynamic XY pair count selector next to `XY plot`
- Per-pair XY enable checkbox plus X/Y channel selectors
- XY channel selectors can be auto-populated from the current plotted channel list order when XY mode is enabled
- Manual band-plot axis controls for `X axis range` and `Y axis range` with `Apply`

### Band analysis
- Enable or disable the analysis band
- Move and resize the band interactively on the main plot
- Show band statistics for plotted channels
- Display X1, X2, and dX band coordinates
- Band statistics table supports direct copy of selected cells to the clipboard with `Ctrl+C`
- Clipboard copy uses tab-separated rows so the selection can be pasted directly into Excel

### Filters
- Moving Average
- Low-pass Butterworth
- High-pass Butterworth
- Band-pass Butterworth
- Band-pass (Stable SOS)
- Moving Window Peak-to-Peak
- Moving Window RMS
- Subtract Mean

Filter results are added as new channels and the original data is preserved.

### Project save/open
- Save the current project state to `.prj`
- Save dataset data to companion `.tdms`
- Restore current group, channel selections, filter settings, plot ranges, splitters, widgets, and XY selections
- Save and restore project notes
- Save Project is available from the menu and with `Ctrl+S`

### Export
- Export the current dataset to TDMS
- Export the band statistics table to CSV
- Export the band statistics table to Excel (`.xlsx`) when `openpyxl` is installed
- Statistics export includes both horizontal column headers and vertical row headers

## File Format Notes

### TDMS
- TDMS is a binary format, not ASCII text
- Reads waveform channels
- Reads units when available
- Reconstructs the X axis from waveform properties
- Preserves root and group properties
- Exports waveform metadata including:
  - `wf_increment`
  - `wf_start_offset`
  - `wf_samples`
  - `wf_xname`
  - `wf_xunit_string`
  - `wf_start_time` when group date/time metadata can be parsed

### sLAB
- Reads metadata from the header
- Reads channel descriptions and units
- Handles malformed trailing tabs and missing fields
- Validates uniform time sampling

### SRM
- Assumes:
  - column 0 = Time
  - column 1 = X1
  - column 2 = Y1
  - column 3 = X2
  - column 4 = Y2
  - column 5 = X3
  - column 6 = Y3
  - column 7 = Pgen
  - column 8 = KP
- Reads scan rate from the header when available
- Accepts rounded time columns when they are consistent with the header scan rate
- Detects mismatch between header sampling rate and time column
- Can repair missing SRM samples by rebuilding a uniform timebase and interpolating signal values

### VIB
- Reads metadata from the header
- Reads sampling rate
- Builds the time axis from sampling rate
- Reads channel names from channel definitions/header
- Imports all valid numeric channels

## Information And Notes Panels

The info panel displays:
- Root properties
- Source file path for the selected group
- Group name
- Group properties
- Sampling rate
- dt
- Number of samples
- Number of channels in the selected group
- Duration

The right panel also includes:
- A `Notes` panel for project-only notes
- A live memory usage indicator in the lower-right corner of the main panel

## User Interface Summary

### Left panel
- File open buttons
- Loaded file or group summary
- Group selector with right-click group management actions
- Channel list with right-click channel management and reordering actions

### Center
- Main plot
- Bottom band or XY plot

### Right panel
- Band enable checkbox
- Band coordinate label
- XY controls and pair selectors
- Band statistics table
- Band axis range controls
- Information box
- Notes box
- Memory usage label

## Dependencies

Required Python packages:
- numpy
- scipy
- nptdms
- qtpy
- pyqtgraph

A Qt backend is also required, for example:
- PyQt5
- PyQt6
- PySide2
- PySide6

Excel export additionally requires:
- openpyxl

Typical installation example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy nptdms qtpy pyqtgraph openpyxl PyQt5
```

## Running

Start the application with:

```powershell
python MyScope.py
```

## Notes

- TDMS export requires uniformly sampled X data.
- Some imported text formats skip malformed rows.
- Legacy text-based formats are read using `cp1250` encoding.
- Main-plot downsampling affects display only, not stored data.
- Undo stores dataset snapshots and may use more memory for large projects.
- The application is currently implemented as a single Python file.
