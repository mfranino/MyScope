# MyScope 0.3.8

## Overview

MyScope is a desktop signal viewer and signal-analysis tool built with QtPy and PyQtGraph for opening, viewing, comparing, filtering, inspecting, and exporting measurement data.

This version supports:
- TDMS files
- sLAB files
- SRM files
- VIB files

## Main Features

### 1. File import
- Open TDMS
- Open sLAB
- Open SRM
- Open VIB
- Open saved project files (`.prj` + companion `.tdms`)
- Append newly loaded files as additional groups when data is already loaded

### 2. Group and channel browser
- List all groups in the dataset
- List all imported channels for the selected group
- Show units in channel names when available
- Multi-select channels
- Plot selected channels automatically
- Rename channels from context menu
- Delete channels from context menu
- Rename groups from context menu
- Delete groups from context menu

### 3. Main plot
- Interactive plotting of selected channels
- White background
- Grid enabled
- Legend in the top-right corner
- Mouse interaction through custom zoom/pan modes
- Uses min/max envelope downsampling for responsive display

### 4. Zoom and navigation
- Box Zoom
- Zoom X
- Zoom Y
- Pan
- Zoom to Band
- Reset View
- Auto Range
- Auto-scale lower-plot Y axis once
- Optional continuous lower-plot autoscale Y

### 5. Band analysis
- Enable/disable band region with checkbox
- Interactive band region on main graph
- Band statistics table for all plotted channels
- Statistics shown:
  - Y@X1
  - Y@X2
  - Delta Y
  - Mean
  - Min
  - Max
  - Peak-to-Peak
  - StdDev
  - RMS
  - AC RMS
- Band label with:
  - X1
  - X2
  - dX

### 6. Bottom graph
- Resizable lower plot panel
- Displays only data inside the current band
- Hidden when band is disabled
- Uses min/max envelope downsampling for responsive plotting
- Supports optional Y autoscaling controls

### 7. XY plot mode in bottom graph
- Checkbox to switch lower graph from time plot to XY plot
- Four independent XY curve slots
- Each slot contains:
  - enable checkbox
  - X channel selector
  - Y channel selector
- Multiple XY curves can be shown simultaneously
- XY controls are resizable with splitter

### 8. Plot performance
- Main graph uses min/max envelope downsampling
- Bottom graph uses min/max envelope downsampling
- Full-resolution data is still preserved for:
  - filters
  - band statistics
  - export
  - project save/load

### 9. Filters
- Moving Average
- Low-pass Butterworth
- High-pass Butterworth
- Band-pass Butterworth
- Band-pass (Stable SOS)
- Moving Window Peak-to-Peak
- Moving Window RMS
- Subtract Mean

Filter output:
- New filtered channels are created and stored in the dataset
- Original channels remain unchanged
- Filter settings are reused as defaults for the next run

### 10. Undo
- Ctrl+Z supported
- Undo restores previous dataset state
- Works for:
  - channel deletion
  - group deletion
  - rename operations
  - filter-generated channels

### 11. Project save/open
- Save project state to `.prj`
- Save dataset to companion `.tdms`
- Restore selected group and channels
- Restore filter settings
- Restore band position
- Restore plot ranges
- Restore splitters and widget state
- Restore XY plot selections

### 12. Export
- Export current dataset to TDMS
- Exports all groups and channels currently stored in dataset
- Uses waveform-style TDMS properties when sampling is uniform

## Supported File Formats

### 1. TDMS
- Reads waveform channels
- Reads units if available
- Reconstructs X axis from waveform properties
- Preserves root and group properties

### 2. sLAB
- Reads metadata from header
- Reads channel descriptions and units
- Handles irregular trailing tabs and missing fields
- Builds dataset directly without DataFrame
- Validates uniform time sampling

### 3. SRM
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
- Reads scan rate if available
- Validates uniform time sampling

### 4. VIB
- Reads metadata fields from header
- Reads scan rate
- Builds time axis from sampling rate
- Reads channel names from channel definitions/header
- Imports all valid numeric channels

## Information Panel

The info panel displays:
- Root properties
- Group name
- Group properties
- Sampling rate
- dt
- Number of samples
- Number of channels in current group
- Total number of channels in dataset
- Number of selected channels in current group
- Duration

## User Interface Summary

### Left panel
- Open buttons
- File path / loaded-file summary
- Group selector
- Channel list

### Center
- Main interactive plot
- Bottom band / XY plot

### Right panel
- Band enable checkbox
- Band coordinates label
- Band statistics table
- Information box

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

Typical installation example:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy nptdms qtpy pyqtgraph PyQt5
```

## Running

Start the application with:

```powershell
python MyScope_0_3_8.py
```

## Notes

- TDMS export requires uniformly sampled X data.
- Some imported text formats may skip malformed rows.
- Some imported legacy text formats are read using `cp1250` encoding.
- Band statistics are always computed from full-resolution data.
- Plot downsampling affects only display, not stored data.
- Undo stack stores dataset snapshots and may use more memory for large files.
- The application is currently implemented as a single Python file.

## Further Upgrade Suggestions

1. Notch filter
2. FFT / spectrum view
3. Viewport-based downsampling
4. Export statistics table to CSV
5. Export plot image / PDF
6. Channel search/filter box
7. Multi-axis plotting
8. Cursors and markers
9. Data quality checks
10. CSV / Excel import wizard
11. Batch processing
12. Comparison and alignment tools
13. Plugin-style custom analysis extensions

## Repository Contents

- `MyScope_0_3_8.py` - main application source
- `README.md` - project overview and usage instructions

## Version

Application name: MyScope  
Version: 0.3.8
