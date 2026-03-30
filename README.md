# MyScope

MyScope is a desktop signal-analysis tool for viewing, inspecting, and processing measurement data. It is built in Python with Qt and `pyqtgraph`, and is aimed at engineering workflows where you need to open waveform data, compare channels, inspect regions of interest, and create derived signals with common filters.

## Features

- Open and combine datasets from `TDMS`, `sLAB`, `SRM`, and `VIB` files
- Organize data by groups and channels
- Plot multiple channels at once in the main view
- Inspect a selected band/region with detailed statistics
- Switch the lower view into XY plotting mode
- Create derived channels using:
  - Moving average
  - Low-pass filter
  - High-pass filter
  - Band-pass filter
  - Moving-window peak-to-peak
  - Moving-window RMS
  - Mean subtraction
- Rename and delete groups/channels
- Undo processing actions
- Save and reopen project state
- Export processed datasets to TDMS

## Requirements

- Windows
- Python 3.10+ recommended

Python packages used by the application:

- `numpy`
- `scipy`
- `pyqtgraph`
- `qtpy`
- `nptdms`
- A Qt binding supported by `qtpy`, such as `PyQt5`, `PyQt6`, `PySide2`, or `PySide6`

## Installation

Create and activate a virtual environment if you want an isolated setup:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install numpy scipy pyqtgraph qtpy nptdms PyQt5
```

If your environment already uses a different Qt binding, install that instead of `PyQt5`.

## Running MyScope

Start the application with:

```powershell
python MyScope_0_3_8.py
```

## How It Works

### Loading data

Use the buttons on the left side of the window or the `File` menu to open supported files. If data is already loaded, newly opened files are appended as additional groups.

### Exploring signals

- Select a group from the group selector
- Select one or more channels to plot
- Use the zoom tools for box zoom, X-only zoom, Y-only zoom, or pan
- Drag the band region on the main plot to inspect a smaller portion of the signal

### Band statistics

The right-side table calculates values for the selected band, including:

- `Y@X1`
- `Y@X2`
- `Delta Y`
- Mean
- Min / Max
- Peak-to-peak
- Standard deviation
- RMS
- AC RMS

### XY mode

Enable `XY plot` to use the lower plot as an X/Y comparison view instead of a time-domain band plot. Up to four XY pairs can be enabled.

### Signal processing

The `Filters` menu contains the signal-processing tools. Each processing action creates a new derived channel instead of overwriting the original one.

Current filters:

- Moving average
- Low-pass
- High-pass
- Band-pass
- Moving-window peak-to-peak
- Moving-window RMS
- Subtract mean

## Project Files

MyScope can save a project as:

- A `.prj` file containing UI state, filter settings, selected channels, and plot state
- A companion `.tdms` file containing the dataset itself

This lets you reopen your analysis session later with the same view configuration.

## Notes

- TDMS export expects uniformly sampled data
- Some imported legacy text formats are read using `cp1250` encoding
- The application is currently implemented as a single Python file

## Repository Contents

- `MyScope_0_3_8.py` - main application source
- `README.md` - project overview and usage instructions
