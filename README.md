# C-FLEX
Flexible pavement analysis software package developed for the [Department of Defense (DoD)](https://www.defense.gov).

## Author Information
The program is developed by Haohang Huang and Jiayi Luo @ University of Illinois at Urbana-Champaign (UIUC). The advisor is [Professor Erol Tutumluer](https://cee.illinois.edu/directory/profile/tutumlue).

The development of this software program is sponsored by [Engineer Research and Development Center (ERDC)](https://www.erdc.usace.army.mil), United States Army Corps of Engineers (USACE). The ERDC project manager is [Jeb S. Tingle](https://www.erdc.usace.army.mil/Media/Images/igphoto/2002117275/).

## Dependencies

This repository collects the libraries, extensions, Graphical User Interface (GUI) associated with the following standalone Finite Element Methods (FEM) engines developed by the authors:

* **C-FLEX2D**: 2D Axisymmetric FEM engine for flexible pavement analysis. [[github]](https://github.com/symphonylyh/C-FLEX2D)
* **FLEX3D**: 3D FEM engine for flexible pavement analysis. [[github]]((https://github.com/symphonylyh/FLEX3D))

Please see github repos above for more details of the FEM codes.

## Software Overview
This software is mainly written in Python. Modules include:
* 2D Mesh Generator
* 3D Mesh Generator
* 2D Superposition
* Software Verification
* Test Validation
* Graphical User Interface

## Installation Guide

### Prerequisites
Please follow the github repos [C-FLEX2D](https://github.com/symphonylyh/C-FLEX2D) and [FLEX3D]((https://github.com/symphonylyh/FLEX3D)) to compile the programs and put them under `/bin/` folder. Executables on Windows and MacOS platforms are given here as examples.

### Environment
[Anaconda](https://www.anaconda.com) is recommended to provide the environment of the Python code.

Create virtual environment:

`conda create --name fem`

Activate virtual env any time when you run the code:

`conda activate fem`

### Details
Please see the README files in each library for details of their functionalities, installations, etc.
