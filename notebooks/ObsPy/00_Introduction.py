# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">ObsPy Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Introduction/Index</div>
#         </div>
#     </div>
# </div>

# ![](images/obspy_logo_full_524x179px.png)

# These notebooks are an introduction to ObsPy and will take about 2 days to fully complete. Some basic knowledge of scientific Python is assumed.

# ## Timetable/Contents
#
# 1. **[Introduction to File Formats and read/write in ObsPy](01_File_Formats.ipynb)** [45 Minutes]  -  *[[Solution]](01_File_Formats-with_solutions.ipynb)*
# 2. **[Handling Time](02_UTCDateTime.ipynb)** [35 Minutes]  -  *[[Solution]](02_UTCDateTime-with_solutions.ipynb)*
# 3. **[Handling Waveform Data: Stream & Trace Classes](03_waveform_data.ipynb)** [45 Minutes]  -  *[[Solution]](03_waveform_data-with_solutions.ipynb)*
# 4. **[Handling Station Metainformation](04_Station_metainformation.ipynb)** [15 Minutes]  -  *[[Solution]](04_Station_metainformation-with_solutions.ipynb)*
# 5. **[Handling Event Metainformation](05_Event_metadata.ipynb)** [25 Minutes]  -  *[[Solution]](05_Event_metadata-with_solutions.ipynb)*
# 6. **[Retrieving Data from Data Centers](06_FDSN.ipynb)** [30 Minutes]  -  *[[Solution]](06_FDSN-with_solutions.ipynb)*
# 7. **[Basic Downloading/Processing Exercise](07_Basic_Processing_Exercise.ipynb)** [60 - x Minutes]  -  *[[Solution]](07_Basic_Processing_Exercise-with_solutions.ipynb)*
# 8. **[Exercise: Event Detection and Magnitude Estimation in an Aftershock Series](08_Exercise__2008_MtCarmel_Earthquake_and_Aftershock_Series.ipynb)** [open end]  -  *[[Solution]](08_Exercise__2008_MtCarmel_Earthquake_and_Aftershock_Series-with_solutions.ipynb)*
