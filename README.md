# SIMULATION
Building dynamic simulation tool

Calculation of building heating and cooling demand during a cold wave, a hot wave or an average climatic year in Belgium.
The simulation includes the following features :
- permanent solar screens surrounding the windows
- blind control
- Radiant and convective heating system
- Radiant and convective cooling system
- Free cooling by air extraction
- Free chilling with dry cooler

Weather data files must be stored in the folder of the Jupyter Notebook.
Python file simulation_231217.py must be stored in the folder of the Jupyter Notebook.

The required data are described by the comments listed in the Jupyter Notebook.

# INPUTS:

weather data file 
Latitude and longitude of the location of analysis (and of weather data)
Temperature set points for the heating and cooling system during occupancy
Min temperature set point for the heating system and Max temperature setpoint for the cooling system during unoccupancy
Lighting level requirement (lux)
Daylight factor
Leakness air renewal rate at 50 Pa (1/h)
Reduction factor of the 50 Pa leakness air renewal rate for average weather conditions
Building environment reflexion factor (-)
Glazing U-value [W/m2/K]
Windows frame fraction (-)
Glazing Luminous Transmission Factor (-)
Glazing solar heat gains factor (-)
Blinds solar heat gain factor (-)
Internal convection heat transfert coefficient [W/m2/K]
Long wave infrared emmission factor (-)
Simulation time step (s)
Number of working hours of the night radiant cooling system preceeding the start of the cooling convective system (-)
Fresh air supply controlled by CO2 probe (True/False)
Lighting controled by occupancy detector (True/False)

## Zone by zone occupancy profiles
Zone name,
Occupancy profile
Number of occupants
Radiative heat gain per occupant (W/occ), Convective heat gain per occupant (W/occ), 
Appliance heat gain per occupant (W/occ)
Radiative heat gain from lighting (W/m²), Convective heat gain from lighting (W/m²)
Installed heating power           (W/m²), Installed cooling power

## HVAC system schedules
Zone name,
Hourly profile
Daily profile

## Ventilation data
Zone name,
number of floors, total floor area, internal volume, 
fan supply air flow (m³/h), fan extracted air flow (m³/h), 
efficiency of heat recovery, free-cooling air change rate (1/h)

## Wall construction data
Five allowed wall types names : floor, ceiling, wall_ext, wall_uncond, wall_cond
Material and thicknesses of the wall layers from indoor to outdoor
List of external wall types (one face exposed to the sun radiation and to the infrared heat losses)
List of internal walls types in contact with a conditioned zone
The walls that are not included in the two preceeding lists 
are supposed to be in contact with an unconditionned zone 
whose temperature is the average between the external and the internal temperatures)

## Wall dimensions and radiative emission
Wall by wall:
Zone name, azimuth, slope, wall_type, dimension 1, dimension 2, 
Fraction of the zone heating power provided by the wall as radiant heating emmission
Fraction of the zone cooling power provided by the wall as radiant cooling emmission

## Windows dimensions and solar screens
Window type by window type:
Zone name, azimuth, number of identical windows, breadth, height
Side screens angle (deg), Top screen angle (deg)
Angle of front screen (deg), ratio of the screen distance to the window height
(Angles are measured from the center of the windows).

The results are written in an Excel file (one sheet per building zone)
# OUTPUTS
'hour_yr’,  	  Hour					(h)
't_out’, 		    External temperature 				(°C)
't_in’, 		    Internal temperature 				(°C)
‘f_occ’, 		    Occupancy rate				(--)
‘q_m3h’, 		    Ventilation air flow					(m³/h)
‘Qh_Wm2’, 	    Heating power demand per square meter of floor area	(W/m²)
'Qc _Wm2’, 	    Cooling power demand per square meter of floor area	(W/m²)
'Ql _Wm2’, 	    Lighting electric power	per square meter of floor area		(W/m²)
'Qs _Wm2’, 	    Solar heat gains	per square meter of floor area		(W/m²)
'Qf _Wm2’, 	    Dry cooler free chilling cooling power by radiant ceiling (W/m²)
'overcool’,		  Occurence of an overcooling uncomfort  (--)
'overheat’, 	  Occurence of an overheating uncomfort	(--)
'Qh_kWhm2’, 	  Heating energy demand par square meter of floor area 	(kWh/m²)
'Qc_kWhm2’, 	  Cooling energy demand par square meter of floor area 	(kWh/m²)
'Ql_kWhm2’, 	  Lighting electric consumption	per square meter of floor area	(kWh/m²)
'Qs_kWhm2’, 	  Internal solar heat energy per square meter of floor area	(kWh/m²)
'Qf_kWhm2’, 	  Dry cooler free chilling cooling energy	per square meter of floor area	(kWh/m²)
'fr_overcool’,  Fraction of the occupancy time with occurence of overcooling uncomfort	(--)
'fr_overheat’ 	Fraction of the occupancy time with occurence of overheating uncomfort		(--)



