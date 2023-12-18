import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

from scipy.interpolate import interpn
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from math import sqrt, atan, log, exp, sin, cos, tan


pi = np.pi
rho_a = 1.197 #[kg/m^3]
c_p_a = 1020  #[J/kg-K] specific heat capacity of humid air per kg of dry air
sigma_b = 5.67 * 10**-8
T_ref = 273.15

materials = pd.DataFrame(columns=["material","lambda","rho","c"])

materials = pd.concat([materials, pd.DataFrame({"material":'concrete_bloc_19cm' ,"lambda":1.36,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'concrete_bloc_14cm' ,"lambda":1.27,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'insulation'         ,"lambda":0.026, "rho":80,   "c":840 }, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'reinforced_concrete',"lambda":2.30,  "rho":2400, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'concrete'           ,"lambda":2.10,  "rho":2400, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'brick'              ,"lambda":1.59,  "rho":2200, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'screed'             ,"lambda":1.40,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'foamglass'          ,"lambda":0.055, "rho":150,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'wood'               ,"lambda":0.13,  "rho":600,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'glass'              ,"lambda":1.00,  "rho":2500, "c":840 }, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'Air_layer_60mm'     ,"lambda":0.35,  "rho":1.2,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'Air_layer_275mm'    ,"lambda":1.62,  "rho":1.2,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'Waterproof_membrane',"lambda":0.230, "rho":1300, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'Gravels'            ,"lambda":2.0,   "rho":2000, "c":1000}, index=[0])])

materials = materials.reset_index(drop=True)


def prepro_w(weather, month, phi_deg, lambda_deg):
    
    if weather == 1:
        w = pd.read_excel('hot_wave.xlsx')
        w['hour_sol'] = w['hour_yr'] - lambda_deg/15 - 2 - 0.5
    elif weather == -1:
        w = pd.read_excel('cold_wave.xlsx')
        w['hour_sol'] = w['hour_yr'] - lambda_deg/15 - 2 + 1.5
    else:
        w = pd.read_excel('weather_data_uccle_meteonorm.xlsx')
        w['hour_sol'] = w['hour_yr'] - lambda_deg/15 - 2 + 0.5
        
    w = w.iloc[1:].apply(pd.to_numeric)
    w = w.rename(columns={"T_out": "t_out", "RH": "rh_out", "I_gl": "I_th", "I_diff": "I_dh"})
    w['I_bh'] = (w['I_th'] - w['I_dh']).where((w['I_th'] > w['I_dh']) & (w['I_th'] > 0.0001), 0 * w['I_th'])
    column_to_move = w.pop('I_bh')
    w.insert(w.columns.get_loc("I_dh"), "I_bh", column_to_move)
    
    w['h_out'] = 5.8 + 3.94 * w['wind_speed']
    
    w['numday'] =  w['hour_yr']/24 -  7 * (w['hour_yr']/24/7).astype(int)
    w['intday'] = (w['hour_yr']/24).astype(int)
    
    wpred       = w.copy()
    wpred['t_out_p'] = wpred['t_out']
    wpred       = wpred.groupby("intday").agg({'t_out':'mean','t_out_p':'max'})
    wpred       = wpred[['t_out', 't_out_p']]
    wpred       = wpred.rename(columns={"t_out": "t_out_mean_day", "t_out_p": "t_out_pred_day"})
    T_out_mean  = wpred['t_out_mean_day'].rolling(12, min_periods=1).mean().values
    T_out_pred  = wpred['t_out_pred_day'].rolling(12, min_periods=1).mean().values
    
    wpred["t_out_mean"]  = T_out_mean
    wpred["t_out_pred"]  = T_out_pred
    wpred["t_out_pred"]  = np.roll(T_out_pred, - 12) # predicted temperature
    wpred                = wpred[['t_out_mean', "t_out_pred"]]
    
    w           = pd.merge(w, wpred, left_on='intday'  , right_index=True, how= 'outer')
    w           = w.reset_index(drop=True)
    
    hour_sol_local, gamma_s_deg, theta_z_deg = sun_location(phi_deg, lambda_deg, w['hour_sol'].values)
    w['theta_z_deg'] = theta_z_deg
    w['gamma_s_deg'] = gamma_s_deg
    
    I_ir_h  = w['I_ir_h'].values
    clearness_sky = (- I_ir_h - 45) / 55
    w['c_sky'] = np.clip(1 - clearness_sky, 0, 1)
    
    h_start_month =                 24 * np.array([ 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]).cumsum() + 1
    h_stop_month  = h_start_month + 24 * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) - 1 
    
    if (weather == 0) and (month > 0) :
        w = w.loc[(w['hour_yr'] >= h_start_month[month]) & (w['hour_yr'] <= h_stop_month[month])]
        w = w.reset_index(drop=True)
    
    return w


def prepro_wd(windows, f_fr, f_LT, g_gl, g_bl, rho_env, epsilon_ir, w, hourly):
    
    zname_list  = list(set(pd.DataFrame(hourly , columns=('zone', 'f_on_h')) ['zone']))
    
    wd = pd.DataFrame(windows, columns=('zone', 'azimuth', 'number', 'breadth', 'height', \
                                    'As_deg', 'At_deg', 'Af_deg', 'D_H_f'))

    wd['wall_type'] = 'wall_ext'
    wd['slope']     = 90
    wd['area_wd']   = wd['number'] * wd['breadth'] * wd['height']

#   Multilevel Dataframes
    Qwdsol = {}
    Qwdir  = {}
    Itwd   = {}
    Ibwd   = {}


    for zname in zname_list:
        Qwdsol[str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)
        Qwdir [str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)
        Itwd  [str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)
        Ibwd  [str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)

    for i in range(0, len(wd)):
        zname = wd.iloc[i]['zone']
        az    = wd.iloc[i]['azimuth']
        Qsol_wd, Qir_wd, I_t_wd, I_b_wd = Qsolirwd(i, wd, f_fr, f_LT, g_gl, g_bl, rho_env, epsilon_ir, w)
        Qwdsol[str(zname)] = pd.concat([Qwdsol[str(zname)], \
                                        pd.DataFrame(columns=[str(az)], data = Qsol_wd)]).groupby(level=0).sum()
        Qwdir [str(zname)] = pd.concat([Qwdir [str(zname)], \
                                        pd.DataFrame(columns=[str(az)], data = Qir_wd )]).groupby(level=0).sum()
        Itwd  [str(zname)] = pd.concat([Itwd[str(zname)], \
                                        pd.DataFrame(columns=[str(az)], data = I_t_wd )]).groupby(level=0).max()
        Ibwd  [str(zname)] = pd.concat([Ibwd[str(zname)], \
                                        pd.DataFrame(columns=[str(az)], data = I_b_wd )]).groupby(level=0).max()
    
    return wd, Qwdsol, Qwdir, Itwd, Ibwd


def prepro_wl(walls, rho_env, epsilon_ir, EXTwalls, w, wd, hourly):
    
    zname_list  = list(set(pd.DataFrame(hourly , columns=('zone', 'f_on_h')) ['zone']))

    wl = pd.DataFrame(walls, columns=('zone', 'azimuth', 'wall_type', 'dim1', 'dim2', 'f_h_rad', 'f_c_rad'))

    wl['gross_area'] = wl['dim1'] * wl['dim2']
    wl = wl.drop(columns = ['dim1', 'dim2']).copy()
    wl['slope'] = 90
    wl.loc[wl['wall_type'] == 'floor', 'slope']   = 0
    wl.loc[wl['wall_type'] == 'ceiling', 'slope'] = 180

    wl = wl.groupby(by=['zone', 'azimuth', 'wall_type', 'slope'],as_index = False).sum()
    wd = wd.groupby(by=['zone', 'azimuth', 'wall_type'],as_index = False).agg({'area_wd':'sum'})
    wl = pd.merge(wl, wd, how="outer")
    wl = wl.fillna(0)

    wl['area_wl'] =  wl['gross_area'] - wl['area_wd'] 
    wl['area_wl'] =  wl['area_wl'].where(wl['area_wl'] >= 0, 0)
    wl            =  wl.drop(columns=['gross_area'])

    fh_rad_z = wl[['zone', 'f_h_rad']].groupby(by=['zone'],as_index = False).sum()
    fc_rad_z = wl[['zone', 'f_c_rad']].groupby(by=['zone'],as_index = False).sum()
    
    wle      = wl.loc[wl['wall_type'].isin(EXTwalls)].copy()
    
    lst_wt = ['wall_ext', 'wall_uncond', 'wall_cond', 'floor', 'ceiling']

    #   Multilevel Dataframes
    Qwlsolir = {}

    for zname in zname_list:
        Qwlsolir[str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)
        
    for i in range(0, len(wle)):
        zname  = wle.iloc[i]['zone']
        wtname = wle.iloc[i]['wall_type']
        Q_wl  = Qsolirwl(i, wle, rho_env, epsilon_ir, w)
        Qwlsolir[str(zname)] = pd.concat([Qwlsolir[str(zname)], \
                                        pd.DataFrame(columns=[str(wtname)], data = Q_wl)]).groupby(level=0).sum()
    
    return wl, fh_rad_z, fc_rad_z, Qwlsolir


def prepro_occ_data(w, hourly, daily, occupancy, vent, e_leak, n_50):
    
    hour = w['hour_yr'].values

    dfh = pd.DataFrame(hourly , columns=('zone', 'f_on_h'))
    dfd = pd.DataFrame(daily  , columns=('zone', 'f_on_d'))

    dfo = pd.DataFrame(occupancy, columns=('zone', 'occ_profile', 'n_occ', 'q_r_Wocc', 'q_c_Wocc', 'q_c_appl_Wocc', \
                                              'q_r_light_Wm2', 'q_c_light_Wm2', 'q_h_Wm2', 'q_c_Wm2'))


    dfv = pd.DataFrame(vent, columns=('zone', 'n_fl', 'area_fl', 'vol_int',  \
                              'q_su_m3h', 'q_ex_m3h', 'epsilon_rec', 'n_fc' ))

    dfv['q_leak_m3h'] = e_leak * n_50  * dfv['vol_int']
    dfv['q_fc_m3h']   = dfv['n_fc'] * dfv['vol_int']


    dfv  = dfv[['zone', 'n_fl', 'area_fl', 'vol_int',  \
                'q_su_m3h', 'q_ex_m3h', 'epsilon_rec', 'q_fc_m3h', 'q_leak_m3h']]


    dfQM = pd.merge(dfv[['zone', 'area_fl']], dfo, on = 'zone', how='left')

    dfQM['Qorad']    = dfQM['n_occ']   *  dfQM['q_r_Wocc'] 
    dfQM['Qoconv']   = dfQM['n_occ']   * (dfQM['q_c_Wocc'] + dfQM['q_c_appl_Wocc']) 
    dfQM['Qlrad']    = dfQM['area_fl'] *  dfQM['q_r_light_Wm2']
    dfQM['Qlconv']   = dfQM['area_fl'] *  dfQM['q_c_light_Wm2']
    dfQM['Qheating'] = dfQM['area_fl'] *  dfQM['q_h_Wm2']
    dfQM['Qcooling'] = dfQM['area_fl'] *  dfQM['q_c_Wm2']


    dfsortz = dfh[['zone']]
    dfd  = pd.merge(dfsortz, dfd,  on = 'zone', how='left')
    dfo  = pd.merge(dfsortz, dfo,  on = 'zone', how='left')
    dfv  = pd.merge(dfsortz, dfv,  on = 'zone', how='left')
    dfQM = pd.merge(dfsortz, dfQM, on = 'zone', how='left').set_index('zone')
    
    return dfh, dfd, dfo, dfv, dfQM


def schedules(zname, dfh, dfd, dfo, hour):
    
    f_on_h = list(dfh.loc[dfh['zone'] == zname]['f_on_h'].values[0])
    f_on_h = np.tile(f_on_h, int(len(hour)/24))

    f_on_d = list(dfd.loc[dfd['zone'] == zname]['f_on_d'].values[0])
    f_on_d = np.repeat(f_on_d, 24)
    f_on_d = np.tile(f_on_d, int(len(hour)/24/7) + 1)
    f_on_d = f_on_d[0:len(hour)]

    f_on   = f_on_h * f_on_d

    f_oc_24h = list(dfo.loc[dfo['zone'] == zname]['occ_profile'].values[0])
    f_oc     = np.tile(f_oc_24h, int(len(hour)/24)) * f_on_d

    flag_occupancy = np.where(f_oc > 0, 1, 0)

    return f_on, f_oc, flag_occupancy


def prepro_hbh_profile(w, dfh, dfd, dfo, dfQM, OCC_light_control, t_set_heating, t_set_min, t_set_cooling, t_set_max): 
    
    hour = w['hour_yr'].values
    
    # Hour by hour profiles
    
    #   Multilevel Dataframes
    hbh_profiles = {}

    for i in range(0, len(dfo)):

        zname = dfo.iloc[i]['zone']

        f_on, f_oc, flag_occupancy = schedules(zname, dfh, dfd, dfo, hour)
        
        f_light  = f_oc if OCC_light_control else flag_occupancy
        t_set_h  = t_set_heating * f_on + t_set_min * (1-f_on)
        t_set_c  = t_set_cooling * f_on + t_set_max * (1-f_on)
        
        Qroccup  = f_oc    * dfQM.iloc[i]['Qorad'] 
        Qcoccup  = f_oc    * dfQM.iloc[i]['Qoconv']
        Qrlight  = f_light * dfQM.iloc[i]['Qlrad'] 
        Qclight  = f_light * dfQM.iloc[i]['Qlconv']
        
        hbh_profiles[str(zname)] = pd.DataFrame(columns=['hour_yr'], data = w['hour_yr'].values)
        
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['tseth'], data = t_set_h)]).groupby(level=0).sum()
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['tsetc'], data = t_set_c)]).groupby(level=0).sum()
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qro'],   data = Qroccup)]).groupby(level=0).sum()
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qco'],   data = Qcoccup)]).groupby(level=0).sum()
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qrl'],   data = Qrlight)]).groupby(level=0).sum()
        hbh_profiles[str(zname)] = pd.concat([hbh_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qcl'],   data = Qclight)]).groupby(level=0).sum()

    return hbh_profiles


def prepro_sbs_profile(w, dfh, dfd, dfo, dfQM, dfv, fh_rad_z, fc_rad_z, CO2_vent_control, h_restart_rad, time_step): 
    
    hour  = w['hour_yr'].values
    h_min = w['hour_yr'].min()
    h_max = w['hour_yr'].max()
    wstep = pd.DataFrame(data={'hour_yr': np.arange(h_min, h_max + 1, time_step/3600 )})
    
    nbstph = int(3600 / time_step)

    # Step by step profiles
    
    #   Multilevel Dataframes
    sbs_profiles = {}

    for i in range(0, len(dfo)):

        zname = dfo.iloc[i]['zone']

        f_on, f_oc, flag_occupancy = schedules(zname, dfh, dfd, dfo, hour)
        
        f_vs = f_oc if CO2_vent_control else flag_occupancy

        # By time step
        f_on   = np.repeat(f_on, nbstph)
        f_vs   = np.repeat(f_vs, nbstph)

        # Convective system always active with a set point varying as function of f_on
        f_conv = np.ones(len(f_on))

        # radiative systems working before the start of the convective system
        nbstrh = int(h_restart_rad * nbstph)
        f_rad  = f_on
        for step in range( nbstrh ):
            f_rad  = np.roll(f_rad, -1)
        f_rad = np.clip(f_rad - f_on, 0, 1)

        # progressive restart of the ventilation system
        f_resartv = np.maximum(np.diff(f_vs, prepend=0), np.zeros(len(f_vs)))
        f_resartv = np.roll(f_resartv, - nbstph)
        for n in range(nbstph) :
            f_vs = np.maximum(f_vs, n/nbstph * np.roll(f_resartv, n)) 

        fh_rad = min(1, fh_rad_z.loc[fh_rad_z['zone'] == zname]['f_h_rad'].values[0])
        fc_rad = min(1, fc_rad_z.loc[fc_rad_z['zone'] == zname]['f_c_rad'].values[0])

        Qheatmr  = fh_rad     * f_rad  * dfQM.iloc[i]['Qheating']
        Qcoolmr  = fc_rad     * f_rad  * dfQM.iloc[i]['Qcooling']
        Qheatmc  = (1-fh_rad) * f_conv * dfQM.iloc[i]['Qheating']
        Qcoolmc  = (1-fc_rad) * f_conv * dfQM.iloc[i]['Qcooling']
        q_su_m3h =              f_vs   * dfv .iloc[i]['q_su_m3h']
        q_ex_m3h =              f_vs   * dfv .iloc[i]['q_ex_m3h']
        
        sbs_profiles[str(zname)] = pd.DataFrame(columns=['hour_yr'], data = wstep['hour_yr'].values)
        
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qhr'], data = Qheatmr)]).groupby(level=0).sum()
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qcr'], data = Qcoolmr)]).groupby(level=0).sum()
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qhc'], data = Qheatmc)]).groupby(level=0).sum()
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['Qcc'], data = Qcoolmc)]).groupby(level=0).sum()
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['qsu'], data = q_su_m3h)]).groupby(level=0).sum()
        sbs_profiles[str(zname)] = pd.concat([sbs_profiles[str(zname)] , \
                                   pd.DataFrame(columns=['qex'], data = q_ex_m3h)]).groupby(level=0).sum()

    return sbs_profiles


def mij_table(dfv, wl, epsilon_ir):
    
    mijrad = pd.DataFrame(columns=["zone", "mij"])
    Airad  = pd.DataFrame(columns=["zone", "Ai"])

    for i in range(0, len(dfv)):
        zname    = dfv.iloc[i]['zone']
        zmij, Ai =  zone_mij(zname, i, dfv, wl, epsilon_ir)
        mijrad = mijrad.append({"zone":zname ,"mij":zmij}, ignore_index=True)
        Airad  = Airad .append({"zone":zname ,"Ai":Ai  }, ignore_index=True)

    mij = pd.DataFrame.from_dict([mijrad.set_index('zone').to_dict()['mij']])
    Ai  = pd.DataFrame.from_dict([Airad .set_index('zone').to_dict()['Ai']])
    
    return mij, Ai


def azimuth_norm(az):
    # azimuth comprised between -360° and 360°
    az = az - 360 * np.trunc(az/360)
    # azimuth comprised between -180° and 180°"
    if (abs(az) > 180) :
        if (az > 180) : 
            az = az - 360
        else:
            az = az + 360
    return az


def sun_location(phi_deg, lambda_deg, hour_sol_local_0):

    n_days_year =  365
    lambda_h    = lambda_deg/15

    sin_phi = sin(phi_deg * pi/180)
    cos_phi = cos(phi_deg * pi/180)
    tan_phi = tan(phi_deg * pi/180)

    betaJ = (2*pi/n_days_year)*(hour_sol_local_0/24)
    ET =  (1/60)*(-0.00037+0.43177*np.cos(betaJ)-3.165*np.cos(2*betaJ)-0.07272*np.cos(3*betaJ)\
                  -7.3764*np.sin(betaJ)-9.3893*np.sin(2*betaJ)-0.24498*np.sin(3*betaJ))

    hour_sol_local = hour_sol_local_0 + ET
    hour_sol_local_per = hour_sol_local-24*np.trunc(hour_sol_local/24)
    hour_sol_local_per[hour_sol_local_per == 0] = 24

#     time_rad = 2*pi*hour_sol_local/(24*365)
#     cos_time=np.cos(time_rad)

    # hour_south_per = heure périodique égale à 0h quand le soleil est au Sud (azimut gamma = 0)
    hour_south_per = hour_sol_local_per - 12

    # Angle horaire omega en degres : omega = 0 quand le soleil est au Sud (azimut gamma = 0)
    omega_deg = hour_south_per*15   
    sin_omega = np.sin(omega_deg * pi/180)
    cos_omega = np.cos(omega_deg * pi/180)

    # Sun declination delta en degres
    time_rad=2*pi*hour_sol_local/(24*n_days_year)
    time_lag_rad = 2*pi*(284/n_days_year)
    sin_time_decl = np.sin(time_rad+time_lag_rad)
    delta_rad=0.40928*sin_time_decl
    delta_deg=(180/pi)*delta_rad

    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    tan_delta = np.tan(delta_rad)

    # Angle theta_z between sun beams and vertical"
    theta_z_rad = np.abs(np.arccos(sin_delta*sin_phi+cos_delta*cos_phi*cos_omega))
    cos_theta_z= np.cos(theta_z_rad)
    sin_theta_z= np.sin(theta_z_rad)
    theta_z_deg= (180/pi)*theta_z_rad

    # Compute gamma_s : Sun azimuth "
    # Azimut value comprised between -pi and +pi
    gamma_s_rad = np.arctan2(sin_omega, cos_omega * sin_phi - tan_delta * cos_phi)

    sin_gamma_s = np.sin(gamma_s_rad)
    cos_gamma_s = np.cos(gamma_s_rad)
    # Azimuth value comprised between -180 and +180
    gamma_s_deg = (180/pi)*gamma_s_rad 

    # Components of the unit vector parallel to sun  beams in axes: South, East, Vertical
    n_sun_beam_South = cos_gamma_s*sin_theta_z
    n_sun_beam_East = sin_gamma_s*sin_theta_z
    n_sun_beam_Vert = cos_theta_z

#     h_s_deg = np.where(theta_z_deg < 90, 90 - theta_z_deg, 0)
#     h_s_rad = (pi/180) * h_s_deg
    
    return hour_sol_local, gamma_s_deg, theta_z_deg


def Qsolirwd(i, wd, f_fr, f_LT, g_gl, g_bl, rho_env, epsilon_ir, w):
    
    t_out  = w['t_out'].values
    I_bh   = w['I_bh'].values
    I_dh   = w['I_dh'].values
    I_th   = w['I_th'].values
    I_ir_h = w['I_ir_h'].values
    c_sky  = w['c_sky'].values
    theta_z_deg = w['theta_z_deg'].values
    gamma_s_deg = w['gamma_s_deg'].values
    
    slope_wd_deg   = wd.iloc[i]['slope']
    azimuth_wd_deg = wd.iloc[i]['azimuth']
    number_wd  = wd.iloc[i]['number']
    Breadth_wd = wd.iloc[i]['breadth']
    Height_wd  = wd.iloc[i]['height']
    As         = wd.iloc[i]['As_deg']
    At         = wd.iloc[i]['At_deg']
    Af         = wd.iloc[i]['Af_deg']
    Df_H       = wd.iloc[i]['D_H_f']
    An         = 0
    Dn_H       = 0
    Ap         = 0
    Dp_H       = 0
    
    p_wd_deg = abs(slope_wd_deg - 180)
    p_wd_rad = p_wd_deg*pi/180
    cos_p    = cos(p_wd_rad)
    sin_p    = sin(p_wd_rad)

    area_wd    = Height_wd*Breadth_wd
    area_gl    = area_wd * (1 - f_fr)
 
    angle_setback_screens_deg  = As
    angle_top_screen_deg       = At
    angle_front_screen_deg     = min(Af,85)
    dist_front_screen          = Df_H * Height_wd
    angle_neg_screen_deg       = min(An,85)
    dist_neg_screen            = Dn_H * Height_wd
    angle_pos_screen_deg       = min(Ap,85)
    dist_pos_screen            = Dp_H * Height_wd
    
    if dist_front_screen > 0: 
        iflag_front_screen= 1 
    else:
        iflag_front_screen= 0 
    
    gamma_w_deg = azimuth_wd_deg
    gamma_w_rad = gamma_w_deg*pi/180
    sin_gamma_w = sin(gamma_w_rad)
    cos_gamma_w = cos(gamma_w_rad)

    deltagamma_deg = np.where(abs(gamma_s_deg - gamma_w_deg) > 180, abs(abs(gamma_s_deg - gamma_w_deg) - 360),
                            abs(gamma_s_deg - gamma_w_deg))
    
    Ah_deg = min(angle_top_screen_deg + angle_front_screen_deg, 90)
    Ah_rad = pi * Ah_deg / 180
    cos_Ah = np.cos(Ah_rad)
    Av_deg = min(max(angle_setback_screens_deg, angle_neg_screen_deg) \
               + max(angle_setback_screens_deg, angle_pos_screen_deg), 180)

    f_shading = np.zeros(len(gamma_s_deg))
    I_t_wd    = np.zeros(len(gamma_s_deg))
    I_b_wd    = np.zeros(len(gamma_s_deg))
    Q_s_wd    = np.zeros(len(gamma_s_deg))
    Q_dr_wd   = np.zeros(len(gamma_s_deg))
    Q_ir_wd   = np.zeros(len(gamma_s_deg))
    
    dt_sky_cs = -21
    h_r = 5 # W/m²K
    
    for i in range(len(gamma_s_deg)):
        
        # Extra IR heat losses due to the temperature difference between the sky and the air
        Q_ir_wd[i] = number_wd * area_gl * cos_Ah * (1 - Av_deg/180) * (1 - c_sky[i]) * epsilon_ir * h_r * dt_sky_cs * (1 + cos_p) / 2
        
        if ((theta_z_deg[i] < 88) & (Height_wd > 0.001) & (deltagamma_deg[i] < 90)):
            
            deltagamma_rad = deltagamma_deg[i] * pi/180
            cos_dgamma = np.cos(deltagamma_rad)
            sin_dgamma = np.sin(deltagamma_rad)
            tan_dgamma = np.tan(deltagamma_rad)
            
            theta_z_rad = theta_z_deg[i] * pi/180
            cos_theta_z = np.cos(theta_z_rad)
            sin_theta_z = np.sin(theta_z_rad)
            tan_theta_z = np.tan(theta_z_rad)

            # Compute ratio= cos(theta)/cos(theta_z)
            # Cos of angle theta between sun beams and normal direction to the wall
            # Mask effect if sun flicking the wall with an angle < 2°
#             cos_theta = cos_p * cos_theta_z + sin_p * sin_theta_z * cos_dgamma
            cos_theta = sin_theta_z * cos_dgamma

            # setback vertical screens supposed symetrical: horizontal angle measured from the center of the window
            if (angle_setback_screens_deg > 0):
                tan_As                    = tan(angle_setback_screens_deg*pi/180)
                Depth_setback_screen      = tan_As*Breadth_wd/2
                b_wd_shade_setback_screen = Depth_setback_screen*abs(tan_dgamma) if(cos_dgamma> 0) else 0
            else:
                b_wd_shade_setback_screen = 0

            # Horizontal screen upside the window: vertical angle measured from the center of the window
            if (angle_top_screen_deg > 0):
                tan_At                    = tan(angle_top_screen_deg*pi/180)
                Depth_top_screen          = tan_At*Height_wd/2
                h_wd_shade_top_screen     = Depth_top_screen/(tan_theta_z*cos_dgamma) \
                                            if ((tan_theta_z*cos_dgamma > 0.001) & (cos_dgamma>0)) else 0
            else:
                h_wd_shade_top_screen     = 0
                
            # Vertical screen facing the window: vertical angle measured from the center of the window
            if (dist_front_screen > 0):
                Hypoth_front_screen       = dist_front_screen/cos_dgamma if(cos_dgamma > 0.001) else 0
                h_front_screen_no_shade   = Hypoth_front_screen/tan_theta_z if(tan_theta_z > 0.001) else 0
                tan_Af                    = tan(angle_front_screen_deg*pi/180)
                h_front_screen            = Height_wd/2 + dist_front_screen * tan_Af
                h_front_screen_shade      = h_front_screen - h_front_screen_no_shade if(h_front_screen > h_front_screen_no_shade) else 0
                h_wd_shade_front_screen   = h_front_screen_shade if(h_front_screen_no_shade > 0) else 0
            else:
                h_wd_shade_front_screen   = 0
            
            # Vertical negative azimuth side lateral screen: vertical angle measured from the center of the windowed area
            if (dist_neg_screen > 0):
                Hypoth_neg_screen         = dist_neg_screen/sin_dgamma if(sin_dgamma > 0.001) else 0
                h_neg_screen_no_shade     = Hypoth_neg_screen/tan_theta_z if(tan_theta_z > 0.001) else 0
                tan_An                    = tan(angle_neg_screen_deg*pi/180)
                h_neg_screen              = Height_wd/2 + dist_neg_screen * tan_An
                h_neg_screen_shade        = h_neg_screen - h_neg_screen_no_shade if(h_neg_screen > h_neg_screen_no_shade) else 0
                h_wd_shade_neg_screen     = h_front_neg_shade if(h_front_neg_no_shade > 0) else 0
            else:
                h_wd_shade_neg_screen   = 0
                
            # Vertical positive azimuth side lateral screen: vertical angle measured from the center of the windowed area
            if (dist_pos_screen > 0):
                Hypoth_pos_screen         = dist_pos_screen/sin_dgamma if(sin_dgamma > 0.001) else 0
                h_pos_screen_no_shade     = Hypoth_pos_screen/tan_theta_z if(tan_theta_z > 0.001) else 0
                tan_An                    = tan(angle_pos_screen_deg*pi/180)
                h_pos_screen              = Height_wd/2 + dist_pos_screen * tan_An
                h_pos_screen_shade        = h_pos_screen - h_pos_screen_no_shade if(h_pos_screen > h_pos_screen_no_shade) else 0
                h_wd_shade_pos_screen     = h_front_pos_shade if(h_front_pos_no_shade > 0) else 0
            else:
                h_wd_shade_pos_screen   = 0


            # Shading factor
            h_wd_shade = h_wd_shade_top_screen + max(h_wd_shade_front_screen, h_wd_shade_neg_screen, h_wd_shade_pos_screen)
            dh_shade   = Height_wd - h_wd_shade if (Height_wd > h_wd_shade) else 0
            b_wd_shade = b_wd_shade_setback_screen
            db_shade   = Breadth_wd - b_wd_shade if (Breadth_wd>b_wd_shade) else 0
            area_no_shaded_wd = dh_shade * db_shade
            f_no_shaded_wd    = area_no_shaded_wd / area_wd
            f_shading[i]  = (1-f_no_shaded_wd)
            
            # Irradiation
            I_beam     = I_bh[i] / cos_theta_z  if (I_bh[i] > 0) else 0
            I_b_wd[i]  = cos_theta * I_beam if (cos_theta > 0) else 0
            Q_s_wd[i]  = number_wd * area_gl * g_gl * (1-f_shading[i]) * I_b_wd[i]
            
            I_dr       = cos_Ah * (1 - Av_deg/180) * (I_dh[i] * (1 + cos_p) / 2 +  rho_env * I_th[i] * (1 - cos_p) / 2)
            Q_dr_wd[i] = number_wd * area_gl * g_gl * I_dr
            
            I_t_wd[i]  = I_b_wd[i] + I_dr 
             
    return Q_s_wd + Q_dr_wd, Q_ir_wd, I_t_wd, I_b_wd


def Qsolirwl(i, wl, rho_env, epsilon_ir, w):
    
    t_out  = w['t_out'].values
    I_bh   = w['I_bh'].values
    I_dh   = w['I_dh'].values
    I_th   = w['I_th'].values
    I_ir_h = w['I_ir_h'].values
    c_sky  = w['c_sky'].values
    theta_z_deg = w['theta_z_deg'].values
    gamma_s_deg = w['gamma_s_deg'].values
    
    slope_wl_deg   = wl.iloc[i]['slope']
    p_wl_deg = abs(slope_wl_deg - 180)
    p_wl_rad = p_wl_deg*pi/180
    cos_p    = cos(p_wl_rad)
    sin_p    = sin(p_wl_rad)
    
    alpha_wl = 0.6 if slope_wl_deg < 135 else 0.9
   
    azimuth_wl_deg = wl.iloc[i]['azimuth']
    area_wl = wl.iloc[i][['area_wl']][0]
    
    gamma_w_deg = azimuth_wl_deg
    gamma_w_rad = gamma_w_deg*pi/180
    sin_gamma_w = sin(gamma_w_rad)
    cos_gamma_w = cos(gamma_w_rad)

    deltagamma_deg = np.where(abs(gamma_s_deg - gamma_w_deg) > 180, abs(abs(gamma_s_deg - gamma_w_deg) - 360),
                            abs(gamma_s_deg - gamma_w_deg))
    
    Q_s_wl    = np.zeros(len(gamma_s_deg))
    Q_dr_wl   = np.zeros(len(gamma_s_deg))
    Q_ir_wl   = np.zeros(len(gamma_s_deg))
    
    dt_sky_cs = -21
    h_r = 5 # W/m²K
    
    for i in range(len(gamma_s_deg)):
        
        # Extra IR heat losses due to the temperature difference between the sky and the air
        Q_ir_wl[i] =  area_wl * (1 - c_sky[i]) * epsilon_ir * h_r * dt_sky_cs * (1 + cos_p) / 2
        
        if ((theta_z_deg[i] < 88) & (area_wl > 0.001) & (deltagamma_deg[i] < 90)):
            
            deltagamma_rad = deltagamma_deg[i] * pi/180
            cos_dgamma = np.cos(deltagamma_rad)
            sin_dgamma = np.sin(deltagamma_rad)
            tan_dgamma = np.tan(deltagamma_rad)
            
            theta_z_rad = theta_z_deg[i] * pi/180
            cos_theta_z = np.cos(theta_z_rad)
            sin_theta_z = np.sin(theta_z_rad)
            tan_theta_z = np.tan(theta_z_rad)

            # Compute ratio= cos(theta)/cos(theta_z)
            # Cos of angle theta between sun beams and normal direction to the wall
            # Mask effect if sun flicking the wall with an angle < 2°
            cos_theta = cos_dgamma * sin_theta_z

            # Irradiation
            I_beam = I_bh[i] / cos_theta_z  if (I_bh[i] > 0) else 0
            I_b_wl = cos_theta * I_beam if (cos_theta > 0) else 0
            Q_s_wl[i] = area_wl * alpha_wl * I_b_wl
    
            I_dr       =  I_dh[i] * (1 + cos_p) / 2 +  rho_env * I_th[i] * (1 - cos_p) / 2
            Q_dr_wl[i] =  area_wl * alpha_wl * I_dr
            
        
    return Q_s_wl + Q_dr_wl + Q_ir_wl


def mysumdown(series):
    mysum = 0
    stop = 0
    for s in series:
        if s==0: 
            stop = 1
            break
        else:
            mysum = mysum + s
    if stop == 0:
        mysum = mysum/2
    return mysum

def mysumup(series):
    mysum = 0
    stop = 0
    for s in series[::-1]:
        if s==0:
            stop = 1
            break
        else:
            mysum = mysum + s
    if stop == 0:
        mysum = mysum/2
    return mysum


# Parallel circular plates aligned, same size
def VF_h_to_h(r,h) :
    if r > 0: 
        VF = 0.5 * ( h**2 + 2*r**2 - h * (h**2 + 4 * r**2)**0.5 ) / r**2
    else :
        VF = 0
    return VF

# Parallel rectangular plates aligned, same size
def VF_para_aligned(X, Y, L) :
    x = X/L
    y = Y/L
    if (x > 0) and (y > 0) :
        VF1 = log( ( (1+x**2)*(1+y**2) / (1 + x**2 + y**2) )**0.5) 
        VF2 = x * (1+y**2)**0.5 * atan(x / (1+y**2)**0.5)
        VF3 = y * (1+x**2)**0.5 * atan(y / (1+x**2)**0.5)
        VF = 2 / (pi * x * y) * (VF1 + VF2 + VF3 - x * atan(x) - y * atan(y))
    else :
        VF = 0
    return VF

def BB(x,y,eta,xi,z):
    BB = ((y-eta)*sqrt((x-xi)**2+z**2)*atan((y-eta)/sqrt((x-xi)**2+z**2)) \
          +(x-xi)*sqrt((y-eta)**2+z**2)*atan((x-xi)/sqrt((y-eta)**2+z**2))-z**2/2*log((x-xi)**2+(y-eta)**2+z**2))
    return BB

# This VF is checked and it is OK
# a1, b1, a2, b2 : half breadths (along x) and half lenghts (along y) of surfaces 1 and 2
# a0, b0 : coordinates of the center of surface 1 projected on surface 2, refering to surface 2 center
def VF_para(a1, b1, a2, b2, a0, b0, d):
        if d != 0 :
            z = d
            x = np.array([a0 - a1, a0 + a1])
            y = np.array([b0 - b1, b0 + b1])
            xi = np.array([- a2, a2])
            eta = np.array([- b2, b2])
            F = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            F = F + ((-1)**(4+i+j+k+l)) * BB(x[i],y[j],eta[k],xi[l],z)
            A=(x[1]-x[0])*(y[1]-y[0])
            VF = F / (2 * pi * A)

        else:
            VF =0
        return VF


# View factor perpendicular plates sharing a common edge from 1 to 2, without offsets
def VF_per(a,b,c):
    if a != 0 and b != 0 and c != 0 :
        H=b/c
        W=a/c
        A=((1+W**2)*(1+H**2))/(1+W**2+H**2)
        B=((W**2*(1+W**2+H**2))/((1+W**2)*(W**2+H**2)))**(W**2)
        C=((H**2*(1+W**2+H**2))/((1+H**2)*(W**2+H**2)))**(H**2)
        VF=1/(pi*W)*(W*atan(1/W)+H*atan(1/H)-sqrt(H**2+W**2)*atan(1/sqrt(H**2+W**2))+ (1/4)* log(A*B*C))
    else:
        VF =0
    return VF

# View factor between two not touching perpendicular rectangles from 1 to 2
def VF_perp_nt(a1, a2, b1, b2, c1, c2, c3):
# a1, a2, b1, b2, c1, c2, and  c3:  dimensions as shown in the figure.
# Martinez I., Radiation View Factors, 1995, p.27
    BB012345=VF_per((a1+a2),(b1+b2),(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    BB1234=VF_per((a1+a2),(b1+b2),(c1+c2))*(a1+a2)*(c1+c2)
    BB0145=VF_per((a1+a2),(b1+b2),(c2+c3))*(a1+a2)*(c2+c3)
    BB345=VF_per(a1,b1,(c1+c2+c3))*a1*(c1+c2+c3)
    B345_012345=VF_per(a1,(b1+b2),(c1+c2+c3))*a1*(c1+c2+c3)
    B012345_345=VF_per((a1+a2),b1,(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    B0145_45=VF_per((a1+a2),b1,(c2+c3))*(a1+a2)*(c2+c3)
    B1234_34=VF_per((a1+a2),b1,(c1+c2))*(a1+a2)*(c1+c2)
    B45_0145=VF_per(a1,(b1+b2),(c2+c3))*a1*(c2+c3)
    B34_1234=VF_per(a1,(b1+b2),(c1+c2))*a1*(c1+c2)
    BB14=VF_per((a1+a2),(b1+b2),c2)*(a1+a2)*c2
    B14_4=VF_per((a1+a2),b1,c2)*(a1+a2)*c2
    BB45=VF_per(a1,b1,(c2+c3))*a1*(c2+c3)
    BB34=VF_per(a1,b1,(c1+c2))*a1*(c1+c2)
    B4_14=VF_per(a1,(b1+b2),c2)*a1*c2
    BB4=VF_per(a1,b1,c2)*a1*c2
    VF=(BB012345-BB1234-BB0145+BB345-B345_012345-B012345_345+B0145_45+B1234_34+B45_0145+B34_1234+BB14-B14_4 \
        -BB45-BB34-B4_14+BB4)/(2*a2*c3)
    return VF

# View factor perpendicular plates of different breath along a common edge from 1 to 2, without offsets
def VF_perp_aligned(p, q, r, s) :
    VF1 = VF_per(p, s, 2 * q)
    VF2 = VF_perp_nt(0, p, 0, s, r-q, 0, 2*q)
    VF = VF1 + 2 * VF2
    return VF


# View factor perpendicular plates from 1 to 2, with offsets
def VF_perp_aligned_offsets(a_1,a_2,b_1,b_2,c):
    A_1 = a_2*c
    A_13= (a_1+a_2)*c
    A_3 = a_1*c
    A_2 = b_2*c
    A_24= (b_1+b_2)*c
    A_4 = b_1*c
    if a_1 == 0 and  b_1 == 0:
        VF = VF_per(a_2,b_2,c)
    elif a_1 == 0:
        VF = VF_per(a_2,b_1+b_2,c) - VF_per(a_2,b_1,c)
    elif b_1 == 0:
        VF = (A_13*VF_per(a_1+a_2,b_2,c)-A_3*VF_per(a_1,b_2,c))/A_1
    else:
        VF=(A_13*VF_per(a_1+a_2,b_1+b_2,c)+A_3*VF_per(a_1,b_1,c) \
            -A_3*VF_per(a_1,b_1+b_2,c)-A_13*VF_per(a_1+a_2,b_1,c))/A_1
    return VF
    
    
def add_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy, t[1], t[2], t[3]) for t in walls_types    if t[0] == name_orig]
    return wtl


def add_inverse_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy,t[1],t[2],t[3]) for t in walls_types[::-1] if t[0] == name_orig]
    return wtl


def wall_parameters(walls_types, walls, wl, wd, n_50, U_wd, h_c, h_out, EXTwalls, CONDwalls):

    walls_types = pd.DataFrame(walls_types, columns=('wall_type', 'layer', 'thickness', 'material')) 

    dfwt        = pd.merge(walls_types, materials, on = 'material', how='left')
    dfwt['R']   = dfwt['thickness'] / dfwt['lambda']
    dfwt['C']   = dfwt['thickness'] * dfwt['rho'] * dfwt['c']
    dfwt['M']   = dfwt['thickness'] * dfwt['rho']
    dfwt['Ci']  = dfwt['C'] .where((dfwt['lambda'] >= 0.1) & (dfwt['rho'] >= 2) , 0)
    dfwt['Ce']  = dfwt['Ci'].copy()

    dfwt = dfwt.groupby(by='wall_type',as_index=False).agg({'thickness':'sum','R':'sum','C':'sum','M':'sum', 'Ci':mysumdown,'Ce':mysumup})

    wl = pd.merge(wl, dfwt, on = "wall_type",  how='left')
    
    wl['Aho']   =  wl['area_wl'] * h_out
    wl['Ahm']   =  wl['area_wl'] / wl['R']
    wl['Ahc']   =  wl['area_wl'] * h_c
    wl['ACi']   =  wl['area_wl'] * wl['Ci']
    wl['ACe']   =  wl['area_wl'] * wl['Ce']
    
    R_wd  = U_wd and 1/U_wd or 0
    Rm_wd = max(R_wd - 1/8 - 1/23, 0.003)
    hm_wd = 1/Rm_wd
    
    wl['Aho_wd']  =  wl['area_wd'] * h_out
    wl['Ahm_wd']  =  wl['area_wd'] * hm_wd
    wl['Ahc_wd']  =  wl['area_wd'] * h_c
    
    wl = wl[['zone', 'slope', 'wall_type', 'area_wd', 'area_wl', 'f_h_rad', 'f_c_rad', \
           'Aho', 'Ahm', 'Ahc', 'ACi', 'ACe', 'Aho_wd', 'Ahm_wd', 'Ahc_wd']] \
            .groupby(by=['zone', 'slope', 'wall_type'],as_index=False).sum().copy()
    
    return wl


def zone_mij(zname, dfv, wl, epsilon_ir):
    
    nfl = dfv.loc[dfv['zone'] == zname]['n_fl'].item()
    Afl = dfv.loc[dfv['zone'] == zname]['area_fl'].item()
    Vi  = dfv.loc[dfv['zone'] == zname]['vol_int'].item()
    
    H = Vi / nfl / Afl if Afl * nfl > 0 else 0 

    wltp = wl  [['zone', 'wall_type', 'area_wl']].loc[wl['zone'] == zname].groupby(by=['zone', 'wall_type'],as_index=False).sum().copy()
    wlsl = wl  [['zone', 'slope', 'area_wd', 'area_wl']].loc[wl['zone'] == zname].groupby(by=['zone', 'slope'],as_index=False).sum().copy()
    zar  = wlsl[['zone', 'area_wd', 'area_wl']].groupby(by='zone',as_index=False).sum().copy()
    
    Awd  = zar.iloc[0]['area_wd']
    
    lstwl = set(wltp['wall_type'])

    Awe = wltp.loc[(wltp['zone']==zname) & (wltp['wall_type']=='wall_ext')]   .iloc[0]['area_wl'] if 'wall_ext'    in lstwl else 0
    Awu = wltp.loc[(wltp['zone']==zname) & (wltp['wall_type']=='wall_uncond')].iloc[0]['area_wl'] if 'wall_uncond' in lstwl else 0
    Awc = wltp.loc[(wltp['zone']==zname) & (wltp['wall_type']=='wall_cond')]  .iloc[0]['area_wl'] if 'wall_cond'   in lstwl else 0
    
    Lwd      = H and  Awd              / 4 / H or 0
    Lwdwe    = H and (Awd+Awe)         / 4 / H or 0
    Lwdwewu  = H and (Awd+Awe+Awu)     / 4 / H or 0
    L        = H and (Awd+Awe+Awu+Awc) / 4 / H or 0
    
    VF_h_to_h = VF_para_aligned(L, L, H)
    VF_h_to_v = 1 - VF_h_to_h

    # View Factors
    # 1: windows, 2 : vertical ext walls, 3 : vertical cond walls, 4 : vertical uncond walls, 5 : floor, 6 : ceiling
    
    A1 = 4 *  Lwd * H
    A2 = 4 * (Lwdwe - Lwd) * H
    A3 = 4 * (Lwdwewu - Lwdwe) * H
    A4 = 4 * (L       - Lwdwewu) * H
    A5 = L**2
    A6 = A5
    
    VF_5_to_5 = 0
    VF_6_to_6 = 0
    VF_5_to_6 = VF_h_to_h
    VF_6_to_5 = VF_h_to_h
    
    VF_1_to_1    = VF_para_aligned(Lwd, H, L)                    + 2 * VF_perp_aligned_offsets(L/2-Lwd/2, Lwd, L/2-Lwd/2,     Lwd, H)
    VF_1_to_12   = VF_para(Lwd/2, H/2, Lwdwe/2,    H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwd/2, Lwd, L/2-Lwdwe/2,   Lwdwe, H)
    VF_1_to_123  = VF_para(Lwd/2, H/2, Lwdwewu/2,  H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwd/2, Lwd, L/2-Lwdwewu/2, Lwdwewu, H)
    
    VF_12_to_12  = VF_para(Lwdwe/2, H/2, Lwdwe/2,  H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwdwe/2,Lwdwe,L/2-Lwdwe/2, Lwdwe, H)
    VF_12_to_123 = VF_para(Lwdwe/2, H/2,Lwdwewu/2, H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwdwe/2,Lwdwe,L/2-Lwdwewu/2, Lwdwewu, H)
    VF_12_to_v   = VF_para(Lwdwe/2, H/2, L/2,      H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwdwe/2,Lwdwe, 0, L, H)
    
    VF_123_to_123= VF_para(Lwdwewu/2,H/2,Lwdwewu/2,H/2,0,0,L) + 2 * VF_perp_aligned_offsets(L/2-Lwdwewu/2,Lwdwewu,L/2-Lwdwewu/2,Lwdwewu, H)
    VF_123_to_v  = VF_para(Lwdwewu/2, H/2, L/2,       H/2, 0, 0, L) + 2 * VF_perp_aligned_offsets(L/2-Lwdwewu/2,Lwdwewu, 0, L, H)
    
    VF_1_to_2   = VF_1_to_12 - VF_1_to_1
    VF_2_to_1   = A2 and VF_1_to_2 * A1 / A2  or 0 
    VF_1_to_3   = VF_1_to_123 - VF_1_to_1 - VF_1_to_2
    VF_3_to_1   = A3 and VF_1_to_3 * A1 / A3 or 0 
    
    VF_1wd_to_5 = VF_perp_aligned(H, Lwd/2, L/2, L)
    VF_5_to_1wd = A5 and VF_1wd_to_5 * H * Lwd / A5  or 0 
    VF_5_to_1   = 4 * VF_5_to_1wd
    VF_1_to_5   = A1 and VF_5_to_1 * A5 / A1 or 0 
    
    VF_1wdwe_to_5 = VF_perp_aligned(H, Lwdwe/2, L/2, L)
    VF_5_to_1wdwe = A5 and VF_1wdwe_to_5 * H * Lwdwe / A5 or 0 
    VF_5_to_12    = 4 * VF_5_to_1wdwe
    VF_5_to_2     = VF_5_to_12 - VF_5_to_1
    VF_2_to_5     = A2 and VF_5_to_2 * A5 / A2 or 0 
    
    VF_1wdwewu_to_5 = VF_perp_aligned(H, Lwdwewu/2, L/2, L)
    VF_5_to_1wdwewu = A5 and VF_1wdwewu_to_5 * H * Lwdwewu / A5 or 0 
    VF_5_to_123     = 4 * VF_5_to_1wdwewu
    VF_5_to_3       = VF_5_to_123 - VF_5_to_1 - VF_5_to_2
    VF_3_to_5       = A3 and VF_5_to_3 * A5 / A3 or 0 
    
    VF_5_to_4 = VF_h_to_v - VF_5_to_1 - VF_5_to_2 - VF_5_to_3
    VF_4_to_5 = A4 and VF_5_to_4 * A5 / A4  or 0 
    
    VF_6_to_1 = VF_5_to_1 
    VF_1_to_6 = VF_1_to_5
    VF_6_to_2 = VF_5_to_2 
    VF_2_to_6 = VF_2_to_5
    VF_6_to_3 = VF_5_to_3 
    VF_3_to_6 = VF_3_to_5
    VF_6_to_4 = VF_5_to_4 
    VF_4_to_6 = VF_4_to_5
    
    VF_1_to_4 = 1 - VF_1_to_1 - VF_1_to_2 - VF_1_to_3 - VF_1_to_5 - VF_1_to_6
    VF_4_to_1 = A4 and VF_1_to_4 * A1 / A4 or 0 
    
    VF_12_to_3 = VF_12_to_123 - VF_12_to_12
    VF_3_to_12 = A3 and VF_12_to_3 * (A1 + A2) / A3 or 0 
    VF_3_to_2  = VF_3_to_12 - VF_3_to_1
    VF_2_to_3  = A2 and VF_3_to_2 * A3 / A2 or 0 
    
    VF_12_to_1234 = VF_12_to_v
    VF_12_to_4 = VF_12_to_1234 - VF_12_to_12 - VF_12_to_3
    VF_4_to_12 = A4 and VF_12_to_4 * (A1 + A2) / A4 or 0 
    VF_4_to_2  = VF_4_to_12 - VF_4_to_1
    VF_2_to_4  = A2 and VF_4_to_2 * A4 / A2 or 0 
    
    VF_123_to_1234 = VF_123_to_v
    VF_123_to_4    = VF_123_to_1234 - VF_123_to_123
    VF_4_to_123    = A4 and VF_123_to_4 * (A1 + A2 + A3) / A4 or 0 
    VF_4_to_3      = VF_4_to_123 - VF_4_to_1 - VF_4_to_2
    VF_3_to_4      = A3 and VF_4_to_3 * A4 / A3 or 0 
    
    VF_2_to_2 = 1 - VF_2_to_1 - VF_2_to_3 - VF_2_to_4 - VF_2_to_5 - VF_2_to_6
    VF_3_to_3 = 1 - VF_3_to_1 - VF_3_to_2 - VF_3_to_4 - VF_3_to_5 - VF_3_to_6
    VF_4_to_4 = 1 - VF_4_to_1 - VF_4_to_2 - VF_4_to_3 - VF_4_to_5 - VF_4_to_6
   
    VF = np.array([[VF_1_to_1, VF_1_to_2, VF_1_to_3, VF_1_to_4, VF_1_to_5, VF_1_to_6], \
                   [VF_2_to_1, VF_2_to_2, VF_2_to_3, VF_2_to_4, VF_2_to_5, VF_2_to_6], \
                   [VF_3_to_1, VF_3_to_2, VF_3_to_3, VF_3_to_4, VF_3_to_5, VF_3_to_6], \
                   [VF_4_to_1, VF_4_to_2, VF_4_to_3, VF_4_to_4, VF_4_to_5, VF_4_to_6], \
                   [VF_5_to_1, VF_5_to_2, VF_5_to_3, VF_5_to_4, VF_5_to_5, VF_5_to_6], \
                   [VF_6_to_1, VF_6_to_2, VF_6_to_3, VF_6_to_4, VF_6_to_5, VF_6_to_6], ])

    eps_1 = epsilon_ir 
    eps_2 = epsilon_ir 
    eps_3 = epsilon_ir 
    eps_4 = epsilon_ir 
    eps_5 = epsilon_ir 
    eps_6 = epsilon_ir 
    
    L1 = - (1-eps_1)/eps_1 * np.array([VF_1_to_1, VF_1_to_2, VF_1_to_3, VF_1_to_4, VF_1_to_5, VF_1_to_6])
    L2 = - (1-eps_2)/eps_2 * np.array([VF_2_to_1, VF_2_to_2, VF_2_to_3, VF_2_to_4, VF_2_to_5, VF_2_to_6])
    L3 = - (1-eps_3)/eps_3 * np.array([VF_3_to_1, VF_3_to_2, VF_3_to_3, VF_3_to_4, VF_3_to_5, VF_3_to_6])
    L4 = - (1-eps_4)/eps_4 * np.array([VF_4_to_1, VF_4_to_2, VF_4_to_3, VF_4_to_4, VF_4_to_5, VF_4_to_6])
    L5 = - (1-eps_5)/eps_5 * np.array([VF_5_to_1, VF_5_to_2, VF_5_to_3, VF_5_to_4, VF_5_to_5, VF_5_to_6])
    L6 = - (1-eps_6)/eps_6 * np.array([VF_6_to_1, VF_6_to_2, VF_6_to_3, VF_6_to_4, VF_6_to_5, VF_6_to_6])
    
    MD = np.diag([1 / eps_1, 1 / eps_2, 1 / eps_3, 1 / eps_4, 1 / eps_5, 1 / eps_6])
    
    mij = np.array([L1, L2, L3, L4, L5, L6]) + MD
    Ai  = np.array([A1, A2, A3, A4, A5, A6])
        
    return mij, Ai


 
def simulation(w, dfh, dfd, dfo, dfv, wl, weather, t_set_heating, t_set_cooling, \
               Qwdsol, Qwdir, Itwd, Ibwd, Qwlsolir, \
               hbh_profiles, sbs_profiles, \
               h_c, epsilon_ir, n_50, e_leak, E_lux, DLF, g_bl, \
               EXTwalls, CONDwalls, time_step):
    
    def X_hplant(t_out, t_out_max_h) :
        return 1/(1+np.exp(( t_out - t_out_max_h  + 0.5) / 0.125))

    def X_cplant(t_out, t_out_min_c) :
        return 1/(1+np.exp((- t_out + t_out_min_c + 0.5) / 0.125))

    def X_h(t_in, t_set):
        return 1 / (1 + np.exp( (t_in - t_set)/ 0.125) ) 

    def X_c(t_in, t_set):
        return 1 / (1 + np.exp( (t_set - t_in)/ 0.125) ) 

    def X_bl(It, It_max, Ib, Ib_max) :
        X1 = 1 / (1 + np.exp((It_max - It)  / 2))
        X2 = 1 / (1 + np.exp((Ib_max - Ib)  / 2))
        return max(X1, X2)

    def X_fc(t_out, t_out_min_fc, t_out_max_fc) :
        X_tmin = 1 / (1 + np.exp( (- t_out + t_out_min_fc + 0.5) / 0.125 ) )
        X_tmax = 1 / (1 + np.exp( (  t_out - t_out_max_fc + 0.5) / 0.125) )
        return X_tmin * X_tmax

    def X_dc(t_out):
        t_dc = 12
        return 1 / (1 + np.exp( (t_out - t_dc)/ 0.5) ) 
    
    def model_dT_t(t, T_val):
        T01, T02, T03, T04, T05, T06, T1, T2, T3, T4, T5, T6, T7 = np.nan_to_num(T_val, nan=0, posinf=0, neginf=0)
        TI = T7

        ihr = min(int((t - ts[0]) / 3600), len(ts)-1)
        its = min(int((t - ts[0]) / time_step), len(ts)-1)
            
        T0  = w["t_out"].values[ihr]
        T0m = w["t_out_mean"].values[ihr]
        T0p = w["t_out_pred"].values[ihr]
        Ith = w["I_th"].values[ihr]
        
        TM = 0.5 * T0 + 0.5 * TI
        
        if 'floor' in EXTwalls :
            TF = T0
        elif 'floor' in CONDwalls :
            TF = TI
        else :
            TF = TM
            
        if 'ceiling' in EXTwalls :
            TC = T0
        elif 'ceiling' in CONDwalls :
            TC = TI
        else :
            TC = TM
        
        syson   = f_on       [ihr]
        occup   = flag_occupancy[ihr]
        Tseth   = Tseth_hbh  [ihr]
        Tsetc   = Tsetc_hbh  [ihr]
        Qorad   = Qorad_hbh  [ihr]
        Qoconv  = Qoconv_hbh [ihr]
        Qlradm  = Qlrad_hbh  [ihr]
        Qlconvm = Qlconv_hbh [ihr]
        
        Qwdsolz_az = Qwdsolz_ha[ihr, :]
        Qwdirz_az  = Qwdirz_ha [ihr, :]
        Itwdz_az   = Itwdz_ha  [ihr, :]
        Ibwdz_az   = Ibwdz_ha  [ihr, :]
        
        Qwlrad2 = Qwlrad2_hbh[ihr] 
        Qwlrad3 = Qwlrad3_hbh[ihr] 
        Qwlrad4 = Qwlrad4_hbh[ihr] 
        Qwlrad5 = Qwlrad5_hbh[ihr] 
        Qwlrad6 = Qwlrad6_hbh[ihr] 
        
        qsu_m3h = qsu_m3h_sbs[its]
        qex_m3h = qex_m3h_sbs[its]
        Qhrmax  = Qhrmax_sbs [its]
        Qcrmax  = Qcrmax_sbs [its]
        Qhcmax  = Qhcmax_sbs [its]
        Qccmax  = Qccmax_sbs [its]
        
        # BLINDS CONTROL
        
        Itm = 150
        Ibm = 25
        
        Qwdsol_a = Qwdsolz_ha[ihr, :]
        Qwdir_a  = Qwdirz_ha [ihr, :]
        Itwd_a   = Itwdz_ha  [ihr, :]
        Ibwd_a   = Ibwdz_ha  [ihr, :]
        
        if (syson == 1) :
            Qsol = np.array([Qi * (1 - (1 - g_bl) * X_bl(Iti, Itm, Ibi, Ibm)) for (Qi, Iti, Ibi) in zip(Qwdsol_a, Itwd_a, Ibwd_a)]).sum()
            Qir  = np.array([Qi * (1 - (1 - g_bl) * X_bl(Iti, Itm, Ibi, Ibm)) for (Qi, Iti, Ibi) in zip(Qwdir_a,  Itwd_a, Ibwd_a)]).sum()
            daylight_lux = (Qsol/Qwdsol_a.sum()) * Ith * 100 * DLF if Qwdsol_a.sum() > 0 else 0
        else:
            Qsol = g_bl * Qwdsol_a.sum()
            Qir  = g_bl * Qwdir_a.sum()
            daylight_lux = g_bl * Ith * 100 * DLF
        
        Q01  = Qir
        
        f_light  = np.clip(1 - daylight_lux / E_lux, 0, 1)
        Qlrad    = f_light * Qlradm
        Qlconv   = f_light * Qlconvm
        Qlight   = Qlrad + Qlconv
        
        
        # HEATING AND COOLING CONTROL
        
        t_out_max_h = 11
        t_out_min_c = 9
        
        Xhp = X_hplant(T0m, t_out_max_h)
        Xcp = X_cplant(T0m, t_out_min_c)
        Xhe = X_h(TI, Tseth)
        Xce = X_c(TI, Tsetc)
        Xdc = X_dc(T0m)
        
        t_set_fc = Xcp * t_set_heating + Xhp * t_set_cooling
        
        Qh_rad  = Xhp * Xhe * Qhrmax
        Qc_rad  = Xcp * Xce * Qcrmax
        Qh_conv = Xhp * Xhe * Qhcmax
        Qc_conv = Xcp * Xce * Qccmax
        
        Qcdc_rad  = Qc_rad  * Xdc       # Dry cooler
        Qcdc_conv = Qc_conv * Xdc       # Dry cooler
        Qcch_rad  = Qc_rad  - Qcdc_rad  # Chiller
        Qcch_conv = Qc_conv - Qcdc_conv # Chiller
      
        
        # FREE COOLING CONTROL AND HEAT RECOVERY CONTROL
        
        t_out_min_fc = occup * 14 + (1-occup) * 10
        t_out_max_fc = 25
        
        f_fc = X_fc(T0, t_out_min_fc, t_out_max_fc) * X_c(TI, t_set_fc) 
        qex_tot_m3h = qex_m3h + f_fc * q_fc_m3h
        
        f_leak = 8
        
        q_leak_bal_m3h = q_leak_m3h / (1 + (f_leak/e_leak) * (qsu_m3h - qex_tot_m3h)**2 / Vi**2 / n_50**2)      
        q_leak_tot_m3h = q_leak_bal_m3h + max(qex_tot_m3h - qsu_m3h, 0)
        q_vent_m3h     = q_leak_tot_m3h + qsu_m3h * (1 - (1 - f_fc) * epsilon_rec)
        
        U0 = rho_a * c_p_a * q_vent_m3h / 3600
        
        
        # COOLING CEILING CONTROL (EER = 3, dta_cd = 35 - 30, epsilon_dc = 0.7, epsilon_hx = 0.6)
        
        C_dot_a = Qcrmax * (1 + 1/3) / (35 - 30)
        Qrfc    = Xcp * Xdc * X_c(TI, t_set_fc) * 0.7 * 0.6 * C_dot_a * (T6 - T0)
        
        
        # MODEL
        
        Ai_wl = np.array([A1, A2, A3, A4, A5, A6])
        qinrad = (Qorad + Qlrad) / Ai_wl.sum() if Ai_wl.sum() > 0 else np.zeroes(6)
        
        [i02, i03, i04, i05, i06] = [Qwlrad2, Qwlrad3, Qwlrad4, Qwlrad5, Qwlrad6]
        [i1, i2, i3, i4, i5, i6]  = qinrad * Ai_wl + Qh_rad * fhri - Qc_rad * fcri 
        
        i7 = Qoconv + Qlconv + Qh_conv - Qc_conv
        
        Tiwl = np.array([T1, T2, T3, T4, T5, T6])
        
        Ebi =  sigma_b * T_ref**4 + sigma_b * 4 * T_ref**3 * Tiwl  
        Ji  =  mij_inv @ Ebi
        qi  = (Ebi - Ji) * epsilon_ir / (1 - epsilon_ir)
        Qnet  = qi * Aiz

        dT01_t = (- 1/C01 * U01 * (T01 - T0) + 1/C01 * U1  * (T1 - T01) + Q01 /C01)  if C01 > 0 else 0
        dT02_t = (- 1/C02 * U02 * (T02 - T0) + 1/C02 * U2  * (T2 - T02) + i02 /C02)  if C02 > 0 else 0
        dT03_t = (- 1/C03 * U03 * (T03 - TM) + 1/C03 * U3  * (T3 - T03) + i03 /C03)  if C03 > 0 else 0
        dT04_t = (- 1/C04 * U04 * (T04 - TI) + 1/C04 * U4  * (T4 - T04) + i04 /C04)  if C04 > 0 else 0
        dT05_t = (- 1/C05 * U05 * (T05 - TF) + 1/C05 * U5  * (T5 - T05) + i05 /C05)  if C05 > 0 else 0
        dT06_t = (- 1/C06 * U06 * (T06 - TC) + 1/C06 * U6  * (T6 - T06) + i06 /C06)  if C06 > 0 else 0
        
        dT1_t = (- 1/C1 * U1 * (T1 - T01) + 1/C1 * U17 * (T7 - T1) + (i1 - Qnet[0])/C1) if C1 > 0 else 0
        dT2_t = (- 1/C2 * U2 * (T2 - T02) + 1/C2 * U27 * (T7 - T2) + (i2 - Qnet[1])/C2) if C2 > 0 else 0
        dT3_t = (- 1/C3 * U3 * (T3 - T03) + 1/C3 * U37 * (T7 - T3) + (i3 - Qnet[2])/C3) if C3 > 0 else 0
        dT4_t = (- 1/C4 * U4 * (T4 - T04) + 1/C4 * U47 * (T7 - T4) + (i4 - Qnet[3])/C4) if C4 > 0 else 0
        dT5_t = (- 1/C5 * U5 * (T5 - T05) + 1/C5 * U57 * (T7 - T5) + (i5 - Qnet[4] + Qsol)/C5) if C5 > 0 else 0
        dT6_t = (- 1/C6 * U6 * (T6 - T06) + 1/C6 * U67 * (T7 - T6) + (i6 - Qnet[5] - Qrfc)/C6) if C6 > 0 else 0
    
        dT7_t = (- 1/C7 * U0  * (T7 - T0) \
                - 1/C7 * U17 * (T7 - T1) - 1/C7 * U27 * (T7 - T2) - 1/C7 * U37 * (T7 - T3)  \
                - 1/C7 * U47 * (T7 - T4) - 1/C7 * U57 * (T7 - T5) - 1/C7 * U67 * (T7 - T6) + i7 /C7) if C7 > 0 else 0

        dti = [dT01_t, dT02_t, dT03_t, dT04_t, dT05_t, dT06_t, dT1_t, dT2_t, dT3_t, dT4_t, dT5_t, dT6_t, dT7_t]
        
        return  (dti, Qh_rad, Qh_conv, Qcch_rad, Qcch_conv, Qlight, Qsol, Qrfc, q_vent_m3h)

    
    def dT_t(tau, T_i):
        ret = model_dT_t(tau, T_i)
        return ret[0]
    
    
    nsph = int(3600 / time_step) 
    hour_sbs  = np.repeat(w['hour_yr'].values, nsph)
    t_out_sbs = np.repeat(w['t_out'].values, nsph)
    ts  = np.arange(0, time_step * len(hour_sbs), time_step)
    hr  = ts /3600
    day = hr /24
    
    zname_list = []
    dfr_list   = []

    for i in range(0, len(dfv)):
        
        hour = w['hour_yr'].values

        zname       = dfv.iloc[i]['zone']
        area_floor  = dfv.iloc[i]['area_fl']
        Vi          = dfv.iloc[i]['vol_int']
        epsilon_rec = dfv.iloc[i]['epsilon_rec']
        q_fc_m3h    = dfv.iloc[i]['q_fc_m3h']
        q_leak_m3h  = dfv.iloc[i]['q_leak_m3h']
        
        Ca          = Vi * rho_a * c_p_a * 5
        
        f_on, f_oc, flag_occupancy = schedules(zname, dfh, dfd, dfo, hour) # hour by hour
        
        # Hour by hour vectors
        h_profiles  = hbh_profiles[zname].copy()
        Tseth_hbh   = h_profiles['tseth'].values
        Tsetc_hbh   = h_profiles['tsetc'].values
        Qorad_hbh   = h_profiles['Qro'].values
        Qoconv_hbh  = h_profiles['Qco'].values
        Qlrad_hbh   = h_profiles['Qrl'].values
        Qlconv_hbh  = h_profiles['Qcl'].values
        
        # Hour by hour dataframes to arrays (hour, azimuth) 
        Qwdsolz_ha  = Qwdsol[zname].copy().drop(columns=['hour_yr']).to_numpy()
        Qwdirz_ha   = Qwdir [zname].copy().drop(columns=['hour_yr']).to_numpy()
        Itwdz_ha    = Itwd  [zname].copy().drop(columns=['hour_yr']).to_numpy()
        Ibwdz_ha    = Ibwd  [zname].copy().drop(columns=['hour_yr']).to_numpy()
        
        # Hour by hour dataframes to vectors        
        Qwlradz     = Qwlsolir[zname].copy()
        Qwlrad2_hbh = Qwlradz['wall_ext'   ].values if 'wall_ext'    in Qwlradz else np.zeros(len(Qwlradz))
        Qwlrad3_hbh = Qwlradz['wall_uncond'].values if 'wall_uncond' in Qwlradz else np.zeros(len(Qwlradz))
        Qwlrad4_hbh = Qwlradz['wall_cond'  ].values if 'wall_cond'   in Qwlradz else np.zeros(len(Qwlradz))
        Qwlrad5_hbh = Qwlradz['floor'      ].values if 'floor'       in Qwlradz else np.zeros(len(Qwlradz))
        Qwlrad6_hbh = Qwlradz['ceiling'    ].values if 'wall_ext'    in Qwlradz else np.zeros(len(Qwlradz))

        # Step by step vectors
        s_profiles  = sbs_profiles[zname].copy()
        Qhrmax_sbs  = s_profiles['Qhr'].values
        Qcrmax_sbs  = s_profiles['Qcr'].values
        Qhcmax_sbs  = s_profiles['Qhc'].values
        Qccmax_sbs  = s_profiles['Qcc'].values
        qsu_m3h_sbs = s_profiles['qsu'].values
        qex_m3h_sbs = s_profiles['qex'].values
        
        wlz = wl.loc[wl['zone'] == zname].copy()
        lstwl = set(wlz['wall_type'])
        lstsl = set(wlz['slope'])
        
        condwe = (wlz['slope'] ==  90) & (wlz['wall_type'] == 'wall_ext'   )
        condwu = (wlz['slope'] ==  90) & (wlz['wall_type'] == 'wall_uncond')
        condwc = (wlz['slope'] ==  90) & (wlz['wall_type'] == 'wall_cond'  )
        condfl =  wlz['slope'] ==   0
        condce =  wlz['slope'] == 180
        
        [U01, U1, U17,          A1] = wlz.loc[condwe][['Aho_wd','Ahm_wd','Ahc_wd','area_wd']].values[0] \
                                        if 'wall_ext' in lstwl    else [0, 0, 0, 0]
        [U02, U2, U27, C2, C02, A2] = wlz.loc[condwe][['Ahc', 'Ahm', 'Aho', 'ACi', 'ACe', 'area_wl']].values[0] \
                                        if 'wall_ext' in lstwl    else [0, 0, 0, 0, 0, 0]
        [U03, U3, U37, C3, C03, A3] = wlz.loc[condwe][['Ahc', 'Ahm', 'Aho', 'ACi', 'ACe', 'area_wl']].values[0] \
                                        if 'wall_uncond' in lstwl else [0, 0, 0, 0, 0, 0]
        [U04, U4, U47, C4, C04, A4] = wlz.loc[condwe][['Ahc', 'Ahm', 'Aho', 'ACi', 'ACe', 'area_wl']].values[0] \
                                        if 'wall_cond' in lstwl   else [0, 0, 0, 0, 0, 0]
        [U05, U5, U57, C5, C05, A5] = wlz.loc[condfl][['Ahc', 'Ahm', 'Aho', 'ACi', 'ACe', 'area_wl']].values[0] \
                                        if   0 in lstsl           else [0, 0, 0, 0, 0, 0]
        [U06, U6, U67, C6, C06, A6] = wlz.loc[condce][['Ahc', 'Ahm', 'Aho', 'ACi', 'ACe', 'area_wl']].values[0] \
                                        if 180 in lstsl           else [0, 0, 0, 0, 0, 0]
            
        rho_gl = 2500 #[kg/m^3] glazing
        c_gl   = 750 #[J/kg.K]
        t_gl   = 0.003 #[m]
        C1     = t_gl * rho_gl * c_gl * A1
        C01    = 14 * C1 / 3    # 4 mm glazing + 1 cm Blinds
        
        C7 = Ca 
        
        [fhr2, fcr2] = wlz.loc[condwe][['f_h_rad', 'f_c_rad']].values[0] if 'wall_ext'    in lstwl else [0,0]
        [fhr3, fcr3] = wlz.loc[condwe][['f_h_rad', 'f_c_rad']].values[0] if 'wall_uncond' in lstwl else [0,0]
        [fhr4, fcr4] = wlz.loc[condwe][['f_h_rad', 'f_c_rad']].values[0] if 'wall_cond'   in lstwl else [0,0]
        [fhr5, fcr5] = wlz.loc[condfl][['f_h_rad', 'f_c_rad']].values[0] if   0           in lstsl else [0,0]
        [fhr6, fcr6] = wlz.loc[condce][['f_h_rad', 'f_c_rad']].values[0] if 180           in lstsl else [0,0]
        
        fhri = np.array([0, fhr2, fhr3, fhr4, fhr5, fhr6])
        fcri = np.array([0, fcr2, fcr3, fcr4, fcr5, fcr6])
        
        mijz, Aiz = zone_mij(zname, dfv, wl, epsilon_ir)
        mij_inv   = np.linalg.inv(mijz)

        ti_init = t_set_heating
        to_init = 20 if weather == 1 else 10
        Tinit   = list(to_init * np.ones(6)) + list(ti_init * np.ones(7))
        
        T_array = solve_ivp(dT_t, (ts[0],ts[-1]), Tinit, t_eval = ts, method = "BDF", rtol = 0.01)
    
        t_in = T_array.y[12]
                
        Qh_r = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[1] for i in range(*T_array.t.shape)])
        Qh_c = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[2] for i in range(*T_array.t.shape)])
        Qc_r = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[3] for i in range(*T_array.t.shape)])
        Qc_c = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[4] for i in range(*T_array.t.shape)])
        Ql_t = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[5] for i in range(*T_array.t.shape)])
        Qs_t = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[6] for i in range(*T_array.t.shape)])
        Qf_t = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[7] for i in range(*T_array.t.shape)])
        qm3h = np.array([model_dT_t(T_array.t[i], T_array.y[:,i])[8] for i in range(*T_array.t.shape)])

        Qh_Wm2   = (Qh_r + Qh_c) / area_floor if area_floor > 0 else 0
        Qc_Wm2   = (Qc_r + Qc_c) / area_floor if area_floor > 0 else 0 
        Ql_Wm2   =  Ql_t         / area_floor if area_floor > 0 else 0 
        Qs_Wm2   =  Qs_t         / area_floor if area_floor > 0 else 0 
        Qf_Wm2   =  Qf_t         / area_floor if area_floor > 0 else 0 
        
        Qh_kWhm2 = np.cumsum(Qh_Wm2) /1000 * time_step /3600 
        Qc_kWhm2 = np.cumsum(Qc_Wm2) /1000 * time_step /3600
        Ql_kWhm2 = np.cumsum(Ql_Wm2) /1000 * time_step /3600
        Qs_kWhm2 = np.cumsum(Qs_Wm2) /1000 * time_step /3600
        Qf_kWhm2 = np.cumsum(Qf_Wm2) /1000 * time_step /3600
        Vv_m3    = np.cumsum(qm3h)         * time_step /3600
        
        f_oc_sbs      = np.repeat(f_oc, nsph)
        occupancy_sbs = np.repeat(flag_occupancy, nsph)
        t_set_h_sbs   = np.repeat(Tseth_hbh, nsph)
        t_set_c_sbs   = np.repeat(Tsetc_hbh, nsph)
        
        overcool_ts   = np.where((f_oc_sbs > 0) & (t_in < t_set_h_sbs - 0.5), 1, 0)
        overheat_ts   = np.where((f_oc_sbs > 0) & (t_in > t_set_c_sbs + 0.5), 1, 0)
        
        fr_overcool   = np.cumsum(overcool_ts) / occupancy_sbs.sum()
        fr_overheat   = np.cumsum(overheat_ts) / occupancy_sbs.sum()
        
        dfr = pd.DataFrame(list(zip ( hr, day, t_out_sbs, t_in, f_oc_sbs, qm3h, \
                                   Qh_Wm2,   Qc_Wm2,  Ql_Wm2, Qs_Wm2, Qf_Wm2, \
                                   overcool_ts, overheat_ts, Vv_m3, \
                                   Qh_kWhm2, Qc_kWhm2, Ql_kWhm2, Qs_kWhm2, Qf_kWhm2, \
                                   fr_overcool, fr_overheat)), \
                        columns = ['hour_yr', 'day', 't_ext', 't_in', 'f_occ', 'q_m3h', \
                                   'Qh_Wm2', 'Qc_Wm2', 'Ql_Wm2', 'Qs_Wm2', 'Qf_Wm2', \
                                   'overcool', 'overheat', 'Vv_m3', \
                                   'Qh_kWhm2', 'Qc_kWhm2', 'Ql_kWhm2', 'Qs_kWhm2', 'Qf_kWhm2', \
                                   'fr_overcool', 'fr_overheat'])
        
        zname_list.append(zname)
        dfr_list.append(dfr)
    

    return zname_list, dfr_list




