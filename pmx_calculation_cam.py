# ---------------------------------------------------------------
# A post-processing script for reading aerosols and calculating
# PM2.5 (ambient μg/m3 or STP μg/sm3) from CESM's output files
# version 1 by dmleung: 6 Nov 2026
# ---------------------------------------------------------------

import numpy as np
import xarray as xr
from scipy.special import erf
import re

# ---------------------------------------------------------------
# PMx function for CAM/MAM (volume/mass CDF)
# ---------------------------------------------------------------
def pmx_total(mmrs, dgns, sigmas, rhos, Dc):
    """
    PMx from modal aerosol mass assuming lognormal, internally mixed modes.

    mmrs   : (..., nm) mass per mode (kg/kg, ug/m3, ... any unit)
    dgns   : (..., nm) number-median (geometric mean) dry diameters [m]
    sigmas : (nm,)     geometric standard deviations
    rhos   : (nm,) or (..., nm) dry particle density of each mode [g/cm3]
    Dc     : scalar    cutoff *aerodynamic* diameter [m]
    Mode axis is the last axis.
    """
    mmrs     = np.asarray(mmrs)
    dgns     = np.asarray(dgns)
    sigmas   = np.asarray(sigmas, dtype=float)
    rhos     = np.asarray(rhos, dtype=float)
    ln_sigma = np.log(sigmas)

    # mass median aerodynamic diameter, as daero_a in CESM:
    #   daero = dgn * exp(3 ln^2 sigma) * sqrt(rho / rho_water)
    daero = dgns * np.exp(3.0 * ln_sigma**2) * np.sqrt(rhos / RHO_water)
    arg   = np.log(Dc / daero) / (np.sqrt(2.0) * ln_sigma)
    frac  = 0.5 * (1.0 + erf(arg))
    return np.sum(mmrs * frac, axis=-1)

# converting from mass mixing ratio (MMR, kg/kg) to mass density (or mass concentration) μg/m3 or μg/sm3
def mmr_to_ugm3(da, stp_or_amb):
    # please identify also whether you want STP (typically for aircraft campaigns) or ambient densities (typical for ground-based measurements)
    if (stp_or_amb == 'STP'): return da * RHO_STP * 1e9
    elif (stp_or_amb == 'AMB'): return da * RHO_AIR * 1e9
    else: raise ValueError("stp_or_amb must be 'STP' or 'AMB'")


# ---------------------------------------------------------------
# Cut sizes and unit conversion
# ---------------------------------------------------------------
SIG = (1.6, 1.6, 1.8)          # modes a1-a3 for most species; must match sigmag in the run's mode files
SIG_bcpom = (1.6, 1.6)          # modes a1, a4 for bc and pom; must match sigmag in the run's mode files
SIG_soa = (1.6, 1.6)          # modes a1, a2 for soa; must match sigmag in the run's mode files

# g/cm3, reference density for aerodynamic diameter
RHO_water = 1.0


# modal densities g/cm3
RHO_MODE = np.array([1.7, 1.7, 2.3, 1.0])   # modes a1-a4, placeholder dry densities [g/cm3]

# dmleung: A rigorous calculation of RHO_MODE requires all species. Ignored for now but uncomment when you want it.
# To use this block, move it below the "load dataset" section.
# densities are the 'density' variable (kg/m3) in each species' physprops file listed
# under rad_climate in the run's atm_in, e.g.
#   ncdump -v density /glade/campaign/cesm/cesmdata/inputdata/atm/cam/physprops/sulfate_rrtmg_c080918.nc
# following is the default (g/cm3):
# RHO_SO4 = 1.77
# RHO_POM = 1.0
# RHO_SOA = 1.0
# RHO_BC  = 1.7
# RHO_DST = 2.6
# RHO_NCL = 1.9
# # ... and so on. Add the species you need here.
# # total SOA per mode (sums soa1_aN, soa2_aN, ... or a single soa_aN)
# soa = {m: sum(ds[v] for v in ds.data_vars if re.fullmatch(rf'soa\d*_a{m}', v)) for m in (1, 2)}
# # mode-mean dry density, rho_mode = sum(m_i) / sum(m_i / rho_i); kg/kg is fine, units cancel
# # default MAM4: a1 = so4 pom soa bc dst ncl, a2 = so4 soa ncl dst, a3 = dst ncl so4, a4 = pom bc
# rho_a1 = ((ds['so4_a1'] + ds['pom_a1'] + soa[1] + ds['bc_a1'] + ds['dst_a1'] + ds['ncl_a1']) /
#           (ds['so4_a1']/RHO_SO4 + ds['pom_a1']/RHO_POM + soa[1]/RHO_SOA +
#            ds['bc_a1']/RHO_BC + ds['dst_a1']/RHO_DST + ds['ncl_a1']/RHO_NCL)).fillna(1.77)
# rho_a2 = ((ds['so4_a2'] + soa[2] + ds['ncl_a2'] + ds['dst_a2']) /
#           (ds['so4_a2']/RHO_SO4 + soa[2]/RHO_SOA + ds['ncl_a2']/RHO_NCL + ds['dst_a2']/RHO_DST)).fillna(1.77)
# rho_a3 = ((ds['dst_a3'] + ds['ncl_a3'] + ds['so4_a3']) /
#           (ds['dst_a3']/RHO_DST + ds['ncl_a3']/RHO_NCL + ds['so4_a3']/RHO_SO4)).fillna(2.6)
# rho_a4 = ((ds['pom_a4'] + ds['bc_a4']) /
#           (ds['pom_a4']/RHO_POM + ds['bc_a4']/RHO_BC)).fillna(1.0)
# RHO_MODE = np.stack([rho_a1, rho_a2, rho_a3, rho_a4], axis=-1)   # modes a1-a4, modal mean dry densities [g/cm3]



# specify mass diameter
Dcutoff_PM25  = 2.5e-6    # aerodynamic diameter, m
Dcutoff_PM10  = 10.0e-6   # aerodynamic diameter, m


# STP air density (273.15 K, 1013.25 hPa) -> ug/sm3. Verify this matches the obs STP definition.
RHO_STP = 101325.0 / (287.05 * 273.15)     # ~1.292 kg/m3


# ---------------------------------------------------------------
# load dataset (*hs* or *aircraft*)
# ---------------------------------------------------------------
ds = xr.open_dataset('/glade/derecho/scratch/dleung/260713_mosaic_cesm3_LTt4s.ne30/run/260713_mosaic_cesm3_LTt4s.ne30.cam.hs.2016-01-01-01800.nc')

# ambient air density (kg/m3), only used for 'AMB' in mmr_to_ugm3(). 
# If 'STP' is used, there is no RHO_AIR in mmr_to_ugm3().
RHO_AIR = ds['PMID'] / (287.05 * ds['T'])
# RHO_AIR = ds['RHO_CLUBB'] # alternatively, use RHO_CLUBB
# RHO_AIR = ds['RHO_AIR'] # alternatively, use RHO_AIR


# now choose stp_or_amb
STP_OR_AMB = 'STP' # if comparing against aircraft measurements
# STP_OR_AMB = 'AMB' # otherwise

# species unit conversion from kg/kg to μg/m3 (AMB) or μg/sm3 (STP)
so4_a1, so4_a2, so4_a3 = [mmr_to_ugm3(ds[f'so4_a{i}'], STP_OR_AMB) for i in (1, 2, 3)]
bc_a1, bc_a4 = [mmr_to_ugm3(ds[f'bc_a{i}'], STP_OR_AMB) for i in (1, 4)]
dst_a1, dst_a2, dst_a3 = [mmr_to_ugm3(ds[f'dst_a{i}'], STP_OR_AMB) for i in (1, 2, 3)]
soa1_a1, soa1_a2 = [mmr_to_ugm3(ds[f'soa1_a{i}'], STP_OR_AMB) for i in (1, 2)]


# dry number mean geometric diameter, m
dgnum1 = ds['dgnum1']
dgnum2 = ds['dgnum2']
dgnum3 = ds['dgnum3']
dgnum4 = ds['dgnum4']


# PM mass density (or mass concentration) calculation, μg/m3 or μg/sm3

# sulfate PM2.5 mass concentration (μg/m3 or μg/sm3)
so4_pm25_a  = pmx_total(np.stack([so4_a1, so4_a2, so4_a3], axis=-1),
                            np.stack([dgnum1, dgnum2, dgnum3], axis=-1),
                            SIG, RHO_MODE[..., [0, 1, 2]], 
                            Dcutoff_PM25)


# black carbon PM2.5 mass concentration (μg/m3 or μg/sm3)
bc_pm25_a  = pmx_total(np.stack([bc_a1, bc_a4], axis=-1),
                            np.stack([dgnum1, dgnum4], axis=-1),
                            SIG_bcpom, RHO_MODE[..., [0, 3]],   # bc: a1, a4
                            Dcutoff_PM25)

# SOA1 PM2.5 mass concentration (μg/m3 or μg/sm3)
soa1_pm25_a  = pmx_total(np.stack([soa1_a1, soa1_a2], axis=-1),
                            np.stack([dgnum1, dgnum2], axis=-1),
                            SIG_soa, RHO_MODE[..., [0, 1]], Dcutoff_PM25)

#... add other species you like...



