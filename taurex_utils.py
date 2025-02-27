import taurex
import numpy as np
import matplotlib.pyplot as plt
import inspect

from taurex.cache import OpacityCache, CIACache
from taurex.contributions import *
from taurex.data.stellar import BlackbodyStar
from taurex.data import Planet
from taurex.data.profiles.chemistry import TaurexChemistry
from taurex.data.profiles.chemistry import ConstantGas
from taurex.data.profiles.temperature import Isothermal
from taurex.model import TransmissionModel
import taurex.log
taurex.log.disableLogging()

opacity_path="../TauRex_tutorial/taurex3_xsec_hdf5_sampled_R15000_0.3-50/"
CIA_path="../TauRex_tutorial/HITRAN"

###############################
# params from the dataset gen #
###############################

he_h2 = 0.17
n_layers = 100
# max_pressure = 10 #bar
# min_pressure = 10e-10

max_pressure = 1e6 #pascal
min_pressure = 1e-4

high_resolution_R = 10000



def create_full_forward_model(Rs, Ts, Rp, Mp, Tp,
                        X_h2o,
                        X_ch4,
                        X_co,
                        X_co2,
                        X_nh3,
                        opacity_path=opacity_path,
                        CIA_path=CIA_path):
    """This function helps to set up a basic planet from taurex. TauREx creates an Object for each component,
    and brings them all togther using Transmission Model"""

    OpacityCache().set_opacity_path(opacity_path)
    CIACache().set_cia_path(CIA_path)

    planet = Planet(planet_radius=Rp, planet_mass=Mp) ## set up the "shape" about the exoplanet itself 

    star = BlackbodyStar(temperature=Ts, radius=Rs)## Information about the host star

    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=he_h2) ## chemistry, here fill the atmosphere with H2, He (aka primary atmosphere)
    
    # print(f'X_h2o: {X_h2o}, X_ch4: {X_ch4}, X_co: {X_co}, X_co2: {X_co2}, X_nh3: {X_nh3}')
    # print(f'rs: {Rs}, Ts: {Ts}, Rp: {Rp}, Mp: {Mp}, Tp: {Tp}')
    
    ## we add our gases here, and the corresponding abundances
    chemistry.addGas(ConstantGas('H2O', mix_ratio=10**X_h2o)) 
    chemistry.addGas(ConstantGas('CH4', mix_ratio=10**X_ch4)) 
    chemistry.addGas(ConstantGas('CO', mix_ratio=10**X_co))
    chemistry.addGas(ConstantGas('CO2', mix_ratio=10**X_co2))
    chemistry.addGas(ConstantGas('NH3', mix_ratio=10**X_nh3))
    
    iso_profile = Isothermal(T=Tp) # temperature - pressure profile of the planet
    
    # observe the planet in transit mode. Set up the extent of the atmosphere (in terms of pressure level), and the number of layers (accuruacy of the simulation)
    forward_model = TransmissionModel(planet=planet, temperature_profile=iso_profile, chemistry=chemistry, star=star,
                                      atm_min_pressure=min_pressure, atm_max_pressure=max_pressure, nlayers=n_layers) 

    # atmospheric phenomena, there are quite a few, another must have is CIA
    forward_model.add_contribution(AbsorptionContribution())
#     forward_model.add_contribution(LeeMieContribution()) <-- this is a specific assumption on cloud

    forward_model.build()

    return forward_model

def create_contribution_model(Rs, Ts, Rp, Mp, Tp,
                        X_mol,
                        mol,
                        opacity_path=opacity_path,
                        CIA_path=CIA_path):
    """This function helps to set up a basic planet from taurex. TauREx creates an Object for each component,
    and brings them all togther using Transmission Model"""

    OpacityCache().set_opacity_path(opacity_path)
    CIACache().set_cia_path(CIA_path)

    planet = Planet(planet_radius=Rp, planet_mass=Mp) ## set up the "shape" about the exoplanet itself 
    star = BlackbodyStar(temperature=Ts, radius=Rs)## Information about the host star
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=he_h2) ## chemistry, here fill the atmosphere with H2, He (aka primary atmosphere)

    # print(f'X_mol: {X_mol}')
    # print(f'rs: {Rs}, Ts: {Ts}, Rp: {Rp}, Mp: {Mp}, Tp: {Tp}')
    ## we add our gases here, and the corresponding abundances
    chemistry.addGas(ConstantGas(mol, mix_ratio=10**X_mol)) 

    iso_profile = Isothermal(T=Tp) # temperature - pressure profile of the planet
    # observe the planet in transit mode. Set up the extent of the atmosphere (in terms of pressure level), and the number of layers (accuruacy of the simulation)
    forward_model = TransmissionModel(planet=planet, temperature_profile=iso_profile, chemistry=chemistry, star=star,
                                      atm_min_pressure=min_pressure, atm_max_pressure=max_pressure, nlayers=n_layers) 
    # atmospheric phenomena, there are quite a few, another must have is CIA
    forward_model.add_contribution(AbsorptionContribution())
#     forward_model.add_contribution(LeeMieContribution()) <-- this is a specific assumption on cloud

    forward_model.build()

    return forward_model

def bin_forward_model(native_wl, native_spectrum, wn, wn_width):
    from taurex.binning import FluxBinner

    fb = FluxBinner(wn, wn_width)

    out = fb.bindown(native_wl, native_spectrum)

    return out[0], out[1]

def update_forward_model(forward_model, mol, X_mol):
    '''code to update the paramters without updating the model assumption, note that the object should be re-initiated
    if there are changes in model assumptions'''
    from taurex.constants import RSOL

    for x_i, mol_i in zip(X_mol, mol):
        forward_model[mol_i] = 10**x_i


    res = forward_model.model()

    return res, forward_model

ariel_wn_grid = np.array([1374.46479629,  1467.62296582,  1567.09518906,  1673.30941854,
            1786.72261246,  1907.82270064,  2037.13068369,  2175.20287447,
            2322.63329152,  2480.05621461,  2685.68203171,  2739.66424054,
            2794.73149178,  2850.90559476,  2908.20879722,  2966.66379404,
            3026.2937363,   3087.1222404,   3149.17339743,  3212.47178272,
            3277.04246555,  3342.91101911,  3410.1035306,   3478.64661156,
            3548.56740845,  3619.89361336,  3692.65347499,  3766.87580984,
            3842.59001362,  3919.82607289,  3998.61457696,  4078.98672995,
            4160.97436323,  4244.60994793,  4329.92660788,  4416.9581327,
            4505.73899117,  4596.30434489,  4688.69006222,  4782.93273247,
            4879.06968039,  4977.13898097,  5077.17947449,  5308.9460905,
            5853.11306478,  6453.05715392,  7114.4955122,   7843.7313022,
            8647.71376067, 10526.31578947, 14285.71428571, 18181.81818182])
ariel_wnwidth = np.array([90.20153667,   96.31519638,  102.84322636,  109.8137117,
                 117.25664105,  125.20403561,  133.69008691,  142.75130392,
                 152.42667007,  162.75781104,   53.45037745,   54.52473004,
                 55.62067711,   56.73865272,   57.87909964,   59.04246954,
                 60.22922318,   61.43983057,   62.67477116,   63.93453406,
                 65.2196182,   66.53053252,   67.86779622,   69.23193893,
                 70.6235009,   72.04303327,   73.49109824,   74.96826931,
                 76.47513153,   78.01228167,   79.58032853,   81.17989313,
                 82.81160899,   84.47612233,   86.17409239,   87.90619164,
                 89.67310609,   91.47553553,   93.31419379,   95.18980909,
                 97.10312425,   99.05489705,  101.04590048,  518.87123057,
                 572.05553171,  630.69122371,  695.33707414,  766.60912424,
                 845.18655947, 3409.09090909, 4166.66666667, 3333.33333333])


def full_contribution_array(mols, X_mols, Rs, Ts, Rp, Mp, Tp):
    FM_c_dic = {}
    o_c_dic = {}

    for mol, X_mol in zip(mols, X_mols):
        FM_c_dic[mol] = create_contribution_model(mol=mol, X_mol=X_mol, Rp=Rp, Tp=Tp, Ts=Ts, Mp=Mp, Rs = Rs)#
        o_c_dic[mol] = bin_forward_model(FM_c_dic[mol].model()[0], FM_c_dic[mol].model()[1], ariel_wn_grid, ariel_wnwidth)
    
    FM = create_full_forward_model(Rs, Ts, Rp, Mp, Tp,
                        -5,
                        -5,
                        -5,
                        -5,
                        -5)#placeholder values we will update now
    res, __ = update_forward_model(FM,  mols, X_mols)# 
    o = bin_forward_model(res[0], res[1], ariel_wn_grid,ariel_wnwidth, )
    o_c_dic['Full Model'] = o

    return o_c_dic


def get_mols():
    # # print(f'the get_mols() function (line {inspect.currentframe().f_back.f_lineno}) is depreciated and should not be used!')
    return ['H2O','CO2','CH4','CO', 'NH3']
    

def get_ariel():
    '''get the ariel grid and width
    returns
    -grid
    -width'''
    return ariel_wn_grid, ariel_wnwidth

############################
# start with a basic model #
############################
if __name__ == '__main__':
#                                                                  R_sun, T_kelvin, R_earth, M_earth, T_kelvin
    o_c_dic = full_contribution_array(get_mols(), [-5, -5, -5, -5, -5], Rs=1, Ts=1000, Rp=1, Mp=1, Tp=200)

    for mol in get_mols():
        plt.plot(10000/o_c_dic[mol][0], o_c_dic[mol][1], label=mol)

    plt.plot(10000/o_c_dic['Full Model'][0], o_c_dic['Full Model'][1],"k--", label='Full Model')
    plt.xscale('log')
    plt.legend()
    plt.show()