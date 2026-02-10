from manhattan import manhattan
from mnms.time import Dt
import numpy as np
from pt_lines_descript import *

def ghost_acc(t):
    seconds = t.to_seconds()
    base = 0
    if seconds < 7 * 60 * 60 or seconds > 9 * 60 * 60:
        ncars = base
    elif 7 * 60 * 60 <= seconds <= 8 * 60 * 60:
        ncars = (50 / 36) * seconds - 35000
    else:
        ncars = -(50 / 36) * seconds + 45000
    return {"CAR": ncars}

if __name__=='__main__':

    basic = True
    car_traffic = False
    sharp_peak = False
    exhaustive_lt = False

    ### LYON - ALL METHODS - EFFECTS OF PT CONGESTION NO CAR ###
    if basic:
        SCENARIO = "Lyon"

        ## Simu params
        SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                       "DT_FLOW": Dt(minutes=1),
                       "AFFECTATION_FACTOR": 5
                       }

        ## Grid params
        GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
        GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
        GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

        ## ODLayer params
        ODLAYER_PARAMS = {
            "ODLAYER_CONNECTION_DIST": 1e-3,  # m

            "MAX_TRANSFER_DIST": 1000,

            "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
            "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
            "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
            "MAX_ACCESS_EGRESS_DIST_PT": 1,
        }

        ## Traffic params
        TRAFFIC_PARAMS = {
            "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
            "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
            "EXPECTED_PT_SHARE": 1
        }


        def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
            Pc = mfd_params['Pc']
            nc = mfd_params['nc']
            njam = mfd_params['njam']
            n = max(1, dacc['CAR'] + dacc['BUS'])
            p = 0
            if n <= nc:
                p = Pc * n * (2 * nc - n) / nc ** 2
            elif n > nc and n < njam:
                p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
            V = p / max(n, 0.00001)
            V = max(V, 0.001)  # min speed to avoid gridlock
            return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2}  # 1,43


        ## Mobility services params
        MOB_SERVICES_PARAMS = {
            "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                         "TRAM": fake_lyon_tram_stops},

            "PT_START_RNG": np.random.default_rng(seed=0),
            "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

            "BUS_CAPA": 5,  # 50
            "BUS_FREQUENCY": Dt(minutes=15),
            "BUS_DEFAULT_SPEED": 8.4,  # m/s

            "TRAM_CAPA": 10,  # 100
            "TRAM_FREQUENCY": Dt(minutes=15),
            "TRAM_DEFAULT_SPEED": 13.9,

            "WALK_DEFAULT_SPEED": 1.2,
            "BIKE_DEFAULT_SPEED": 4,

            "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
        }

        for tau in [30, 60, 120, 300, 600]:
            ## Static(_distrib) parameters
            ESTIM_METHODS_PARAMS = {
                "TAU": Dt(seconds=tau),
                "ZONE_SIZE": 1000,
                "LOAD_THRESHOLD": 1
            }

            for n_users in [1300]:  # [200, 1300, 2600]:
                DEMAND_PARAMS = {"N_USERS": n_users,
                                 "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                                 "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                                 "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                                 "T_MU": "08:00:00.00",  # mean/peak of departure times
                                 "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                                 "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                                 "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                                 "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                                 "DISTANCE_FACTOR": 1.1,
                                 "EPSILON": 50,
                                 "DEMAND_SHAPE": 'multicentric'
                                 }
                DEMAND_PARAMS[
                    "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

                ## Other
                OTHER_SPECIFIER = f'_TAU={tau}sec'

                for estim_method in ['static']:
                    manhattan(simu_p=SIMU_PARAMS,
                              grid_p=GRID_PARAMS,
                              odlayer_p=ODLAYER_PARAMS,
                              traffic_p=TRAFFIC_PARAMS,
                              mfdspeedfunc=mfdspeed,
                              mobserv_p=MOB_SERVICES_PARAMS,
                              demand_p=DEMAND_PARAMS,
                              estim_p=ESTIM_METHODS_PARAMS,
                              other_specifier=OTHER_SPECIFIER,
                              estim_method=estim_method,
                              seed=0)

    ### LYON - WITH CAR TRAFFIC ###
    if car_traffic:
        SCENARIO = f"Lyon"

        ## Simu params
        SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                       "DT_FLOW": Dt(minutes=1),
                       "AFFECTATION_FACTOR": 5
                       }

        ## Grid params
        GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
        GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
        GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

        ## ODLayer params
        ODLAYER_PARAMS = {
            "ODLAYER_CONNECTION_DIST": 1e-3,  # m

            "MAX_TRANSFER_DIST": 1000,

            "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
            "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
            "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
            "MAX_ACCESS_EGRESS_DIST_PT": 1,
        }

        ## Traffic params
        TRAFFIC_PARAMS = {
            "N_CARS": lambda x: ghost_acc(x),  # number of cars in the zone, to create congestion
            "MFD_PARAMS": {'Pc': 18000, 'nc': 3000, 'njam': 8000},
            "EXPECTED_PT_SHARE": 1
        }

        def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
            Pc = mfd_params['Pc']
            nc = mfd_params['nc']
            njam = mfd_params['njam']
            n = max(1, dacc['CAR'] + dacc['BUS'])
            p = 0
            if n <= nc:
                p = Pc * n * (2 * nc - n) / nc ** 2
            elif n > nc and n < njam:
                p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
            V = p / max(n, 0.00001)
            V = max(V, 0.001)  # min speed to avoid gridlock
            return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2}  # 1,43

        ## Mobility services params
        MOB_SERVICES_PARAMS = {
            "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                         "TRAM": fake_lyon_tram_stops},

            "PT_START_RNG": np.random.default_rng(seed=0),
            "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

            "BUS_CAPA": 5,  # 50
            "BUS_FREQUENCY": Dt(minutes=15),
            "BUS_DEFAULT_SPEED": 8.4,  # m/s

            "TRAM_CAPA": 10,  # 100
            "TRAM_FREQUENCY": Dt(minutes=15),
            "TRAM_DEFAULT_SPEED": 13.9,

            "WALK_DEFAULT_SPEED": 1.2,
            "BIKE_DEFAULT_SPEED": 4,

            "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
        }

        ## Static(_distrib) parameters
        ESTIM_METHODS_PARAMS = {
            "TAU": Dt(minutes=2),
            "ZONE_SIZE": 1000,
            "LOAD_THRESHOLD": 1 #0.8
        }

        for n_users in [1300]:
            DEMAND_PARAMS = {"N_USERS": n_users,
                             "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                             "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                             "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                             "T_MU": "08:00:00.00",  # mean/peak of departure times
                             "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                             "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                             "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                             "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                             "DISTANCE_FACTOR": 1.1,
                             "EPSILON": 50,
                             "DEMAND_SHAPE": 'multicentric'
                             }
            DEMAND_PARAMS[
                "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

            ## Other
            OTHER_SPECIFIER = '_fluctcong_23janv'

            for estim_method in ['none', 'static', 'dynamic']:
                manhattan(simu_p=SIMU_PARAMS,
                          grid_p=GRID_PARAMS,
                          odlayer_p=ODLAYER_PARAMS,
                          traffic_p=TRAFFIC_PARAMS,
                          mfdspeedfunc=mfdspeed,
                          mobserv_p=MOB_SERVICES_PARAMS,
                          demand_p=DEMAND_PARAMS,
                          estim_p=ESTIM_METHODS_PARAMS,
                          other_specifier=OTHER_SPECIFIER,
                          estim_method=estim_method,
                          seed=0)

    ### LYON - ALL METHODS - SHARP PEAK HOUR ###
    if sharp_peak:
        SCENARIO = "Lyon"

        ## Simu params
        SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                       "DT_FLOW": Dt(minutes=1),
                       "AFFECTATION_FACTOR": 5
                       }

        ## Grid params
        GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
        GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
        GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

        ## ODLayer params
        ODLAYER_PARAMS = {
            "ODLAYER_CONNECTION_DIST": 1e-3,  # m

            "MAX_TRANSFER_DIST": 1000,

            "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
            "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
            "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
            "MAX_ACCESS_EGRESS_DIST_PT": 1,
        }

        ## Traffic params
        TRAFFIC_PARAMS = {
            "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
            "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
            "EXPECTED_PT_SHARE": 1
        }


        def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
            Pc = mfd_params['Pc']
            nc = mfd_params['nc']
            njam = mfd_params['njam']
            n = max(1, dacc['CAR'] + dacc['BUS'])
            p = 0
            if n <= nc:
                p = Pc * n * (2 * nc - n) / nc ** 2
            elif n > nc and n < njam:
                p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
            V = p / max(n, 0.00001)
            V = max(V, 0.001)  # min speed to avoid gridlock
            return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2}  # 1,43


        ## Mobility services params
        MOB_SERVICES_PARAMS = {
            "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                         "TRAM": fake_lyon_tram_stops},

            "PT_START_RNG": np.random.default_rng(seed=0),
            "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

            "BUS_CAPA": 5,  # 50
            "BUS_FREQUENCY": Dt(minutes=15),
            "BUS_DEFAULT_SPEED": 8.4,  # m/s

            "TRAM_CAPA": 10,  # 100
            "TRAM_FREQUENCY": Dt(minutes=15),
            "TRAM_DEFAULT_SPEED": 13.9,

            "WALK_DEFAULT_SPEED": 1.2,
            "BIKE_DEFAULT_SPEED": 4,

            "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
        }

        ESTIM_METHODS_PARAMS = {
            "TAU": Dt(minutes=2),
            "ZONE_SIZE": 1000,
            "LOAD_THRESHOLD": 1
        }

        for n_users in [200, 1300, 2600]:

            DEMAND_PARAMS = {"N_USERS": n_users,
                             "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                             "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                             "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                             "T_MU": "08:00:00.00",  # mean/peak of departure times
                             "T_SIGMA": 0.25 * 3600,  # std dev of departure times
                             "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                             "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                             "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                             "DISTANCE_FACTOR": 1.1,
                             "EPSILON": 50,
                             "DEMAND_SHAPE": 'multicentric'
                             }
            DEMAND_PARAMS[
                "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

            ## Other
            OTHER_SPECIFIER = f'_sharppeak_23janv'

            for estim_method in ['none', 'static', 'dynamic']:
                manhattan(simu_p=SIMU_PARAMS,
                          grid_p=GRID_PARAMS,
                          odlayer_p=ODLAYER_PARAMS,
                          traffic_p=TRAFFIC_PARAMS,
                          mfdspeedfunc=mfdspeed,
                          mobserv_p=MOB_SERVICES_PARAMS,
                          demand_p=DEMAND_PARAMS,
                          estim_p=ESTIM_METHODS_PARAMS,
                          other_specifier=OTHER_SPECIFIER,
                          estim_method=estim_method,
                          seed=0)

    ### LYON - DYNAMIC - NO PT CONGESTION, LOAD THRESHOLD EQUAL TO 100% ###
    """SCENARIO = "Lyon"

    ## Simu params
    SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                   "DT_FLOW": Dt(minutes=1),
                   "AFFECTATION_FACTOR": 5
                   }

    ## Grid params
    GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
    GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
    GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

    ## ODLayer params
    ODLAYER_PARAMS = {
        "ODLAYER_CONNECTION_DIST": 1e-3,  # m

        "MAX_TRANSFER_DIST": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
        "MAX_ACCESS_EGRESS_DIST_PT": 1,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
        "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
        "EXPECTED_PT_SHARE": 1
    }

    def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
        Pc = mfd_params['Pc']
        nc = mfd_params['nc']
        njam = mfd_params['njam']
        n = max(1, dacc['CAR'] + dacc['BUS'])
        p = 0
        if n <= nc:
            p = Pc * n * (2 * nc - n) / nc ** 2
        elif n > nc and n < njam:
            p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
        V = p / max(n, 0.00001)
        V = max(V, 0.001)  # min speed to avoid gridlock
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2} # 1,43


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                     "TRAM": fake_lyon_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.2,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=2),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 1
    }

    ## Other
    OTHER_SPECIFIER = f'_LT={ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"]}'

    for n_users in [200]:#[800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)], # perrache, doua, villeurbanne
                         "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                         "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                         "DISTANCE_FACTOR": 1.1,
                         "EPSILON": 50,
                         "DEMAND_SHAPE": 'multicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

        for estim_method in ['dynamic']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=estim_method,
                      seed=0)"""

    ### LYON - DYNAMIC - HIGH PT CONGESTION, LOAD THRESHOLD EQUAL TO 60% ###
    """SCENARIO = "Lyon"

    ## Simu params
    SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                   "DT_FLOW": Dt(minutes=1),
                   "AFFECTATION_FACTOR": 5
                   }

    ## Grid params
    GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
    GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
    GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

    ## ODLayer params
    ODLAYER_PARAMS = {
        "ODLAYER_CONNECTION_DIST": 1e-3,  # m

        "MAX_TRANSFER_DIST": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
        "MAX_ACCESS_EGRESS_DIST_PT": 1,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
        "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
        "EXPECTED_PT_SHARE": 1
    }


    def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
        Pc = mfd_params['Pc']
        nc = mfd_params['nc']
        njam = mfd_params['njam']
        n = max(1, dacc['CAR'] + dacc['BUS'])
        p = 0
        if n <= nc:
            p = Pc * n * (2 * nc - n) / nc ** 2
        elif n > nc and n < njam:
            p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
        V = p / max(n, 0.00001)
        V = max(V, 0.001)  # min speed to avoid gridlock
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2}  # 1,43


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                     "TRAM": fake_lyon_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.2,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=2),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.6
    }

    ## Other
    OTHER_SPECIFIER = f'_LT={ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"]}'

    for n_users in [1300]:  # [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                         "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                         "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                         "DISTANCE_FACTOR": 1.1,
                         "EPSILON": 50,
                         "DEMAND_SHAPE": 'multicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

        for estim_method in ['dynamic']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=estim_method,
                      seed=0)"""

    ### LYON - STATIC - NO PT CONGESTION, TEST NEW STATIC METHOD ###
    """SCENARIO = "Lyon"

    ## Simu params
    SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                   "DT_FLOW": Dt(minutes=1),
                   "AFFECTATION_FACTOR": 5
                   }

    ## Grid params
    GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
    GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
    GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

    ## ODLayer params
    ODLAYER_PARAMS = {
        "ODLAYER_CONNECTION_DIST": 1e-3,  # m

        "MAX_TRANSFER_DIST": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
        "MAX_ACCESS_EGRESS_DIST_PT": 1,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
        "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
        "EXPECTED_PT_SHARE": 1
    }

    def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
        Pc = mfd_params['Pc']
        nc = mfd_params['nc']
        njam = mfd_params['njam']
        n = max(1, dacc['CAR'] + dacc['BUS'])
        p = 0
        if n <= nc:
            p = Pc * n * (2 * nc - n) / nc ** 2
        elif n > nc and n < njam:
            p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
        V = p / max(n, 0.00001)
        V = max(V, 0.001)  # min speed to avoid gridlock
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2} # 1,43


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                     "TRAM": fake_lyon_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.2,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 9  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=2),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": None
    }

    ## Other
    OTHER_SPECIFIER = f'_14janv_ptspeed=9'

    for n_users in [1300]:#[800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)], # perrache, doua, villeurbanne
                         "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                         "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                         "DISTANCE_FACTOR": 1.1,
                         "EPSILON": 50,
                         "DEMAND_SHAPE": 'multicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

        for estim_method in ['static']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=estim_method,
                      seed=0)"""

    ### LYON - DYNAMIC - LT TEST EXHAUSTIVE - TC CAPA x2 ###
    if exhaustive_lt:
        SCENARIO = "Lyon"

        ## Simu params
        SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"),  # simu start and end times
                       "DT_FLOW": Dt(minutes=1),
                       "AFFECTATION_FACTOR": 5
                       }

        ## Grid params
        GRID_PARAMS = {"NODES_PER_DIR": 7, "MESH_SIZE": 1000}
        GRID_PARAMS["X_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])
        GRID_PARAMS["Y_BOUNDARIES"] = (0, (GRID_PARAMS["NODES_PER_DIR"] - 1) * GRID_PARAMS["MESH_SIZE"])

        ## ODLayer params
        ODLAYER_PARAMS = {
            "ODLAYER_CONNECTION_DIST": 1e-3,  # m

            "MAX_TRANSFER_DIST": 1000,

            "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 1,
            "MAX_ACCESS_EGRESS_DIST_WALKING": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
            "MAX_ACCESS_EGRESS_DIST_BIKE": 1,

            "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1,
            "MAX_ACCESS_EGRESS_DIST_PT": 1,
        }

        ## Traffic params
        TRAFFIC_PARAMS = {
            "N_CARS": lambda x: {"CAR": 0},  # number of cars in the zone, to create congestion
            "MFD_PARAMS": {'Pc': 18000, 'nc': 2500, 'njam': 5000},
            "EXPECTED_PT_SHARE": 1
        }


        def mfdspeed(dacc, mfd_params=TRAFFIC_PARAMS["MFD_PARAMS"]):
            Pc = mfd_params['Pc']
            nc = mfd_params['nc']
            njam = mfd_params['njam']
            n = max(1, dacc['CAR'] + dacc['BUS'])
            p = 0
            if n <= nc:
                p = Pc * n * (2 * nc - n) / nc ** 2
            elif n > nc and n < njam:
                p = Pc * (njam - n) * (njam + n - 2 * nc) / (njam - nc) ** 2
            V = p / max(n, 0.00001)
            V = max(V, 0.001)  # min speed to avoid gridlock
            return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.2}  # 1,43


        ## Mobility services params
        MOB_SERVICES_PARAMS = {
            "PT_STOPS": {"BUS": fake_lyon_bus_stops,
                         "TRAM": fake_lyon_tram_stops},

            "PT_START_RNG": np.random.default_rng(seed=0),
            "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

            "BUS_CAPA": 10,  # 50
            "BUS_FREQUENCY": Dt(minutes=15),
            "BUS_DEFAULT_SPEED": 8.4,  # m/s

            "TRAM_CAPA": 20,  # 100
            "TRAM_FREQUENCY": Dt(minutes=15),
            "TRAM_DEFAULT_SPEED": 13.9,

            "WALK_DEFAULT_SPEED": 1.2,
            "BIKE_DEFAULT_SPEED": 4,

            "AVG_PT_SPEED": 13  # (8.4*8+13.9*4)/(8+4)
        }

        for lt in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            ## Static(_distrib) parameters
            ESTIM_METHODS_PARAMS = {
                "TAU": None,
                "ZONE_SIZE": 1000,
                "LOAD_THRESHOLD": lt
            }

            ## Other
            OTHER_SPECIFIER = f'_exhaustive_test_LT={ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"]}'

            for n_users in [200, 800, 1300, 2600, 3000]:  # [800, 1300, 2600]:
                DEMAND_PARAMS = {"N_USERS": n_users,
                                 "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                                 "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                                 "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                                 "T_MU": "08:00:00.00",  # mean/peak of departure times
                                 "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                                 "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                                 "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                                 "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                                 "DISTANCE_FACTOR": 1.1,
                                 "EPSILON": 50,
                                 "DEMAND_SHAPE": 'multicentric'
                                 }
                DEMAND_PARAMS[
                    "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min_epsilon={int(DEMAND_PARAMS["EPSILON"])}'

                for estim_method in ['dynamic']:
                    manhattan(simu_p=SIMU_PARAMS,
                              grid_p=GRID_PARAMS,
                              odlayer_p=ODLAYER_PARAMS,
                              traffic_p=TRAFFIC_PARAMS,
                              mfdspeedfunc=mfdspeed,
                              mobserv_p=MOB_SERVICES_PARAMS,
                              demand_p=DEMAND_PARAMS,
                              estim_p=ESTIM_METHODS_PARAMS,
                              other_specifier=OTHER_SPECIFIER,
                              estim_method=estim_method,
                              seed=0)