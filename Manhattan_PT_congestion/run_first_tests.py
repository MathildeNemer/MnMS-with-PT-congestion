from manhattan import manhattan
from mnms.time import Dt
import numpy as np

# Hypercenter original
hypercenter_bus_stops_tilted = {"verticalA0": [2, 16, 30, 44],
                         "verticalA1": [2, 16, 30, 44][::-1],
                         "verticalB0": [4, 18, 32, 46],
                         "verticalB1": [4, 18, 32, 46][::-1],
                         "horizontalA0": [14, 16, 18, 20],
                         "horizontalA1": [14, 16, 18, 20][::-1],
                         "horizontalB0": [28, 30, 32, 34],
                         "horizontalB1": [28, 30, 32, 34][::-1],
                         }
hypercenter_tram_stops_tilted = {"diagonalA0": [0, 16, 24, 32, 48],
                          "diagonalB0": [6, 18, 24, 30, 42],
                          "diagonalA1": [0, 16, 24, 32, 48][::-1],
                          "diagonalB1": [6, 18, 24, 30, 42][::-1]
                          }

# Hypercenter bus concentriques
concentric_bus_stops_tilted = {"outside0": [8, 10, 12, 26, 40, 38, 36, 22, 8],
                        "outside1": [8, 10, 12, 26, 40, 38, 36, 22, 8][::-1],
                        "inside0": [16, 18, 32, 30],
                        "inside1": [16, 18, 32, 30][::-1]}
concentric_tram_stops_tilted = {"diagonalA0": [0, 16, 24, 32, 48],
                          "diagonalB0": [6, 18, 24, 30, 42],
                          "diagonalA1": [0, 16, 24, 32, 48][::-1],
                          "diagonalB1": [6, 18, 24, 30, 42][::-1]
                          }

# Fake Lyon network
fake_lyon_bus_stops_tilted = {"t40": [3, 10, 24, 31, 38, 45],
                       "t41": [45, 38, 31, 24, 10, 3],
                       "t10": [22, 31, 38, 45],
                       "t11": [45, 38, 31, 22],
                       "t30": [27, 25, 24, 31],
                       "t31": [31, 24, 25, 27],
                       "t20": [22, 8, 10, 11, 13, 20],
                       "t21": [20, 13, 11, 10, 8, 22]
                       }
fake_lyon_tram_stops_tilted = {"A0": [22, 36, 38, 39, 41],
                        "A1": [41, 39, 38, 36, 22],
                        "D0": [5, 10, 16, 22, 28],
                        "D1": [28, 22, 16, 10, 5],
                        "B0": [2, 16, 31, 38],
                        "B1": [38, 31, 16, 2]
                        }

# Fake Barcelona network
fake_barcelona_tram_stops_tilted = {'green0': [28, 21, 7, 8, 16, 23, 37, 44],
                             'green1': [44, 37, 23, 16, 8, 7, 21, 28],
                             'red0': [21, 16, 17, 18, 27],
                             'red1': [27, 18, 17, 16, 21]
                             }
fake_barcelona_bus_stops_tilted = {'purple0': [8, 16, 17, 18, 20],
                            'purple1': [20, 18, 17, 16, 8],
                            'yellow0': [13, 11, 17, 25, 27],
                            'yellow1': [27, 25, 17, 11, 13],
                            'blue0': [46, 39, 41, 27, 25, 23, 21],
                            'blue1': [21, 23, 25, 27, 41, 39, 46]
                            }

def row_to_col_major(i, n):
    r = i // n  # ligne
    c = i % n   # colonne
    return c * n + r

def row_to_col_dict(dict, nodes_per_dir):
    new_dict = {}
    for line in dict:
        new_nodes = []
        for node in dict[line]:
            new_nodes.append(row_to_col_major(node, nodes_per_dir))
        new_dict[line] = new_nodes
    return new_dict

if __name__ == '__main__':

    ### ALL METHODS - EFFECTS OF PT CONGESTION NO CAR ###
    """
    SCENARIO = "PT_congestion_no_car"

    ## Simu params
    SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "12:00:00.00"), # simu start and end times
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

        "SECURE_MAX_ACCESS_EGRESS_DIST_WALKING": 7000,
        "MAX_ACCESS_EGRESS_DIST_WALKING": 7000,

        "SECURE_MAX_ACCESS_EGRESS_BIKE_DIST": 7000,
        "MAX_ACCESS_EGRESS_BIKE_DIST": 7000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_ACCESS_EGRESS_DIST_PT": 7000,
    }


    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}

    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00", # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600, # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS["DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none', 'dynamic', 'static', 'static_distrib']:
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
    """

    ### STATIC - EFFECTS OF ZONE SIZE ###
    """
    SCENARIO = "zone_size"

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
        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_TRANSFER_DIST": 7000
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "LOAD_THRESHOLD": 0.8
    }


    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        # Run static and static_distrib methods for all zone sizes
        for zone_size in [500, 1000, 2000]:
            ESTIM_METHODS_PARAMS["ZONE_SIZE"] = zone_size
            OTHER_SPECIFIER = f'_ZS={zone_size}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static',
                      seed=0)

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static_distrib',
                      seed=0)

        # Run 'none' method once per n_users with no specifier bc no zone size to consider
        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """

    ### DYNAMIC - EFFECTS OF LOAD THRESHOLD ###
    """
    SCENARIO = "load_threshold"

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
        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_TRANSFER_DIST": 7000
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000
    }

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for load_threshold in [0.3, 0.5, 0.8, 1]:
            ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"] = load_threshold
            OTHER_SPECIFIER = f'_LT={load_threshold}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='dynamic',
                      seed=0)

        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """

    ### ALL METHODS - SHARP PEAK HOUR  ###
    # std dev of departures distrib is 15 min instead of 30 min
    """
    SCENARIO = "sharp_peak_hour"

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
        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_TRANSFER_DIST": 7000
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.25 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none', 'dynamic', 'static', 'static_distrib']:
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
    """

    ### NO WALKING SCENARIO ###
    # to be sure that the methods really treat congestion
    """
    SCENARIO = "no_walking_allowed"

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
        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_ACCESS_EGRESS_DIST_PT": 7000,
        "MAX_TRANSFER_DIST": 7000
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}

    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for est in ['none', 'dynamic', 'static', 'static_distrib']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=est,
                      seed=0)
    """

    ### TEST NEW NETWORKS ###
    """
    SCENARIO = "TEST_CONCENTRIC_NTK"

    ## Simu params
    SIMU_PARAMS = {"SIMU_T_BOUNDARIES": ("06:55:00.00", "08:00:00.00"),  # simu start and end times
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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'CAR': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {'BUS': concentric_bus_stops,
                     'TRAM': concentric_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=1),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [1]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "07:01:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "07:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.25 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

    for est in ['none']:
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method=est,
                  seed=0)
    """

    ### V2 ALL METHODS - EFFECTS OF PT CONGESTION NO CAR ###
    """
    SCENARIO = "V2_PT_congestion_no_car"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none', 'dynamic', 'static', 'static_distrib']:
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

    ### V2 STATIC - EFFECTS OF ZONE SIZE ###
    """
    SCENARIO = "V2_zone_size"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "LOAD_THRESHOLD": 0.8
    }

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        # Run static and static_distrib methods for all zone sizes
        for zone_size in [500, 1000, 2000]:
            ESTIM_METHODS_PARAMS["ZONE_SIZE"] = zone_size
            OTHER_SPECIFIER = f'_ZS={zone_size}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static',
                      seed=0)

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static_distrib',
                      seed=0)

        # Run 'none' method once per n_users with no specifier bc no zone size to consider
        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """

    ### V2 DYNAMIC - EFFECTS OF LOAD THRESHOLD ###
    """
    SCENARIO = "V2_load_threshold"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {'BUS': hypercenter_bus_stops,
                     'TRAM': hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000
    }

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for load_threshold in [0.3, 0.5, 0.8, 1]:
            ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"] = load_threshold
            OTHER_SPECIFIER = f'_LT={load_threshold}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='dynamic',
                      seed=0)

        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """

    ### V2 ALL METHODS - SHARP PEAK HOUR  ###
    # std dev of departures distrib is 15 min instead of 30 min
    """
    SCENARIO = "V2_sharp_peak_hour"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.25 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none', 'dynamic', 'static', 'static_distrib']:
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
    """

    ### V2 NO WALKING SCENARIO ###
    # to be sure that the methods really treat congestion
    """
    SCENARIO = "V2_very_slow_walking"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 0.001}  # here


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for est in ['none', 'dynamic', 'static', 'static_distrib']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=est,
                      seed=0)
        """

    #################################################################
    #                      FAKE LYON NETWORK                        #
    #################################################################

    ### LYON - ALL METHODS - EFFECTS OF PT CONGESTION NO CAR ###
    SCENARIO = "Lyon-PT_congestion_no_car_walkspeed=1,2"

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
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        "PT_STOPS": {"BUS": row_to_col_dict(fake_lyon_bus_stops_tilted, GRID_PARAMS["NODES_PER_DIR"]),
                     "TRAM": row_to_col_dict(fake_lyon_tram_stops_tilted, GRID_PARAMS["NODES_PER_DIR"])},

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

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [2600]:
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
                         "DEMAND_SHAPE": 'multicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none']:
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

    ### LYON - STATIC - EFFECTS OF ZONE SIZE ###
    """
    SCENARIO = "V2_zone_size"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "LOAD_THRESHOLD": 0.8
    }

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        # Run static and static_distrib methods for all zone sizes
        for zone_size in [500, 1000, 2000]:
            ESTIM_METHODS_PARAMS["ZONE_SIZE"] = zone_size
            OTHER_SPECIFIER = f'_ZS={zone_size}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static',
                      seed=0)

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='static_distrib',
                      seed=0)

        # Run 'none' method once per n_users with no specifier bc no zone size to consider
        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """ # TODO

    ### LYON - DYNAMIC - EFFECTS OF LOAD THRESHOLD ###
    """
    SCENARIO = "V2_load_threshold"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {'BUS': hypercenter_bus_stops,
                     'TRAM': hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000
    }

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for load_threshold in [0.3, 0.5, 0.8, 1]:
            ESTIM_METHODS_PARAMS["LOAD_THRESHOLD"] = load_threshold
            OTHER_SPECIFIER = f'_LT={load_threshold}'

            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method='dynamic',
                      seed=0)

        OTHER_SPECIFIER = ''
        manhattan(simu_p=SIMU_PARAMS,
                  grid_p=GRID_PARAMS,
                  odlayer_p=ODLAYER_PARAMS,
                  traffic_p=TRAFFIC_PARAMS,
                  mfdspeedfunc=mfdspeed,
                  mobserv_p=MOB_SERVICES_PARAMS,
                  demand_p=DEMAND_PARAMS,
                  estim_p=ESTIM_METHODS_PARAMS,
                  other_specifier=OTHER_SPECIFIER,
                  estim_method='none',
                  seed=0)
    """ # TODO

    ### LYON - ALL METHODS - SHARP PEAK HOUR  ###
    # std dev of departures distrib is 15 min instead of 30 min
    """
    SCENARIO = "V2_sharp_peak_hour"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 1.43}


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.25 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none', 'dynamic', 'static', 'static_distrib']:
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
    """ # TODO

    ### LYON - NO WALKING SCENARIO ###
    # to be sure that the methods really treat congestion
    """
    SCENARIO = "V2_very_slow_walking"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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
        return {'BUS': V, 'BIKE': 4, 'TRAM': 13.9, 'WALKING': 0.001}  # here


    ## Mobility services params
    MOB_SERVICES_PARAMS = {
        "PT_STOPS": {"BUS": hypercenter_bus_stops,
                     "TRAM": hypercenter_tram_stops},

        "PT_START_RNG": np.random.default_rng(seed=0),
        "PT_END": SIMU_PARAMS["SIMU_T_BOUNDARIES"][-1],

        "BUS_CAPA": 5,  # 50
        "BUS_FREQUENCY": Dt(minutes=15),
        "BUS_DEFAULT_SPEED": 8.4,  # m/s

        "TRAM_CAPA": 10,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [800, 1300, 2600]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "11:00:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "08:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU": (GRID_PARAMS["X_BOUNDARIES"][-1] / 2, GRID_PARAMS["Y_BOUNDARIES"][-1] / 2),
                         "D_SIGMA": (200, 200),
                         "D_MU_LIST": [(1500, 1500), (500, 500)],  # spatial epicenter of the destinations
                         "D_SIGMA_LIST": [(150, 150), (150, 150)],  # spatial std dev of the destinations
                         "WEIGHTS_LIST": [0.8, 0.2],
                         "DISTANCE_FACTOR": 1.01,
                         "DEMAND_SHAPE": 'unicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for est in ['none', 'dynamic', 'static', 'static_distrib']:
            manhattan(simu_p=SIMU_PARAMS,
                      grid_p=GRID_PARAMS,
                      odlayer_p=ODLAYER_PARAMS,
                      traffic_p=TRAFFIC_PARAMS,
                      mfdspeedfunc=mfdspeed,
                      mobserv_p=MOB_SERVICES_PARAMS,
                      demand_p=DEMAND_PARAMS,
                      estim_p=ESTIM_METHODS_PARAMS,
                      other_specifier=OTHER_SPECIFIER,
                      estim_method=est,
                      seed=0)
        """ # TODO

    ### LYON - TEST CORRECTION FIFO ###
    """
    SCENARIO = "Lyon-test_correction_FIFO"

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
        "MAX_ACCESS_EGRESS_DIST_WALKING": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_BIKE": 1,
        "MAX_ACCESS_EGRESS_DIST_BIKE": 1000,

        "SECURE_MAX_ACCESS_EGRESS_DIST_PT": 1001,
        "MAX_ACCESS_EGRESS_DIST_PT": 1001,
    }

    ## Traffic params
    TRAFFIC_PARAMS = {
        "N_CARS": 0,  # number of cars in the zone, to create congestion
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

        "TRAM_CAPA": 2,  # 100
        "TRAM_FREQUENCY": Dt(minutes=15),
        "TRAM_DEFAULT_SPEED": 13.9,

        "WALK_DEFAULT_SPEED": 1.43,
        "BIKE_DEFAULT_SPEED": 4,

        "AVG_PT_SPEED": 10.2  # (8.4*8+13.9*4)/(8+4)
    }

    ## Static(_distrib) parameters
    ESTIM_METHODS_PARAMS = {
        "TAU": Dt(minutes=10),
        "ZONE_SIZE": 1000,
        "LOAD_THRESHOLD": 0.8
    }

    ## Other
    OTHER_SPECIFIER = ''

    for n_users in [3]:
        DEMAND_PARAMS = {"N_USERS": n_users,
                         "DEMAND_T_BOUNDARIES": ("07:00:00.00", "07:01:00.00"),
                         "X_BOUNDARIES": GRID_PARAMS["X_BOUNDARIES"],
                         "Y_BOUNDARIES": GRID_PARAMS["Y_BOUNDARIES"],
                         "T_MU": "07:00:00.00",  # mean/peak of departure times
                         "T_SIGMA": 0.5 * 3600,  # std dev of departure times
                         "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
                         "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
                         "WEIGHTS_LIST": [0.3, 0.3, 0.4],
                         "DISTANCE_FACTOR": 1.1,
                         "DEMAND_SHAPE": 'multicentric'
                         }
        DEMAND_PARAMS[
            "DEMAND_SPECIFIER"] = f'{SCENARIO}_{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][0][:2]}-{DEMAND_PARAMS["DEMAND_T_BOUNDARIES"][-1][:2]}_peakhour={DEMAND_PARAMS["T_MU"][:5]}-{int(DEMAND_PARAMS["T_SIGMA"] // 60)}min'

        for estim_method in ['none']:
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
    """
