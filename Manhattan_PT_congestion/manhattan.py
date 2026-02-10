import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from generate_manhattan_demands import *
from Static_zone_computation import *

## MnMS
from build.lib.mnms.graph.layers import PublicTransportLayer
from mnms.generation.roads import generate_manhattan_road
from mnms.generation.layers import generate_layer_from_roads, generate_matching_origin_destination_layer
from mnms.graph.layers import MultiLayerGraph, BikeLayer, SimpleLayer
from mnms.graph.zone import Zone
from mnms.demand.manager import CSVDemandManager
from mnms.log import set_all_mnms_logger_level, LOGLEVEL
from mnms.travel_decision.logit import ModeCentricLogitDecisionModel
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.flow.MFD import MFDFlowMotor, Reservoir
from mnms.simulation import Supervisor
from mnms.time import Time, Dt, TimeTable
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.vehicles.veh_type import Bus, Bike, Tram, Walking
from mnms.io.graph import save_graph, save_transit_links

def add_mservice_lines(roads, mservice_anchor_nodes, mservice_layer, mservice_freq, pt_start_rng, simu_start: Time, pt_end: str, separate_zone=False):

    existing_mservice_sections = []
    existing_mservices_nodes = []

    for line in mservice_anchor_nodes:
        line_sections = []

        # 1. Create nodes based on desired stop positions
        stop_ids = []
        stop_nodes = []
        for anchor_node in mservice_anchor_nodes[line]:
            stop_id = f'{line}-{anchor_node}'
            if stop_id in stop_ids: # only works if anchor node is max twice in mservice_anchor_nodes[line]
                stop_id += '-bis'
            stop_ids.append(stop_id)

            stop_node_id = stop_id+'_node'
            stop_nodes.append(stop_node_id)
            existing_mservices_nodes.append(stop_node_id)

            roads.register_node(stop_node_id, pos=roads.nodes[str(anchor_node)].position)

        # 2. Create sections starting from each node except las one
        for idx, current_stop_node in enumerate(stop_nodes[:-1]):
            next_stop_node = stop_nodes[idx + 1]

            section_id = f'{current_stop_node}_{next_stop_node}'
            line_sections.append(section_id)
            if section_id not in existing_mservice_sections:
                roads.register_section(lid=section_id, upstream=current_stop_node, downstream=next_stop_node)
                roads.sections[section_id].zone = 'RES' # pas joli mais comment faire ??
                roads.zones['RES'].sections.add(section_id) # add section to zone
                existing_mservice_sections.append(section_id)

        # 3. Create stops on sections and append
        for idx, current_stop_node in enumerate(stop_nodes[:-1]):
            stop_id = stop_ids[idx]
            stop_section = line_sections[idx]
            roads.register_stop(stop_id, stop_section, 0)

        # Register last node of the line using last section id and relative position 1
        idx = -1
        stop_id = stop_ids[idx]
        stop_section = line_sections[idx]
        roads.register_stop(stop_id, stop_section, 1)

        sec_list_list = []
        # 4. Create sec_list_list
        for idx, section in enumerate(line_sections[:-1]):
            sec_list_list.append([section, line_sections[idx+1]])
        sec_list_list.append([line_sections[-1]])

        # Create line
        delay = pt_start_rng.integers(0, mservice_freq._minutes)
        line_start = simu_start.add_time(Dt(minutes=delay))
        mservice_layer.create_line(line,
                                   stop_ids,
                                   sec_list_list,
                                   timetable=TimeTable.create_table_freq(line_start.__str__(), pt_end, mservice_freq),
                                   bidirectional=False)

    return existing_mservice_sections, existing_mservices_nodes

def manhattan(simu_p, grid_p, odlayer_p, traffic_p, mfdspeedfunc, mobserv_p,
              demand_p, estim_p, estim_method='none', other_specifier='', seed=0):

    n_users = demand_p["N_USERS"]
    demand_shape = demand_p["DEMAND_SHAPE"]

    demand_specifier = demand_p["DEMAND_SPECIFIER"]

    print(f"PREPARING SIMULATION {demand_shape} - {estim_method} - {n_users} - {demand_specifier} - {other_specifier}")

    ## Input and Output files
    insubname = f'INPUTS/{demand_specifier}/{n_users}users_{estim_method}{other_specifier}'
    outsubname = f'OUTPUTS/{demand_specifier}/{n_users}users_{estim_method}{other_specifier}'
    os.makedirs(outsubname, exist_ok=True)
    os.makedirs(insubname, exist_ok=True)

    demand_file = f'INPUTS/{demand_specifier}/Demand_{n_users}users.csv'
    network_file = f'{insubname}/Network.json'
    transit_file = f'{insubname}/Transit.json'

    log_file = f'{outsubname}/sim.log'
    paths_file = f'{outsubname}/Paths.csv'
    user_obs_file = f'{outsubname}/Users.csv'
    pt_obs_file = f'{outsubname}/PT_vehs.csv'
    bike_obs_file = f'{outsubname}/Bike_vehs.csv'
    walk_obs_file = f'{outsubname}/Walking.csv'

    if estim_method == 'static':
        pt_load_file = f'{insubname}/Headway_factors_stat.json'
    else:
        pt_load_file = None

    #### Demand ####
    if not os.path.exists(demand_file):
        if demand_shape == 'uniform':
            generate_uniform_demand(demand_p, demand_file)
        elif demand_shape == 'unicentric':
            generate_unicentric_demand(demand_p, demand_file)
        elif demand_shape == 'multicentric':
            generate_multicentric_demand(demand_p, demand_file, pt_dict=mobserv_p["PT_STOPS"], n_nodes=grid_p["NODES_PER_DIR"])

        plot_distribs(demand_file)

    demand = CSVDemandManager(demand_file)
    demand.add_user_observer(CSVUserObserver(user_obs_file))

    #### RoadDescriptor ####
    roads = generate_manhattan_road(grid_p["NODES_PER_DIR"], grid_p["MESH_SIZE"])

    #### ODLayer ####
    odlayer = generate_matching_origin_destination_layer(roads)

    #### Network description ####

    # Mobility services observers
    pt_observer = CSVVehicleObserver(pt_obs_file)
    bike_observer = CSVVehicleObserver(bike_obs_file)
    walk_observer = CSVVehicleObserver(walk_obs_file)

    # Bus service and layer
    bus_service = PublicTransportMobilityService('BUS',
                                                 estimate_pt_load_method=estim_method,
                                                 veh_capacity=mobserv_p["BUS_CAPA"])
    bus_service.attach_vehicle_observer(pt_observer)
    bus_layer = PublicTransportLayer(roads,
                                     'BUSLayer',
                                     veh_type=Bus,
                                     default_speed=mobserv_p["BUS_DEFAULT_SPEED"],
                                     services=[bus_service],
                                     observer=pt_observer)

    bus_only_sections, bus_only_nodes = add_mservice_lines(roads,
                       mservice_anchor_nodes=mobserv_p["PT_STOPS"]["BUS"],
                       mservice_layer=bus_layer,
                       pt_start_rng=mobserv_p["PT_START_RNG"],
                       mservice_freq=mobserv_p["BUS_FREQUENCY"],
                       simu_start=Time(simu_p["SIMU_T_BOUNDARIES"][0]),
                       pt_end=simu_p["SIMU_T_BOUNDARIES"][-1])

    # Tram service and layer
    tram_service = PublicTransportMobilityService('TRAM', veh_capacity=mobserv_p["TRAM_CAPA"], estimate_pt_load_method=estim_method)
    tram_service.attach_vehicle_observer(pt_observer)
    tram_layer = PublicTransportLayer(roads,
                                      'TRAMLayer',
                                      veh_type=Tram,
                                      default_speed=mobserv_p["TRAM_DEFAULT_SPEED"],  # 13.9 m/s = 50 km/h
                                      services=[tram_service],
                                      observer=pt_observer)

    tram_only_sections, tram_only_nodes = add_mservice_lines(roads,
                       mservice_anchor_nodes=mobserv_p["PT_STOPS"]["TRAM"],
                       mservice_layer=tram_layer,
                       pt_start_rng=mobserv_p["PT_START_RNG"],
                       mservice_freq=mobserv_p["TRAM_FREQUENCY"],
                       simu_start=Time(simu_p["SIMU_T_BOUNDARIES"][0]),
                       pt_end=simu_p["SIMU_T_BOUNDARIES"][-1])

    # Bike service and layer
    bike_service = PersonalMobilityService('BIKE')
    bike_service.attach_vehicle_observer(bike_observer)
    bike_layer = generate_layer_from_roads(roads,
                                           layer_id='BIKELayer',
                                           class_layer=BikeLayer,
                                           veh_type=Bike,
                                           default_speed=mobserv_p["BIKE_DEFAULT_SPEED"],  # 4 m/s = 15 km/h
                                           mobility_services=[bike_service],
                                           banned_sections=bus_only_sections+tram_only_sections,
                                           banned_nodes=bus_only_nodes+tram_only_nodes)

    # Walk service and layer
    walk_service = PersonalMobilityService('WALKING')
    walk_service.attach_vehicle_observer(walk_observer)
    walk_layer = generate_layer_from_roads(roads,
                                           layer_id='WALKINGLayer',
                                           class_layer=SimpleLayer,
                                           veh_type=Walking,
                                           default_speed=mobserv_p["WALK_DEFAULT_SPEED"],
                                           mobility_services=[walk_service],
                                           banned_sections=bus_only_sections+tram_only_sections,
                                           banned_nodes=bus_only_nodes+tram_only_nodes)

    ## MLgraph ##
    mlgraph = MultiLayerGraph([bus_layer, bike_layer, tram_layer, walk_layer],
                              odlayer,
                              odlayer_p["ODLAYER_CONNECTION_DIST"])

    ## Zone filling estimation ##
    if estim_method == 'static':
        static_table = analyze_departures(filename=pt_load_file,
                                          multilaygraph=mlgraph,
                                          demand=demand,
                                          tstart=Time(simu_p["SIMU_T_BOUNDARIES"][0]), tend=Time(simu_p["SIMU_T_BOUNDARIES"][1]),
                                          temporal_step=estim_p["TAU"],
                                          expected_pt_share=traffic_p["EXPECTED_PT_SHARE"],
                                          spatial_step=estim_p["ZONE_SIZE"],
                                          avg_pt_speed=mobserv_p["AVG_PT_SPEED"])

        tram_service.demand_analysis_file = static_table
        bus_service.demand_analysis_file = static_table

    ## Load threshold for dynamic method ##
    if estim_method == 'dynamic':
        tram_service.load_threshold = estim_p["LOAD_THRESHOLD"]
        bus_service.load_threshold = estim_p["LOAD_THRESHOLD"]

    save_graph(mlgraph, network_file)

    ### Transit links ###
    # OD - BIKE
    st = time.time()
    od_bike_transit_links = mlgraph.layers['BIKELayer'].connect_origindestination(odlayer,
                                                                             connection_distance=odlayer_p["MAX_ACCESS_EGRESS_DIST_BIKE"],
                                                                             secure_connection_distance=odlayer_p["SECURE_MAX_ACCESS_EGRESS_DIST_BIKE"])
    mlgraph.add_transit_links(od_bike_transit_links)
    print(f'Connect OD layer with BIKE layer time = {time.time() - st}')

    # OD - WALKING
    st = time.time()
    od_walking_transit_links = mlgraph.layers['WALKINGLayer'].connect_origindestination(odlayer,
                                                                                  connection_distance=odlayer_p["MAX_ACCESS_EGRESS_DIST_WALKING"],
                                                                                  secure_connection_distance=odlayer_p["SECURE_MAX_ACCESS_EGRESS_DIST_WALKING"])
    mlgraph.add_transit_links(od_walking_transit_links)
    print(f'Connect OD layer with BIKE layer time = {time.time() - st}')

    # OD - PT
    st = time.time()
    for ptlid in ['BUSLayer', 'TRAMLayer']:
        od_pt_transit_links = mlgraph.layers[ptlid].connect_origindestination(odlayer,
                                                                              connection_distance=odlayer_p["MAX_ACCESS_EGRESS_DIST_PT"],
                                                                              secure_connection_distance=odlayer_p["SECURE_MAX_ACCESS_EGRESS_DIST_PT"])
        mlgraph.add_transit_links(od_pt_transit_links)
    print(f'Connect OD layer with PT layers time = {time.time() - st}')

    # PT - PT
    st = time.time()
    mlgraph.connect_intra_layer('BUSLayer', odlayer_p["MAX_TRANSFER_DIST"])
    print(f'Create TRANSIT links within BUS Layer time = {time.time() - st}')
    st = time.time()
    mlgraph.connect_intra_layer('TRAMLayer', odlayer_p["MAX_TRANSFER_DIST"])
    print(f'Create TRANSIT links within TRAM Layer time = {time.time() - st}')
    st = time.time()

    # PT - WALKING
    mlgraph.connect_inter_layers(['BUSLayer', 'TRAMLayer', 'WALKINGLayer'], odlayer_p["MAX_TRANSFER_DIST"])
    print(f'Create inter layers TRANSIT links time = {time.time() - st}')

    save_transit_links(mlgraph, transit_file)

    #### Decision model ####
    considered_modes = [({'WALKINGLayer'}, None, 1), ({'WALKINGLayer'}, None, 1), ({'BUSLayer', 'TRAMLayer', 'WALKINGLayer'}, None, 3)] # keep 3 PT paths bc no waiting time considered
    decision_model = ModeCentricLogitDecisionModel(mlgraph, considered_modes=considered_modes, outfile=paths_file, verbose_file=True) # cost is 'travel_time'

    #### Flow motor ####
    flow_motor = MFDFlowMotor()
    res = Reservoir(roads.zones["RES"], ['BUS', 'CAR', 'WALKING', 'TRAM', 'BIKE'], mfdspeedfunc)
    res.set_ghost_accumulation(traffic_p["N_CARS"])
    flow_motor.add_reservoir(res)

    #### Supervisor ####
    supervisor = Supervisor(mlgraph,
                            demand,
                            flow_motor,
                            decision_model,
                            logfile=log_file,
                            loglevel=LOGLEVEL.INFO)

    ### Run simulation ###
    set_all_mnms_logger_level(LOGLEVEL.INFO)

    print(f"RUNNING SIMULATION {demand_shape} - {estim_method} - {n_users}")
    st = time.time()
    supervisor.run(Time(simu_p["SIMU_T_BOUNDARIES"][0]),
                   Time(simu_p["SIMU_T_BOUNDARIES"][1]),
                   simu_p["DT_FLOW"],
                   simu_p["AFFECTATION_FACTOR"],
                   seed=seed)

    with open(log_file, 'a') as f:
        f.write(f'Run time = {time.time() - st}')

    print(f'runtime={time.time()-st}')