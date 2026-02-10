import numpy as np
from mnms.time import Time, Dt
from math import ceil, sqrt
import json
import csv
from build.lib.mnms.graph.layers import PublicTransportLayer
from mnms.mobility_service.public_transport import PublicTransportMobilityService
import time
from itertools import chain

### CASE 1 : BASED ON DISTRIB ###
"""
## Step 1 : create zones dict and intervals list
def create_zones_dict(multilaygraph, tstart: Time, tend: Time, temporal_step: Dt, spatial_step):

    ## Creation of time intervals
    k_times = ceil((tend - tstart).to_seconds() / temporal_step.to_seconds())
    intervals = []
    for kt in range(k_times):
        tmin = tstart.add_time(Dt(seconds=kt * temporal_step.to_seconds()))
        tmax = tstart.add_time(Dt(seconds=(kt + 1) * temporal_step.to_seconds()))
        intervals.append(f"{tmin} {tmax}")

    ## Creation of spatial zones
    g_nodes = multilaygraph.graph.nodes
    xmin, xmax = min([g_nodes[n].position[0] - (spatial_step / 2) for n in g_nodes]), max(
        [g_nodes[n].position[0] + (spatial_step / 2) for n in g_nodes])
    ymin, ymax = min([g_nodes[n].position[1] - (spatial_step / 2) for n in g_nodes]), max(
        [g_nodes[n].position[1] + (spatial_step / 2) for n in g_nodes])

    zones = {}
    kx_zones = ceil((xmax - xmin) / spatial_step)
    ky_zones = ceil((ymax - ymin) / spatial_step)
    for kx in range(kx_zones):
        for ky in range(ky_zones):
            zones[f'{kx}_{ky}'] = {"boundaries_x": [xmin + kx * spatial_step, xmin + (kx + 1) * spatial_step],
                                   "boundaries_y": [ymin + ky * spatial_step, ymin + (ky + 1) * spatial_step],
                                   "NW": {tau: 0 for tau in intervals},
                                   "SW": {tau: 0 for tau in intervals},
                                   "NE": {tau: 0 for tau in intervals},
                                   "SE": {tau: 0 for tau in intervals}
                                   }

    return zones, intervals

## 2 : compute supply matrix, store in file
# {zid: {"theta": [passengers,[lignes]]} }
def compute_supply_matrix(zones: dict, multilayergraph, temporal_step, filename) -> dict:

    supply_matrix = {zid: {"NW": [0,[]], "NE": [0,[]], "SW": [0,[]], "SE": [0,[]]} for zid in zones}
    # supply_matrix[z][theta] = (passengers who can leave z with dir theta during time interval, [lines available to leave z with direction theta])

    layers = multilayergraph.layers
    for lay_name in layers:
        for mob_service in layers[lay_name].mobility_services:
            service = layers[lay_name].mobility_services[mob_service]

            # Public transport layers only
            if type(service).__name__ != "PublicTransportMobilityService":
                continue

            service_nodes = layers[lay_name].graph.nodes
            service_capacity = layers[lay_name].mobility_services[mob_service].veh_capacity

            for node_id, node in service_nodes.items():
                node_pos = node.position

                if len(node.adj) == 0:
                    continue

                adj_node = list(node.adj.values())[0].downstream # downdtream node
                adj_pos = service_nodes[adj_node].position
                line_name, line_descript = find_line(node_id, layers[lay_name].lines)

                headway = line_descript['table'].get_freq()
                nb_veh_in_time_step = temporal_step.to_seconds() / headway
                nb_passengers_in_time_step = nb_veh_in_time_step * service_capacity # number of passengers that can board a vehicle of this line during a time step

                # find local line direction
                directions = []
                if adj_pos[0] <= node_pos[0] and adj_pos[1] <= node_pos[1]:
                    directions.append("SW")
                if adj_pos[0] <= node_pos[0] and adj_pos[1] >= node_pos[1]:
                    directions.append("NW")
                if adj_pos[0] >= node_pos[0] and adj_pos[1] >= node_pos[1]:
                    directions.append("NE")
                if adj_pos[0] >= node_pos[0] and adj_pos[1] <= node_pos[1]:
                    directions.append("SE")

                # Find zones to which the node belongs
                for zid, zdata in zones.items():
                    x_min, x_max = zdata["boundaries_x"]
                    y_min, y_max = zdata["boundaries_y"]

                    if x_min <= node_pos[0] <= x_max and y_min <= node_pos[1] <= y_max:
                        # Add line to list of lines that stop in zid
                        for d in directions:
                            if line_name not in supply_matrix[zid][d][1]:
                                supply_matrix[zid][d][1].append(line_name)
                                supply_matrix[zid][d][0] += nb_passengers_in_time_step/len(directions)

    with open(filename, 'w') as f:
        json.dump(supply_matrix, f, indent=4)

    return supply_matrix

## 3 : compute total zone occupancy, store in file
def compute_total_zone_occupancy_distrib(od_matrix, multilaygraph, zones, time_intervals, avg_pt_speed, filename: str):
    # A partir d'od_matrix[z1][z2][tau] et du multilayergraph,
    # determine :
    #   - zone_occupancy[zone][theta][tau] = total PT demand in zone for direction theta during tau
    # 
    # Principes :
    #   * Si lignes directes (ligne contenant z1 puis z2 dans cet ordre) -> on les utilise.
    #   * Poids d'une ligne ~ inverse du temps de parcours (z1->z2 sur cette ligne).
    #   * Allocation sur une ligne : pour les usagers qui choisissent cette ligne,
    #     la charge est uniformément répartie entre les zones où la ligne s'arrête *utilement*
    #     (p.ex. pour une ligne directe on prend la séquence de zones de z1 → z2 inclus).
    #   * Si pas de ligne directe -> on considère :
    #      - lignes passant par z1 et se dirigeant vers z2 (origin-side),
    #      - lignes passant par z2 et venant depuis direction z1 (arrival-side).
    #     On calcule poids similarement et on répartit.

    # Pré-indexer toutes les lignes : (line_name -> (layer_name, line_descript, node_positions, visited_zones))
    line_index = {}
    for lay_name, lay in multilaygraph.layers.items():
        if type(lay) != PublicTransportLayer:
            continue
        nodes = lay.graph.nodes
        for line_name, line_descript in lay.lines.items():
            node_positions = [nodes[n].position for n in line_descript['nodes']]
            visited_zones = line_zones_for_line(line_descript, nodes, zones)
            line_index[line_name] = {
                'layer': lay_name,
                'line_descript': line_descript,
                'node_positions': node_positions,
                'visited_zones': visited_zones
            }

    # itération
    for z1 in od_matrix:
        for z2 in od_matrix[z1]:
            if z1 == z2:
                continue
            # Find directions of z1->z2
            directions = []
            z1x, z1y = zones[z1]["boundaries_x"][0], zones[z1]["boundaries_y"][0]
            z2x, z2y = zones[z2]["boundaries_x"][0], zones[z2]["boundaries_y"][0]

            if z2x >= z1x and z2y >= z1y:
                directions.append("NE")
            if z2x >= z1x and z2y <= z1y:
                directions.append("SE")
            if z2x <= z1x and z2y >= z1y:
                directions.append("NW")
            if z2x <= z1x and z2y <= z1y:
                directions.append("SW")

            for tau in time_intervals:
                nflow = od_matrix[z1][z2][tau]
                # if z1 == "3_1" and z2 == "4_4" and tau == "07:55:00.00 08:05:00.00":
                #     input()
                if nflow <= 0:
                    continue

                # Find direct lines from z1 to z2
                direct_lines = find_direct_lines(z1, z2, line_index)

                if direct_lines:
                    # Compute proba of choosing each line
                    line_times, line_names = [], []
                    per_line_zone_seq = {}
                    for lname, info in direct_lines:
                        vz = info['visited_zones']
                        # indices des noeuds correspondant aux zones pour extraire positions
                        # on reconstruit le segment node_positions correspondant aux arrêts compris entre z1 et z2
                        # plus simple : extraire positions pour les visited_zones segment
                        idx1 = vz.index(z1)
                        idx2 = vz.index(z2)
                        # node_positions are many-to-many with visited_zones but we will approximate the segment positions
                        # by filtering node_positions whose zone belongs to vz[idx1:idx2+1]
                        seg_positions = []

                        zlist = []
                        for p in info['node_positions']:
                            z, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[p])
                            zlist.append(z)

                        #[find_zone_interval_direction(zones, points_of_interest=[p])[0] for p in info['node_positions']]

                        for pos, zone in zip(info['node_positions'], zlist):
                            if zone in vz[idx1:idx2 + 1]:
                                seg_positions.append(pos)
                        tt = travel_time_along_nodes(seg_positions, speed_m_s=avg_pt_speed)
                        line_times.append(tt)
                        line_names.append(lname)
                        # sequence of zones for allocation: vz[idx1:idx2+1]
                        per_line_zone_seq[lname] = vz[idx1:idx2 + 1]

                    # define line weight : the quicker the line, the larger the weight
                    inv = np.array([1.0 / t if t > 0 else 0.0 for t in line_times])
                    if inv.sum() == 0:
                        weights = np.ones(len(inv)) / len(inv)
                    else:
                        weights = inv / inv.sum()

                    # distribution sur zones : pour chaque ligne, uniformément sur ses zones segment
                    for lname, w in zip(line_names, weights):
                        seq = per_line_zone_seq[lname]
                        if z2 in seq:
                            seq.remove(z2)
                        # m = len(seq)
                        # if m == 0:
                        #     continue
                        per_zone_share = (nflow * w) #/ m
                        for zone_k in seq:
                            for theta in directions:
                                zones[zone_k][theta][tau] += per_zone_share/len(directions)

                else:
                    # If no direct line, find start and finish lines
                    origin_lines = find_origin_side_lines(z1, z2, line_index, zones)
                    arrival_lines = find_arrival_side_lines(z1, z2, line_index, zones)

                    # pour chaque candidate on compute un "temps utile" et une zone-sequence utile à partir de z1 (ou jusqu'à z2)
                    per_line_info = []
                    for lname, info in origin_lines:
                        vz = info['visited_zones']
                        idx = vz.index(z1)
                        # utile = zones après idx (inclus) qui appartiennent à la bbox z1->z2
                        useful = vz[idx:]
                        useful = [z for z in useful if (
                                zones[z1]['boundaries_x'][0] <= zones[z]['boundaries_x'][0] <= zones[z2]['boundaries_x'][0] and
                                zones[z1]['boundaries_y'][0] <= zones[z]['boundaries_y'][0] <= zones[z2]['boundaries_y'][0]
                        )]
                        if not useful:
                            continue
                        # segment positions for travel time: from idx (z1) to last useful
                        seg_pos = []
                        seg_zone_set = set(vz[idx: vz.index(useful[-1]) + 1]) if useful else set([z1])
                        for pos in info['node_positions']:
                            zpos, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[pos])
                            if zpos in seg_zone_set:
                                seg_pos.append(pos)
                        tt = travel_time_along_nodes(seg_pos, speed_m_s=avg_pt_speed)
                        per_line_info.append((lname, tt, list(seg_zone_set)))

                        if per_line_info:
                            # poids inverse sur tt
                            tts = np.array([p[1] for p in per_line_info], dtype=float)
                            inv = np.array([1.0 / t if t > 0 else 0.0 for t in tts])
                            if inv.sum() == 0:
                                wts = np.ones(len(inv)) / len(inv)
                            else:
                                wts = inv / inv.sum()

                            # pour chaque ligne candidate, répartir la fraction correspondante uniformément sur ses zones utiles
                            for (lname, tt, zone_seq), w in zip(per_line_info, wts):
                                if len(zone_seq) == 0:
                                    continue
                                per_zone_share = (nflow * w)
                                for zone_k in zone_seq:
                                    for theta in directions:
                                        zones[zone_k][theta][tau] += per_zone_share / len(directions)

                    per_line_info = []
                    for lname, info in arrival_lines:
                        vz = info['visited_zones']
                        idx = vz.index(z2)
                        # useful are past zones that lie in bbox
                        useful = vz[:idx]
                        useful = [z for z in useful if (
                                zones[z1]['boundaries_x'][0] <= zones[z]['boundaries_x'][0] <= zones[z2]['boundaries_x'][0] and
                                zones[z1]['boundaries_y'][0] <= zones[z]['boundaries_y'][0] <= zones[z2]['boundaries_y'][0]
                        )]
                        if not useful:
                            continue
                        # segment from first useful to z2
                        seg_zone_set = set(useful + [z2])
                        seg_pos = []
                        for pos in info['node_positions']:
                            zpos, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[pos])
                            if zpos in seg_zone_set:
                                seg_pos.append(pos)
                        tt = travel_time_along_nodes(seg_pos, speed_m_s=avg_pt_speed)
                        per_line_info.append((lname, tt, list(seg_zone_set)))

                        if per_line_info:
                            # poids inverse sur tt
                            tts = np.array([p[1] for p in per_line_info], dtype=float)
                            inv = np.array([1.0 / t if t > 0 else 0.0 for t in tts])
                            if inv.sum() == 0:
                                wts = np.ones(len(inv)) / len(inv)
                            else:
                                wts = inv / inv.sum()

                            # pour chaque ligne candidate, répartir la fraction correspondante uniformément sur ses zones utiles
                            for (lname, tt, zone_seq), w in zip(per_line_info, wts):
                                if len(zone_seq) == 0:
                                    continue
                                per_zone_share = (nflow * w)
                                for zone_k in zone_seq:
                                    for theta in directions:
                                        zones[zone_k][theta][tau] += per_zone_share / len(directions)

    with open(filename, 'w') as f:
        json.dump(zones, f, indent=4)

    return zones

## 4 : compute waiting factor matrix (update zones dict)
def compute_waiting_factor_matrix(zones, intervals, total_demand_matrix, supply_matrix, filename):
    waiting_factor_matrix = {zid: {"NW": {tau: 0 for tau in intervals},
                                   "SW": {tau: 0 for tau in intervals},
                                   "NE": {tau: 0 for tau in intervals},
                                   "SE": {tau: 0 for tau in intervals},
                                   "boundaries_x": zones[zid]["boundaries_x"],
                                   "boundaries_y": zones[zid]["boundaries_y"]
                                   }
                             for zid in zones}

    for zid in zones:
        for theta in ["NW", "SW", "SE", "NE"]:
            for tau in intervals:
                if supply_matrix[zid][theta] == [0, []]:
                    continue
                waiting_factor_matrix[zid][theta][tau] = total_demand_matrix[zid][theta][tau] // supply_matrix[zid][theta][0]

    with open(filename, "w") as outfile:
        json.dump(waiting_factor_matrix, outfile, indent=4)

    return waiting_factor_matrix
"""

### CASE 2 : BASED ON DEMAND DRAW ###
def analyze_departures(filename, multilaygraph, demand, tstart: Time, tend: Time, temporal_step: Dt, expected_pt_share, spatial_step, avg_pt_speed):

    print('BEGIN ANALYZE OF DEPARTURES --------')
    st = time.time()

    ## Creation of spatial zones
    g_nodes = multilaygraph.graph.nodes
    xmin, xmax = min([g_nodes[n].position[0] - (spatial_step / 2) for n in g_nodes]), max([g_nodes[n].position[0] + (spatial_step / 2) for n in g_nodes])
    ymin, ymax = min([g_nodes[n].position[1] - (spatial_step / 2) for n in g_nodes]), max([g_nodes[n].position[1] + (spatial_step / 2) for n in g_nodes])

    zones = {}
    kx_zones = ceil((xmax - xmin) / spatial_step)
    ky_zones = ceil((ymax - ymin) / spatial_step)
    for kx in range(kx_zones):
        for ky in range(ky_zones):
            zones[f'{kx}_{ky}'] = {"boundaries_x": [xmin + kx * spatial_step, xmin + (kx + 1) * spatial_step],
                                   "boundaries_y": [ymin + ky * spatial_step, ymin + (ky + 1) * spatial_step],
                                   "NW": {"lines": [], "demand": {}},
                                   "NE": {"lines": [], "demand": {}},
                                   "SW": {"lines": [], "demand": {}},
                                   "SE": {"lines": [], "demand": {}}}
    ## Creation of time intervals
    k_times = ceil((tend - tstart).to_seconds() / temporal_step.to_seconds())
    intervals = []
    for kt in range(k_times):
        tmin = tstart.add_time(Dt(seconds=kt * temporal_step.to_seconds()))
        tmax = tstart.add_time(Dt(seconds=(kt + 1) * temporal_step.to_seconds()))
        intervals.append(f"{tmin} {tmax}")
        for zid in zones:
            zones[zid]["NW"]["demand"][f"{tmin} {tmax}"] = {"amount": 0, "waiting_factor": 0}
            zones[zid]["NE"]["demand"][f"{tmin} {tmax}"] = {"amount": 0, "waiting_factor": 0}
            zones[zid]["SW"]["demand"][f"{tmin} {tmax}"] = {"amount": 0, "waiting_factor": 0}
            zones[zid]["SE"]["demand"][f"{tmin} {tmax}"] = {"amount": 0, "waiting_factor": 0}

    ## Compute total capacity for each zone in each direction
    layers = multilaygraph.layers
    for lay_name in layers:
        for mob_service in layers[lay_name].mobility_services:
            if type(layers[lay_name].mobility_services[mob_service]) is PublicTransportMobilityService:

                service_nodes = layers[lay_name].graph.nodes
                service_capacity = layers[lay_name].mobility_services[mob_service].veh_capacity

                for node in service_nodes:
                    line_name, line_descript = find_line(node, layers[lay_name].lines)
                    headway = line_descript['table'].get_freq()
                    node_position = service_nodes[node].position

                    nb_veh_in_time_step = temporal_step.to_seconds() / headway

                    if len(service_nodes[node].adj) == 1:
                        # get local direction(s) of the line
                        adj_node = list(service_nodes[node].adj.values())[0].downstream
                        adj_position = service_nodes[adj_node].position

                        # <= and >= because a link towards east can serve users going both north-east and south-east
                        directions_of_line = []
                        if adj_position[0] <= node_position[0] and adj_position[1] <= node_position[1]:
                            directions_of_line.append("SW")
                        if adj_position[0] <= node_position[0] and adj_position[1] >= node_position[1]:
                            directions_of_line.append("NW")
                        if adj_position[0] >= node_position[0] and adj_position[1] >= node_position[1]:
                            directions_of_line.append("NE")
                        if adj_position[0] >= node_position[0] and adj_position[1] <= node_position[1]:
                            directions_of_line.append("SE")

                        # for each zone to which node belongs, add capacity of the line to zone
                        for zid in zones:
                            # check if line is already registered in zone, in this case skip to following zone
                            already_in_zone = False
                            for test_dir in ["NW", "NE", "SE", "SW"]:
                                lines_names = [t[0] for t in zones[zid][test_dir]["lines"]]
                                if line_name in lines_names:
                                    already_in_zone = True
                                    break
                            if already_in_zone:
                                continue

                            # check if node is in this zone
                            if zones[zid]["boundaries_x"][0] <= node_position[0] <= zones[zid]["boundaries_x"][1] \
                                    and zones[zid]["boundaries_y"][0] <= node_position[1] <= zones[zid]["boundaries_y"][1]:

                                future_zones = [] # zones that the line will visit after current zone
                                for future_node in line_descript['nodes'][line_descript['nodes'].index(node) + 1:]:
                                    future_node_position = service_nodes[future_node].position
                                    future_node_zone, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[future_node_position])
                                    if future_node_zone not in future_zones:
                                        future_zones.append(future_node_zone)

                                past_zones = []  # zones that the line visited before current zones
                                for past_node in line_descript['nodes'][:line_descript['nodes'].index(node)]:
                                    past_node_position = service_nodes[past_node].position
                                    past_node_zone, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[past_node_position])
                                    if past_node_zone not in past_zones:
                                        past_zones.append(past_node_zone)

                                for dir_line in directions_of_line:
                                    zones[zid][dir_line]["lines"].append((line_name, (service_capacity * nb_veh_in_time_step) / len(dir_line), dir_line, future_zones, past_zones))

                    elif len(service_nodes[node].adj) == 0: # si c'est le dernier arrêt de la ligne, il faut quand même dire qu'on s'arrête buien dans la zone
                        pass
                        prev_node = list(service_nodes[node].radj.values())[0].upstream
                        prev_position = service_nodes[prev_node].position

                        # direction with which line enters zone, slightly different from previously but does not matter
                        # since no passenger will board at this station
                        directions_of_line = []
                        if prev_position[0] >= node_position[0] and prev_position[1] >= node_position[1]:
                            directions_of_line.append("SW")
                        if prev_position[0] >= node_position[0] and prev_position[1] <= node_position[1]:
                            directions_of_line.append("NW")
                        if prev_position[0] <= node_position[0] and prev_position[1] <= node_position[1]:
                            directions_of_line.append("NE")
                        if prev_position[0] <= node_position[0] and prev_position[1] >= node_position[1]:
                            directions_of_line.append("SE")

                        for zid in zones:
                            # check if line is already registered in zone, in this case skip to following zone
                            already_in_zone = False
                            for test_dir in ["NW", "NE", "SE", "SW"]:
                                lines_names = [t[0] for t in zones[zid][test_dir]["lines"]]
                                if line_name in lines_names:
                                    already_in_zone = True
                                    break
                            if already_in_zone:
                                continue

                            # check if node is in this zone
                            if zones[zid]["boundaries_x"][0] <= node_position[0] <= zones[zid]["boundaries_x"][1] \
                                    and zones[zid]["boundaries_y"][0] <= node_position[1] <= zones[zid]["boundaries_y"][1]:

                                future_zones = [] # line will not visit any other zone after this node
                                past_zones = []  # zones that the line visited before current zones
                                for past_node in line_descript['nodes'][:line_descript['nodes'].index(node)]:
                                    past_node_position = service_nodes[past_node].position
                                    past_node_zone, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[past_node_position])
                                    if past_node_zone not in past_zones:
                                        past_zones.append(past_node_zone)

                                for dir_line in directions_of_line:
                                    zones[zid][dir_line]["lines"].append((line_name, 0, dir_line, future_zones, past_zones))

    ## Count number of departures in each zone in each direction and for each time interval
    # Have to treat U0 before looping on users in self._demand._reader
    u0 = demand._current_user
    update_zones_demand(u_available_modes=u0.available_mobility_services,
                        u_origin=u0.origin,
                        u_destination=u0.destination,
                        u_departure_time=u0.departure_time,
                        zones_dict=zones,
                        avg_pt_speed=avg_pt_speed)

    # Now treat users from U1 onwards
    for row in demand._reader:
        update_zones_demand(u_available_modes=row[-1],
                            u_origin=[float(row[2].split(' ')[0]), float(row[2].split(' ')[1])],
                            u_destination=[float(row[3].split(' ')[0]), float(row[3].split(' ')[1])],
                            u_departure_time=Time(row[1]),
                            zones_dict=zones,
                            avg_pt_speed=avg_pt_speed)

    # Re-initialize csv reader, otherwise only user 0 will be treated in simulation
    file = open(demand._filename, 'r', newline='')
    file.seek(0)
    demand._reader = csv.reader(file, delimiter=';', quotechar='|')
    next(demand._reader, None)  # skip header
    next(demand._reader, None)  # skip U0 because it is still in self._demand._current_user

    ## Compute waiting factor
    for zid in zones:
        for direction in ["NE", "NW", "SE", "SW"]:
            for time_interval in zones[zid][direction]["demand"]:
                zone_capa = sum([line[1] for line in zones[zid][direction]["lines"]])  # line[1] = capacity of the line for one direction
                try:
                    w_f = (zones[zid][direction]["demand"][time_interval]["amount"]) // zone_capa
                except ZeroDivisionError:
                    w_f = 0
                zones[zid][direction]["demand"][time_interval]["waiting_factor"] = w_f
                if (zones[zid][direction]["demand"][time_interval]["amount"] > zone_capa) & (w_f < 1):
                    print("ERROR : waiting factor not updated")

    with open(filename, "w") as outfile:
        json.dump(zones, outfile, indent=4)

    print('ANALYZE OF DEPARTURES DONE IN ', time.time()-st, ' SECONDS')

    return zones

def update_zones_demand(u_available_modes, u_origin, u_destination, u_departure_time, zones_dict, avg_pt_speed):
    if "BUS" not in u_available_modes and "TRAM" not in u_available_modes and "METRO" not in u_available_modes:
        return
    z_or, z_dest, t_dep, _ = find_zone_interval_direction(zones_dict, points_of_interest=[u_origin, u_destination], time_of_interest=u_departure_time)


    lines_z_or = list(chain(*[zones_dict[z_or][d]["lines"] for d in ['NW', 'NE', 'SW', 'SE']]))  # lines that stop in user's departure zone
    lines_z_dest = list(chain(*[zones_dict[z_dest][d]["lines"] for d in ['NW', 'NE', 'SW', 'SE']]))  # lines that stop in user's arrival zone

    # If some lines are direct, user will occupy a seat in those lines
    intersection = []  # list of direct lines between origin zone and destination zone
    for l_or in lines_z_or:
        for l_dest in lines_z_dest:
            if l_or[0] == l_dest[0] and z_dest in l_or[-2] and z_or in l_dest[-1]:  # if lines are identical and stop in z_or before z_dest
                intersection.append(l_or)

    if intersection :

        # define weight of each line as nb of zones stopped into between z_o and z_dest
        def find_nb_zones_until_dest(line):
            n = 0
            for future_zone in line[-2]:
                if future_zone == z_dest:
                    break
                n += 1
            return n

        inv = np.array([1 / find_nb_zones_until_dest(line) if find_nb_zones_until_dest(line) > 0 else 0 for line in intersection])
        if inv.sum() == 0:
            weights = np.ones(len(inv)) / len(inv)
        else:
            weights = inv / inv.sum()

        for line, weight in zip(intersection, weights):

            zones_dict[z_or][line[2]]["demand"][t_dep]["amount"] += weight # add weight to departure zone
            future_zones = line[-2]

            prev_zone = z_or
            prev_time = u_departure_time

            for curr_zone in future_zones:
                if curr_zone == z_dest:
                    break

                # Time to go from one zone to the next (not always the same since zones can be in diagonal)
                zone_travel_time = zone_dist(zones_dict[prev_zone], zones_dict[curr_zone]) / avg_pt_speed
                curr_time = prev_time.add_time(Dt(seconds=zone_travel_time))

                # Get time interval in which u arrives in new zone
                _, _, curr_interval, _ = find_zone_interval_direction(zones_dict, time_of_interest=curr_time)

                local_line_directions = []
                for d in ["NW", "NE", "SW", "SE"]:
                    for line_tuple in zones_dict[curr_zone][d]["lines"]:
                        if line_tuple[0] == line[0]:
                            local_line_directions.append(d)

                for curr_dir in local_line_directions:
                    zones_dict[curr_zone][curr_dir]["demand"][curr_interval]["amount"] += weight / len(local_line_directions)

                prev_time = curr_time
                prev_zone = curr_zone

    elif not intersection:  # if no direct line, then user distributes on other lines

        def is_in_rect(zone, zone_or, zone_dest):
            # Return True if zone is inside the rectangle defined by zone_or as lower left corner and zone_dest as upper right corner
            zx = zones_dict[zone]["boundaries_x"][0]
            zy = zones_dict[zone]["boundaries_y"][0]
            zox = zones_dict[zone_or]["boundaries_x"][0]
            zoy = zones_dict[zone_or]["boundaries_y"][0]
            zdx = zones_dict[zone_dest]["boundaries_x"][0]
            zdy = zones_dict[zone_dest]["boundaries_y"][0]
            a = (zox <= zx <= zdx) if zox <= zdx else (zdx <= zx <= zox)
            b = (zoy <= zy <= zdy) if zoy <= zdy else (zdy <= zy <= zoy)
            return a and b

        ###### Then on the lines that stop in departure zone
        # Select lines that have at least one stop between z_or and z_dest
        lines_available_dep = []
        for direction in ['NW', 'NE', 'SW', 'SE']:
            for line_tuple in zones_dict[z_or][direction]["lines"]:
                for future_zone in line_tuple[-2]:
                    if is_in_rect(future_zone, z_or, z_dest) and not line_tuple in lines_available_dep:
                        lines_available_dep.append(line_tuple)

        if lines_available_dep:
            # compute weight of line based on how close it will bring user to dest
            invs = np.array([0 for _ in lines_available_dep])

            # compute last interesting zone for user
            last_interesting_zones = []

            for i, line in enumerate(lines_available_dep):
                future_zones = line[-2]
                last_interesting_zone = z_or
                for new_zone in future_zones:
                    if is_in_rect(new_zone, z_or, z_dest):
                        last_interesting_zone = new_zone
                        if new_zone == future_zones[-1]: # if we are at the end of the list then we will never leave the rectangle so add to list
                            last_interesting_zones.append(last_interesting_zone)
                    else: # if we leave the rectangle, then add last found interesting zone to list
                        last_interesting_zones.append(last_interesting_zone)
                        break

                residual_dist =  zone_dist(zones_dict[last_interesting_zone], zones_dict[z_dest]) # find residual distance to destination
                invs[i] = 1/residual_dist if residual_dist > 0 else 0

            weights = invs/invs.sum() if invs.sum() > 0 else np.ones(len(invs)) / len(invs)

            # inv = np.array([1.0 / len(line[-2]) if not line[-2] == [] else 0 for line in lines_available_dep])
            # if inv.sum() == 0:
            #     weights = np.ones(len(inv)) / len(inv)
            # else:
            #     weights = inv / inv.sum()

            for line, weight, liz in zip(lines_available_dep, weights, last_interesting_zones):
                # add demand to departure zone
                zones_dict[z_or][line[2]]["demand"][t_dep]["amount"] += weight

                future_zones = line[-2]
                prev_zone = z_or
                prev_time = u_departure_time

                for curr_zone in future_zones:
                    if curr_zone == liz:
                        break
                    else:
                        curr_time = prev_time.add_time(Dt(seconds=zone_dist(zones_dict[prev_zone], zones_dict[curr_zone]) / avg_pt_speed))
                        _, _, curr_interval, _ = find_zone_interval_direction(zones_dict, time_of_interest=curr_time)

                        local_line_directions = []
                        for d in ["NW", "NE", "SW", "SE"]:
                            for line_tuple in zones_dict[curr_zone][d]["lines"]:
                                if line_tuple[0] == line[0]:
                                    local_line_directions.append(d)

                        for curr_dir in local_line_directions:
                            zones_dict[curr_zone][curr_dir]["demand"][curr_interval]["amount"] += weight/len(local_line_directions)

                        prev_zone = curr_zone
                        prev_time = curr_time

        ###### Then on the lines that stop in arrival zone
        # Find zones that stop in z_dest and have at least one previous stop between z_o and z_dest
        lines_available_arr = []
        for direction in ['NW', 'NE', 'SW', 'SE']:
            for line_tuple in zones_dict[z_dest][direction]["lines"]:
                for past_zone in line_tuple[-1]:
                    if is_in_rect(past_zone, z_or, z_dest) and not line_tuple in lines_available_arr:
                        lines_available_arr.append(line_tuple)

        if lines_available_arr:
            # weight of each line
            invs = np.array([0 for _ in lines_available_arr])

            # compute first interesting zone for user
            first_interesting_zones = []
            for i, line in enumerate(lines_available_arr):
                past_zones = line[-1]
                first_interesting_zone = z_dest
                for new_zone in past_zones[::-1]:
                    if is_in_rect(new_zone, z_or, z_dest):
                        first_interesting_zone = new_zone
                        if new_zone == past_zones[0]:
                            first_interesting_zones.append(first_interesting_zone)
                    else:
                        first_interesting_zones.append(first_interesting_zone)
                        break

                residual_dist = zone_dist(zones_dict[first_interesting_zone], zones_dict[z_or])  # find residual distance to destination
                invs[i] = 1 / residual_dist if residual_dist > 0 else 0 # delta should not be zero since the line is not supposed to be direct

            weights = invs / invs.sum() if invs.sum() > 0 else np.ones(len(invs)) / len(invs)

            # inv = np.array([1.0 / len(line[-2]) if not line[-2] == [] else 0 for line in lines_available_arr])
            # if inv.sum() == 0:
            #     weights = np.ones(len(inv)) / len(inv)
            # else:
            #     weights = inv / inv.sum()

            for line, weight, fiz in zip(lines_available_arr, weights, first_interesting_zones):
                zones_to_explore = line[-1] # zones of the line before z_dest

                prev_zone = z_or
                prev_time = u_departure_time

                found_fiz = False
                for curr_zone in zones_to_explore: # current interesting zone
                    if curr_zone == fiz:
                        found_fiz = True
                    if found_fiz:
                        # time spent going from the prev interesting zone to current interesting zone
                        time_to_come_from_prev = zone_dist(zones_dict[prev_zone], zones_dict[curr_zone]) / avg_pt_speed
                        curr_time = prev_time.add_time(Dt(seconds=time_to_come_from_prev))

                        # find time interval
                        _, _, curr_interval, _ = find_zone_interval_direction(zones_dict, points_of_interest=None, time_of_interest=curr_time)

                        # find local direction(s) of line
                        local_line_directions = []
                        for d in ["NW","NE","SW","SE"]:
                            for line_tuple in zones_dict[curr_zone][d]["lines"]:
                                if line_tuple[0] == line[0]:
                                    local_line_directions.append(d)

                        # increment demand by weight (half weight if two directions)
                        for curr_dir in local_line_directions:
                            # if curr_zone is None or curr_dir is None or curr_interval is None:
                            #     input("ici")
                            #     print("coucou")
                            zones_dict[curr_zone][curr_dir]["demand"][curr_interval]["amount"] += weight/len(local_line_directions)

                        prev_time = curr_time
                        prev_zone = curr_zone



## Utils
def find_zone_interval_direction(zones_dict, points_of_interest=None, time_of_interest=None):
    """
    TODO
    Args:
        zones_dict:
        points_of_interest:
        time_of_interest:

    Returns:
        if 1 point of interest : returns zone of this poi
        if points of interest : trouver les deux zones + la direction 0 -> 1
        si time_of_interest : trouver l'intervalle

    """

    if points_of_interest is None:
        points_of_interest = []
    zo, zd, t, d = None, None, None, None

    # Find zone of origin point
    if len(points_of_interest) == 1:
        origin = points_of_interest[0]
        for zid in zones_dict:
            if zones_dict[zid]["boundaries_x"][0] <= origin[0] < zones_dict[zid]["boundaries_x"][1]\
                    and zones_dict[zid]["boundaries_y"][0] <= origin[1] < zones_dict[zid]["boundaries_y"][1]:
                zo = zid
        assert zo is not None, f'Point of coord. {origin} does not match any zone'

    # Find zones of origin and destination points
    elif len(points_of_interest) == 2:
        origin = points_of_interest[0]
        destination = points_of_interest[-1]
        for zid in zones_dict:
            if zones_dict[zid]["boundaries_x"][0] <= origin[0] < zones_dict[zid]["boundaries_x"][1] \
                    and zones_dict[zid]["boundaries_y"][0] <= origin[1] < zones_dict[zid]["boundaries_y"][1]:
                zo = zid
            if zones_dict[zid]["boundaries_x"][0] <= destination[0] < zones_dict[zid]["boundaries_x"][1] \
                    and zones_dict[zid]["boundaries_y"][0] <= destination[1] < zones_dict[zid]["boundaries_y"][1]:
                zd = zid
        assert zo is not None, f'Point of coord. {origin} does not match any zone'
        assert zd is not None, f'Point of coord. {destination} does not match any zone'

        # Find direction
        if destination[0] <= origin[0] and destination[1] < origin[1]:
            d = "SW"
        elif destination[0] < origin[0] and destination[1] >= origin[1]:
            d = "NW"
        elif destination[0] >= origin[0] and destination[1] > origin[1]:
            d = "NE"
        elif destination[0] > origin[0] and destination[1] <= origin[1]:
            d = "SE"

    if time_of_interest is not None:
        if zo is not None and d is not None:
            intervals_list = list(zones_dict[zo][d]["demand"].keys())
        else:
            zid = next(iter(zones_dict.keys()))
            intervals_list = list(zones_dict[zid]['NW']["demand"].keys())
        for i in intervals_list:
            tmin, tmax = Time(i.split(' ')[0]), Time(i.split(' ')[1])
            if tmin <= time_of_interest <= tmax:
                t = i

    return zo, zd, t, d

def line_zones_for_line(line_descript, layer_graph_nodes, zones):
    node_list = line_descript['nodes']
    visited_zones = []
    for n in node_list:
        pos = layer_graph_nodes[n].position
        z, _, _, _ = find_zone_interval_direction(zones, points_of_interest=[pos])
        if visited_zones and visited_zones[-1] == z:
            continue
        visited_zones.append(z)
    return visited_zones

# Compute travel time along a list of nodes
def travel_time_along_nodes(node_positions, speed_m_s=9.0):
    # node_positions : list of (x,y)
    if len(node_positions) < 2:
        return 1e9  # évite division par zéro, dissuade la sélection
    dist = 0.0
    for a, b in zip(node_positions[:-1], node_positions[1:]):
        dist += np.hypot(a[0] - b[0], a[1] - b[1])
    return dist / speed_m_s

def find_direct_lines(z1, z2, line_index):
    direct = []
    for lname, info in line_index.items():
        vz = info['visited_zones']
        if z1 in vz and z2 in vz and vz.index(z1) < vz.index(z2):
            direct.append((lname, info))
    return direct

# Find lines stopping in z1 and usefol to go to z2
def find_origin_side_lines(z1, z2, line_index, zones):
    possible = []
    for lname, info in line_index.items():
        vz = info['visited_zones']
        if z1 in vz:
            idx = vz.index(z1)
            future = vz[idx + 1:]

            # a line is "useful to go to z2" if it visits at least 1 zone between z1 and z2 (in the box defined by z1 and z2)
            def is_in_rect(zone):
                zx = zones[zone]['boundaries_x'][0]
                zy = zones[zone]['boundaries_y'][0]
                zox = zones[z1]['boundaries_x'][0]
                zoy = zones[z1]['boundaries_y'][0]
                zdx = zones[z2]['boundaries_x'][0]
                zdy = zones[z2]['boundaries_y'][0]
                return zox <= zx <= zdx and zoy <= zy <= zdy

            useful = [fz for fz in future if is_in_rect(fz)]
            if useful:
                possible.append((lname, info))
    return possible

# Find lines stopping in z2 and useful to come from z1
def find_arrival_side_lines(z1, z2, line_index, zones):
    possible = []
    for lname, info in line_index.items():
        vz = info['visited_zones']
        if z2 in vz and z1 not in vz:
            idx = vz.index(z2)
            past = vz[:idx]

            def is_in_rect(zone):
                zx = zones[zone]['boundaries_x'][0]
                zy = zones[zone]['boundaries_y'][0]
                zox = zones[z1]['boundaries_x'][0]
                zoy = zones[z1]['boundaries_y'][0]
                zdx = zones[z2]['boundaries_x'][0]
                zdy = zones[z2]['boundaries_y'][0]
                return zox <= zx <= zdx and zoy <= zy <= zdy

            useful = [pz for pz in past if is_in_rect(pz)]
            if useful:
                possible.append((lname, info))
    return possible

def zone_dist(z1, z2):
    c1 = [(z1["boundaries_x"][1]+z1["boundaries_x"][0])/2, (z1["boundaries_y"][1]+z1["boundaries_y"][0])/2]
    c2 = [(z2["boundaries_x"][1]+z2["boundaries_x"][0])/2, (z2["boundaries_y"][1]+z2["boundaries_y"][0])/2]
    return sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

def find_line(node, lines_dict):
    for line_name in lines_dict:
        if node in lines_dict[line_name]["nodes"]:
            return line_name, lines_dict[line_name]
