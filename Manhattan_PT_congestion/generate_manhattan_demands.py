import random
import csv
import json
import numpy as np
import pandas as pd
from math import dist
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, multivariate_normal, norm

from mnms.time import Time

def parse_coords(coord_str):
    try:
        x, y = map(float, coord_str.strip().split())
        return x, y
    except:
        return None, None

def plot_distribs(filename):
    # n_rows = 450_000
    # skip = np.arange(n_rows)
    # skip = np.delete(skip, np.arange(0, n_rows, 10))

    df = pd.read_csv(filename, sep=';') #, skiprows=skip)

    df[['origin_x', 'origin_y']] = df['ORIGIN'].apply(lambda x: pd.Series(parse_coords(x)))
    df[['dest_x', 'dest_y']] = df['DESTINATION'].apply(lambda x: pd.Series(parse_coords(x)))

    df = df.dropna(subset=['origin_x', 'origin_y', 'dest_x', 'dest_y'])

    plt.figure(figsize=(8, 6))
    plt.scatter(df['origin_x'], df['origin_y'], color='red', label='Origins', alpha=0.6)
    plt.scatter(df['dest_x'], df['dest_y'], color='blue', label='Destinations', alpha=0.6, marker='+')
    # plt.plot([df['origin_x'], df['dest_x']], [df['origin_y'], df['dest_y']])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(fontsize=12)
    # plt.title('Distributions of OD')
    plt.axis('equal')

    filename_no_extenstion = filename.split('.')[0]
    # plt.show()
    plt.savefig(f'{filename_no_extenstion}_spatial_distrib.pdf')

    ### Hist of departure times
    # Conversion de la colonne DEPARTURE en temps
    df["DEPARTURE"] = pd.to_datetime(df["DEPARTURE"], format="%H:%M:%S.%f")
    df["minutes"] = df["DEPARTURE"].dt.hour * 60 + df["DEPARTURE"].dt.minute

    # Définition des bins : de 0 à 2h, pas de 10 min
    bins = range(7 * 60, 9 * 60 + 10, 10)  # 0 à 1440, pas de 10

    # Histogramme
    plt.figure(figsize=(10, 6))
    plt.hist(df["minutes"], bins=bins, edgecolor="black")

    # Mise en forme des ticks en heures
    plt.xticks(range(7 * 60, 10 * 60, 60), [f"{h:02d}:00" for h in range(7, 10)])

    plt.xlabel("Departure times", fontsize=16)
    plt.ylabel("Departure counts", fontsize=16)
    # plt.title("Distribution of departure times (10 min bins)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{filename_no_extenstion}_time_distrib.pdf')

def generate_uniform_demand(demand_params, filename):
    """
    Generate a demand file with n_users demands where :
    - the departure times are uniformly distributed between t_min and t_max;
    - the origins and destinations are uniformly distributed on x (resp. y) axes, according to x_boundaries = (x_min, x_max) (resp. y_boundaries).
    Arguments :
    - n_users : number of users
    - t_min (resp. t_max) : earliest (rest. latest) departure time, format %H:%M:%S.%f
    - x_boundaries (resp. y_boundaries) : (x_min, x_max), span of the horizontal (resp.vertical) axis of the zone
    - filename : the name of the output csv file
    """

    n_users = demand_params["N_USERS"]
    t_boundaries = demand_params["DEMAND_T_BOUNDARIES"]
    x_boundaries = demand_params["X_BOUNDARIES"]
    y_boundaries = demand_params["Y_BOUNDARIES"]

    # Generate sorted departure times
    t_min = datetime.strptime(t_boundaries[0], "%H:%M:%S.%f")
    t_max = datetime.strptime(t_boundaries[1], "%H:%M:%S.%f")
    time_deltas = []

    for _ in range(n_users):
        delta = t_max - t_min
        random_seconds = random.randint(0, delta.seconds)
        time_deltas.append(random_seconds)
    time_deltas.sort()

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["ID", "DEPARTURE", "ORIGIN", "DESTINATION", "MOBILITY SERVICES"])

        for i, delta in enumerate(time_deltas):
            user_id = f"U{i}"

            departure_time = t_min + timedelta(seconds=delta)
            departure_time_str = departure_time.strftime("%H:%M:%S.%f")[:-4]

            origin_x = random.uniform(x_boundaries[0], x_boundaries[1])
            origin_y = random.uniform(y_boundaries[0], y_boundaries[1])
            origin = f"{origin_x} {origin_y}"

            destination_x = random.uniform(x_boundaries[0], x_boundaries[1])
            destination_y = random.uniform(y_boundaries[0], y_boundaries[1])
            destination = f"{destination_x} {destination_y}"

            writer.writerow([user_id, departure_time_str, origin, destination, "BUS TRAM WALKING"])

def generate_unicentric_demand(demand_params, filename):
    """
    todo
    """

    n_users = demand_params["N_USERS"]
    t_boundaries = demand_params["DEMAND_T_BOUNDARIES"]
    x_boundaries = demand_params["X_BOUNDARIES"]
    y_boundaries = demand_params["Y_BOUNDARIES"]

    t_mu = demand_params["T_MU"]
    t_sigma = demand_params["T_SIGMA"]
    d_mu = demand_params["D_MU"]
    d_sigma = demand_params["D_SIGMA"]
    d_sigma_origins = demand_params["D_SIGMA_ORIGINS"]
    # distance_factor = demand_params["DISTANCE_FACTOR"]

    # assert distance_factor >= 1, "distance_factor must be >= 1 (see function documentation)"

    # Generate sorted departure times following a normal distribution within t_boundaries
    t_min = datetime.strptime(t_boundaries[0], "%H:%M:%S.%f")
    t_max = datetime.strptime(t_boundaries[1], "%H:%M:%S.%f")
    t_mu_seconds = datetime.strptime(t_mu, "%H:%M:%S.%f")

    time_span = (t_max - t_min).total_seconds()
    t_mu_seconds = (t_mu_seconds - t_min).total_seconds()

    X = truncnorm((0 - t_mu_seconds) / t_sigma, (time_span - t_mu_seconds) / t_sigma, loc=t_mu_seconds, scale=t_sigma)
    # normal distrib truncated around 0 and time span

    departures_seconds = X.rvs(n_users)
    departures_seconds.sort()

    departures = []
    for i in range(len(departures_seconds)):
        dep = departures_seconds[i]
        dep = (t_min + timedelta(seconds=dep)).strftime("%H:%M:%S.%f")[:-4]
        departures.append(dep)

    # Generate destinations following a normal distribution around d_mu
    x_min, x_max = x_boundaries
    y_min, y_max = y_boundaries

    destinations = np.random.normal(loc=[d_mu[0], d_mu[1]], scale=[d_sigma[0], d_sigma[1]], size=(n_users, 2))
    destinations = np.clip(destinations, [x_min, y_min], [x_max, y_max])

    # Generate origins with inverted normal distrib (methode de rejet)
    def anti_normal_2d_pdf(x, y, d_mu, d_sigma_origins):
        Cx, Cy = d_mu

        r = np.sqrt((x - Cx) ** 2 + (y - Cy) ** 2)

        normal_term = np.exp(-0.5 * (r / d_sigma_origins) ** 2)
        max_normal_term = 1.0

        anti_density = max_normal_term - normal_term
        out_of_bounds = np.logical_or.reduce((x < x_min, x > x_max,
                                              y < y_min, y > y_max))
        anti_density[out_of_bounds] = 0

        return anti_density

    max_r_in_square = np.sqrt((x_min - d_mu[0]) ** 2 + (y_min - d_mu[1]) ** 2)
    max_density = 1

    origins = []
    while len(origins) < n_users:
        x_candidate = np.random.uniform(x_min, x_max)
        y_candidate = np.random.uniform(y_min, y_max)

        z_candidate = np.random.uniform(0, max_density)

        target_density = anti_normal_2d_pdf(np.array([x_candidate]),
                                            np.array([y_candidate]),
                                            d_mu, d_sigma_origins)[0]

        if z_candidate <= target_density:
            origins.append([x_candidate, y_candidate])

    # Save to file
    data = {
        'ID': [f'U{i}' for i in range(n_users)],
        'DEPARTURE': departures,
        'ORIGIN': [' '.join(map(str, o)) for o in origins],
        'DESTINATION': [' '.join(map(str, d)) for d in destinations],
        'MOBILITY SERVICES': ["BUS TRAM WALKING"] * n_users  # Empty column for mobility services
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=';')

def generate_multicentric_demand(demand_params, filename, pt_dict, n_nodes):
    """
    Generate a demand file with n_users demands where
    - departure times and destinations follow normal distributions according to the specified parameters;
    - origins follow a uniform distributions within a given distance of the destinations.
    Arguments :
    - n_users : number of users
    - t_boundaries : (t_min, t_max), time span of departure times, format %H:%M:%S.%f
    - x_boundaries (resp. y_boundaries) : (x_min, x_max), span of the horizontal (resp.vertical) axis of the zone
    - t_mu : mean of the distribution of departure times (peak demand hour), format %H:%M:%S.%f
    - t_sigma : std deviation of departure times, in seconds
    - d_mu_list : 2D means of the destinations distributions (centers of the destinations areas), of the form [(d_mu_x1, d_mu_y1), (d_mu_x2, d_mu_y2)...]
    - d_sigma_list : 2D std devs of the destinations distributions, of the form [(d_sigma_x1, d_sigma_y1), (d_sigma_x2, d_sigma_y2), ...]
    Must be of the same length than d_mu_list
    - weights-list : i-th element is probability of picking d_sigma_list[i]. Must sum to 1.
    - distance_factor : must be >= 1. The number of origins drawn is equal to distance_foctor * n_users, and only
    the n_users further from d_mu are kept.
    As a consequence, distance_factor=1 corresponds to drawing the n_users origins uniformly on the area, whereas a large
    distance_factor will favor origins far away from the destinations (and concentrated on the borders of the area)
    - filename : the name of the output csv file
    """

    n_users = demand_params["N_USERS"]
    t_boundaries = demand_params["DEMAND_T_BOUNDARIES"]
    x_boundaries = demand_params["X_BOUNDARIES"]
    y_boundaries = demand_params["Y_BOUNDARIES"]

    t_mu = demand_params["T_MU"]
    t_sigma = demand_params["T_SIGMA"]
    d_mu_list = demand_params["D_MU_LIST"]
    d_sigma_list = demand_params["D_SIGMA_LIST"]
    weights_list = demand_params["WEIGHTS_LIST"]
    epsilon = demand_params["EPSILON"]
    # distance_factor = demand_params["DISTANCE_FACTOR"]

    # assert distance_factor >= 1, "distance_factor must be >= 1 (see function documentation)"
    assert len(d_sigma_list) == len(d_mu_list), "d_sigma_list must be of the same length than d_mu_list (see function documentation)"
    assert len(weights_list) == len(d_mu_list) and sum(weights_list) == 1.0

    # Generate departure times following a normal distribution within t_boundaries
    t_min = datetime.strptime(t_boundaries[0], "%H:%M:%S.%f")
    t_max = datetime.strptime(t_boundaries[1], "%H:%M:%S.%f")
    t_mu_seconds = datetime.strptime(t_mu, "%H:%M:%S.%f")

    time_span = (t_max - t_min).total_seconds()
    t_mu_seconds = (t_mu_seconds - t_min).total_seconds()

    X = truncnorm((0 - t_mu_seconds) / t_sigma, (time_span - t_mu_seconds) / t_sigma, loc=t_mu_seconds, scale=t_sigma)
    # normal distrib truncated around 0 and time span

    departures_seconds = X.rvs(n_users)
    departures_seconds.sort()

    departures = []

    for i in range(len(departures_seconds)):
        dep = departures_seconds[i]
        dep = (t_min + timedelta(seconds=dep)).strftime("%H:%M:%S.%f")[:-4]
        departures.append(dep)

    # Generate destinations following a normal distribution around d_mu_list[i], i being random
    x_min, x_max = x_boundaries
    y_min, y_max = y_boundaries

    destinations = []
    for u in range(n_users):
        i = random.choices(population=range(len(d_mu_list)), weights=weights_list, k=1)[0]
        coord = [random.gauss(d_mu_list[i][0], d_sigma_list[i][0]), random.gauss(d_mu_list[i][1], d_sigma_list[i][1])]

        # Ensure that xi (resp. yi) is whithin x_boundaries (resp. y_boundaries)
        coord[0] = max(x_min, min(coord[0], x_max))
        coord[1] = max(y_min, min(coord[1], y_max))

        destinations.append(coord)

    # Fonctions pour tirer les OD autour des lignes
    def lines_from_stops(pt_dict, n_nodes):
        segments = []
        for d in pt_dict:
            for line in pt_dict[d]:
                stops_coords = []
                for node in pt_dict[d][line]:
                    x = (node // n_nodes) * 1000
                    y = (node % n_nodes) * 1000
                    stops_coords.append((x, y))
                for idx in range(len(stops_coords)-1):
                    segments.append((stops_coords[idx], stops_coords[idx+1]))
        return segments

    def point_segment_distance(px, py, x1, y1, x2, y2):
        seg = np.array([x2 - x1, y2 - y1])
        pt = np.array([px - x1, py - y1])

        seg_len2 = seg.dot(seg)
        if seg_len2 == 0:
            return np.hypot(px - x1, py - y1)

        t = max(0, min(1, pt.dot(seg) / seg_len2))
        proj = np.array([x1, y1]) + t * seg
        return np.hypot(px - proj[0], py - proj[1])

    def density(x, y, lines, eps):
        # eps grand => distrib très resserrée autour des lignes
        d = min(point_segment_distance(x, y, *seg[0], *seg[1]) for seg in lines)
        return 1 / (d + eps)

    def sample_points(n, bound_x, bound_y, lines, eps):
        points = []
        max_density = 1 / max(eps, 1)  # valeur approximative

        while len(points) < n:
            x = np.random.uniform(bound_x[0], bound_x[-1])
            y = np.random.uniform(bound_y[0], bound_y[-1])
            p = density(x, y, lines, eps)

            if np.random.rand() < p / max_density:
                points.append((x, y))

        return np.array(points)

    lines = lines_from_stops(pt_dict=pt_dict, n_nodes=n_nodes)
    origins = sample_points(n_users, x_boundaries, y_boundaries, lines=lines, eps=epsilon)

    # Generate origins far from all d_mu
    """
    origins_dict = {}
    for i in range(int(distance_factor * n_users)):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        d = min([dist([x, y], d_mu) for d_mu in d_mu_list])

        if len(origins_dict) < n_users:
            origins_dict[d] = [x, y]

        else:
            mindist = min(list(origins_dict.keys()))
            if mindist < d:
                origins_dict[d] = [x, y]
                del origins_dict[mindist]

    origins = list(origins_dict.values())
    """

    # Save to csv file
    data = {
        'ID': [f'U{i}' for i in range(n_users)],
        'DEPARTURE': departures,
        'ORIGIN': [' '.join(map(str, o)) for o in origins],
        'DESTINATION': [' '.join(map(str, d)) for d in destinations],
        'MOBILITY SERVICES': ["BUS TRAM WALKING"] * n_users  # Empty column for mobility services
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=';')

def compute_timedep_od_matrix_unicentric_distrib(demand_params, zones: dict, time_intervals: list, filename: str,
                                      n_samples: int = 200_000):
    """
    Return od_matrix[z_i][z_j][tau] = expected number of users who leave z_i during interval tau and have destination in z_j.
    """
    n_users = demand_params["N_USERS"]
    t_boundaries = demand_params["DEMAND_T_BOUNDARIES"]
    x_boundaries = demand_params["X_BOUNDARIES"]
    y_boundaries = demand_params["Y_BOUNDARIES"]

    t_mu = demand_params["T_MU"]
    t_sigma = demand_params["T_SIGMA"]
    d_mu = demand_params["D_MU"]
    d_sigma = demand_params["D_SIGMA"]
    distance_factor = demand_params["DISTANCE_FACTOR"]

    assert distance_factor >= 1, "distance_factor must be >= 1"

    t_min = Time(t_boundaries[0])
    t_max = Time(t_boundaries[1])
    t_mu_time = Time(t_mu)
    t_mu_seconds = t_mu_time.to_seconds() - t_min.to_seconds()
    time_span = t_max.to_seconds() - t_min.to_seconds()

    trunc_time = truncnorm(
        (0 - t_mu_seconds) / t_sigma,
        (time_span - t_mu_seconds) / t_sigma,
        loc=t_mu_seconds,
        scale=t_sigma,
    )

    tau_starts, tau_ends = [], []
    for tau in time_intervals:
        s, e = tau.split(" ")
        start_sec = Time(s.strip()).to_seconds() - t_min.to_seconds()
        end_sec = Time(e.strip()).to_seconds() - t_min.to_seconds()
        tau_starts.append(start_sec)
        tau_ends.append(end_sec)

    tau_starts = np.array(tau_starts)
    tau_ends = np.array(tau_ends)
    P_t = trunc_time.cdf(tau_ends) - trunc_time.cdf(tau_starts)
    P_t = P_t / P_t.sum()

    x_min, x_max = x_boundaries
    y_min, y_max = y_boundaries

    xs = np.random.uniform(x_min, x_max, n_samples)
    ys = np.random.uniform(y_min, y_max, n_samples)

    distances = np.sqrt((xs - d_mu[0]) ** 2 + (ys - d_mu[1]) ** 2)
    weights_origins = (distances ** (distance_factor - 1))
    weights_origins /= weights_origins.sum()

    z_ids = list(zones.keys())
    zone_bounds = np.array([
        [zones[z]["boundaries_x"][0], zones[z]["boundaries_x"][1],
         zones[z]["boundaries_y"][0], zones[z]["boundaries_y"][1]]
        for z in z_ids
    ])

    x = xs[:, None]
    y = ys[:, None]
    inside_orig = (
            (x >= zone_bounds[:, 0]) & (x <= zone_bounds[:, 1]) &
            (y >= zone_bounds[:, 2]) & (y <= zone_bounds[:, 3])
    )
    P_o = np.sum(weights_origins[:, None] * inside_orig, axis=0)
    P_o = P_o / P_o.sum()

    mvn = multivariate_normal(mean=d_mu, cov=np.diag(d_sigma))
    dx, dy = mvn.rvs(size=n_samples).T
    # clip in space frame
    dx = np.clip(dx, x_min, x_max)
    dy = np.clip(dy, y_min, y_max)

    inside_dest = (
            (dx[:, None] >= zone_bounds[:, 0]) & (dx[:, None] <= zone_bounds[:, 1]) &
            (dy[:, None] >= zone_bounds[:, 2]) & (dy[:, None] <= zone_bounds[:, 3])
    )
    weights_dest = np.ones(n_samples) / n_samples
    P_d = np.sum(weights_dest[:, None] * inside_dest, axis=0)
    P_d = P_d / P_d.sum()

    # OD spatial et ajout du temps
    OD_spatial = n_users * np.outer(P_o, P_d)  # shape (n_zones, n_zones)
    od_array = OD_spatial[:, :, None] * P_t[None, None, :]  # shape (n_i, n_j, n_tau)

    # conversion to dict {zi: {zj: {tau: nb of passengers leaving zi fpr zj during tau}}}
    od_matrix = {
        z_ids[i]: {
            z_ids[j]: {
                time_intervals[k]: float(od_array[i, j, k])
                for k in range(len(time_intervals))
            } for j in range(len(z_ids))
        } for i in range(len(z_ids))
    }

    with open(filename, 'w') as f:
        json.dump(od_matrix, f, indent=4)

    return od_matrix

if __name__ == '__main__':
    """fake_barcelona_bus_stops = {'purple0': [8, 16, 23, 30, 44], 'purple1': [44, 30, 23, 16, 8],
                                'yellow0': [43, 29, 23, 31, 45], 'yellow1': [45, 31, 23, 29, 43],
                                'blue0': [34, 33, 47, 45, 31, 17, 3], 'blue1': [3, 17, 31, 45, 47, 33, 34]}
    fake_barcelona_tram_stops = {'green0': [4, 3, 1, 8, 16, 17, 19, 20], 'green1': [20, 19, 17, 16, 8, 1, 3, 4],
                                 'red0': [3, 16, 23, 30, 45], 'red1': [45, 30, 23, 16, 3]}

    fake_lyon_bus_stops = {'t40': [21, 22, 24, 25, 26, 27], 't41': [27, 26, 25, 24, 22, 21], 't10': [10, 25, 26, 27],
                           't11': [27, 26, 25, 10], 't30': [45, 31, 24, 25], 't31': [25, 24, 31, 45],
                           't20': [10, 8, 22, 29, 43, 44], 't21': [44, 43, 29, 22, 8, 10]}
    fake_lyon_tram_stops = {'A0': [10, 12, 26, 33, 47], 'A1': [47, 33, 26, 12, 10], 'D0': [35, 22, 16, 10, 4],
                            'D1': [4, 10, 16, 22, 35], 'B0': [14, 16, 25, 26], 'B1': [26, 25, 16, 14]}

    demand_parameters = {
        "N_USERS": 400,

        "DEMAND_T_BOUNDARIES": ("07:00:00.00", "09:00:00.00"),
        "X_BOUNDARIES": (0, 6000),
        "Y_BOUNDARIES": (0, 6000),

        "T_MU": "08:00:00.00",
        "T_SIGMA": 3600/2,

        # # Lyon
        # "D_MU_LIST": [(1000, 3000), (3000, 5000), (6000, 4000)],  # perrache, doua, villeurbanne
        # "D_SIGMA_LIST": [(500, 500), (500, 500), (500, 500)],
        # "WEIGHTS_LIST": [0.3, 0.3, 0.4],

        # Barcelona
        "D_MU_LIST": [(0, 3000), (3000, 2000), (4000, 6000), (2000, 6000)],
        "D_SIGMA_LIST": [(500, 500), (500, 500), (300, 300), (300, 300)],
        "WEIGHTS_LIST": [0.25, 0.35, 0.2, 0.2],

        "DISTANCE_FACTOR": 1.1,
        "EPSILON": 100
    }
    filename = f"zz_od_fake_barcelona_eps={demand_parameters['EPSILON']}.csv"
    generate_multicentric_demand(demand_parameters, filename,
                                 pt_dict={"BUS": fake_barcelona_bus_stops,"TRAM": fake_barcelona_tram_stops},
                                 n_nodes=7)
    plot_distribs(filename)"""

    demand_parameters = {
        "N_USERS": 3000,
        "DEMAND_T_BOUNDARIES": ("07:00:00.00", "09:00:00.00"),
        "X_BOUNDARIES": (0, 6000),
        "Y_BOUNDARIES": (0, 6000),

        "T_MU": "08:00:00.00",
        "T_SIGMA": 3600/2,

        "D_MU": (3000, 3000),
        "D_SIGMA": (300, 300),
        "D_SIGMA_ORIGINS": 2000
    }

    #generate_unicentric_demand(demand_parameters, filename)
    filename = f"INPUTS/Barcelona_07-11_peakhour=08:00-30min_epsilon=100/Demand_2600users.csv"
    plot_distribs(filename)

