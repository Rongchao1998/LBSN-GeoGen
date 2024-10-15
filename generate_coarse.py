import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from utils.preprocess import read_real_trajectories
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean


def read_gps(file_path):
    gps_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split(' ')
            ls = [float(points[0]), float(points[1])]
            gps_points.append(ls)
    # normalize the latitude and longitude to [0, 1] respectively
    gps_points = np.array(gps_points)
    original = gps_points.copy()
    lat_min, lat_max = np.min(gps_points[:, 0]), np.max(gps_points[:, 0])
    lon_min, lon_max = np.min(gps_points[:, 1]), np.max(gps_points[:, 1])
    gps_points[:, 0] = (gps_points[:, 0] - lat_min) / (lat_max - lat_min)
    gps_points[:, 1] = (gps_points[:, 1] - lon_min) / (lon_max - lon_min)
    # transform all values to float
    gps_points = gps_points.tolist()
    return original, gps_points, lat_min, lat_max, lon_min, lon_max


def interpolate(trajs, interval=0.5):
    inter_trajs = []
    for traj in trajs:
        inter_traj = [-1] * int(24 * 7 / interval)
        for i in range(0, len(traj)):
            # find the index of the time offset
            index = int((traj[i][1] + interval) / interval)
            if index < len(inter_traj) and inter_traj[index] == -1:
                inter_traj[index] = traj[i][0]
        for i in range(1, len(inter_traj)):
            if inter_traj[i] == -1:
                inter_traj[i] = inter_traj[i-1]
        inter_trajs.append(inter_traj)
    return inter_trajs


# transform the interpolated data into the same format as the real data,
def transform_poi_trajectories(trajectories, interval=0.5, gps=None, poi=False):
    transformed_trajectories = []

    for trajectory in trajectories:
        transformed_trajectory = []
        previous_poi = None

        for index, poi_id in enumerate(trajectory):
            if poi_id != previous_poi:
                # Calculate time offset in hours
                time_offset = index * interval  # since each index represents a 30-minute interval

                # Append the new check-in point
                if poi and gps:
                    transformed_trajectory.append([gps[poi_id], time_offset])
                else:
                    transformed_trajectory.append([poi_id, time_offset])

                # Update the previous POI id
                previous_poi = poi_id
        # remove the first check-in point
        transformed_trajectory = transformed_trajectory[1:]

        # Store the transformed trajectory
        if len(transformed_trajectory) > 0:
            transformed_trajectories.append(transformed_trajectory)

    return transformed_trajectories

# generate sequence of trajectories
def generate_trajectories_from_gps(syn_trajs, interval=0.5):
    trajectories = []
    length = syn_trajs.shape[2]

    dbscan = DBSCAN(eps=0.3, min_samples=2)
    scaler = StandardScaler()

    for syn_traj in syn_trajs:
        # shape of the syn_traj is (2, length)
        # cluster these coordinates
        syn_traj = syn_traj.T
        gps_scaled = scaler.fit_transform(syn_traj)

        # DBSCAN clustering
        clusters = dbscan.fit_predict(gps_scaled)

        # change -1 to the prior cluster
        for i in range(1, length):
            if clusters[i] == -1:
                clusters[i] = clusters[i-1]

        # Collect coordinates for each cluster
        clustered_coordinates = {}
        for cluster_label in set(clusters):
            if cluster_label != -1:  # Exclude noise points if you want
                indices = np.where(clusters == cluster_label)
                clustered_coordinates[cluster_label] = syn_traj[indices]

        cluster_centers = []
        for cluster_label, coordinates in clustered_coordinates.items():
            cluster_center = np.mean(coordinates, axis=0)
            cluster_centers.append(cluster_center)

        # get the trajectory length from clusters
        clen = 0
        traj = []
        for i in range(1, length):
            if clusters[i] != clusters[i-1]:
                clen += 1
                traj.append([cluster_centers[clusters[i]], i * interval])
        trajectories.append(traj)

    return trajectories


def transform_normalized_gps_to_original(trajs, lat_min, lat_max, lon_min, lon_max):
    result = []
    for traj in trajs:
        new_traj = []
        for i in range(len(traj)):
            new_traj.append([[traj[i][0][0] * (lat_max - lat_min) + lat_min, traj[i][0][1] * (lon_max - lon_min) + lon_min], traj[i][1]])
        result.append(new_traj)

    return result


def get_visiting_distribution(trajs, num_pois, interval=0.5, seq_len=336):
    # trajs: list of trajectories, each trajectory is a list of check-in points. Each check-in point is a list of
    # [poi, time offset]
    visiting_distributions = np.zeros((num_pois, seq_len))
    for traj in trajs:
        for poi, time in traj:
            index = int(time / interval)
            visiting_distributions[poi][index] += 1

    # normalize the visiting distributions
    visiting_distributions = visiting_distributions / visiting_distributions.sum(axis=1, keepdims=True)
    return visiting_distributions


def get_visiting_probabilities(trajs, num_pois):
    visiting_probabilities = np.zeros(num_pois)
    for traj in trajs:
        for poi, _ in traj:
            visiting_probabilities[poi] += 1

    # normalize the visiting probabilities
    visiting_probabilities = visiting_probabilities / visiting_probabilities.sum()
    return visiting_probabilities


import numpy as np


def get_discrete_trajectories(poi_coords, visit_probs, visit_dist, query_trajectories, interval=0.5):
    """
    Convert query trajectories with GPS coordinates into discrete trajectories using POI IDs based on
    proximity, visiting distribution probabilities, and overall visiting probabilities.

    Args:
    poi_coords (numpy.ndarray): Array of shape [N, 2] containing GPS coordinates for each POI.
    visit_probs (numpy.ndarray): Array of shape [N] with the overall visiting probability of each POI.
    visit_dist (numpy.ndarray): Array of shape [N, seq_len] indicating the probability of visiting each POI at each timestamp.
    query_trajectories (list): List of trajectories; each trajectory is a list of check-ins, where each check-in is
                               [[latitude, longitude], timestamp].

    Returns:
    list: List of discrete trajectories, where each trajectory is a list of [POI ID, timestamp] check-ins.
    """
    discrete_trajs = []

    for trajectory in query_trajectories:
        discrete_traj = []
        for checkin in trajectory:
            query_coord = np.array(checkin[0])
            query_time = int(checkin[1] / interval)

            # Compute distances from the query point to each POI
            distances = np.linalg.norm(poi_coords - query_coord, axis=1)

            # Incorporate visiting distribution at the specific timestamp and overall visiting probabilities
            adjusted_distances = distances / (visit_dist[:, query_time] * visit_probs)

            # Find the index of the smallest adjusted distance from all non-inf distances
            valid_distances = {}
            for i, dist in enumerate(adjusted_distances):
                if not np.isinf(dist) and not np.isnan(dist):
                    valid_distances[i] = dist

            if len(valid_distances) == 0:
                # select by distance
                nearest_poi_id = np.argmin(distances)
            else:
                nearest_poi_id = min(valid_distances, key=valid_distances.get)

            # Append the POI ID and timestamp to the discrete trajectory
            discrete_traj.append([nearest_poi_id, checkin[1]])

        # Append the completed discrete trajectory to the list of trajectories
        discrete_trajs.append(discrete_traj)

    return discrete_trajs


def save_discrete_trajectories(discrete_trajectories, save_path):
    with open(save_path, 'w') as f:
        for traj in discrete_trajectories:
            # save each trajectory as a sequence of POI IDs and timestamps in one row
            row = ','.join([f'{poi} {time}' for poi, time in traj])
            f.write(row + '\n')


def read_discrete_trajectories(file_path):
    discrete_trajectories = []
    with open(file_path, 'r') as f:
        for line in f:
            traj = line.strip().split(',')
            traj = [[int(checkin.split(' ')[0]), float(checkin.split(' ')[1])] for checkin in traj if checkin != '']
            discrete_trajectories.append(traj)
    return discrete_trajectories


if __name__ == '__main__':
    dataset = 'NYC'
    train_trajectories = read_real_trajectories(f'data/{dataset}/train_set.csv')
    test_trajectories = read_real_trajectories(f'data/{dataset}/test_set.csv')
    original_gps, gps_points, lat_min, lat_max, lon_min, lon_max = read_gps(f'data/{dataset}/gps')

    inter_train_trajectories = transform_poi_trajectories(interpolate(train_trajectories), gps=gps_points, poi=True)
    inter_test_trajectories = transform_poi_trajectories(interpolate(test_trajectories), gps=gps_points, poi=True)
    real_train_trajectories = []
    real_test_trajectories = []
    for traj in train_trajectories:
        real_train_trajectories.append([[gps_points[poi], time] for poi, time in traj])
    for traj in test_trajectories:
        real_test_trajectories.append([[gps_points[poi], time] for poi, time in traj])

    syn_trajs = np.load(f'Gen_traj_gps_{dataset}.npy')
    # syn_masks = np.load('Gen_mask_gps.npy')

    syn_trajs = generate_trajectories_from_gps(syn_trajs)

    # transform the normalized GPS coordinates back to the original scale
    syn_trajs = transform_normalized_gps_to_original(syn_trajs, lat_min, lat_max, lon_min, lon_max)
    inter_test_trajectories = transform_normalized_gps_to_original(inter_test_trajectories, lat_min, lat_max, lon_min, lon_max)
    inter_train_trajectories = transform_normalized_gps_to_original(inter_train_trajectories, lat_min, lat_max, lon_min, lon_max)

    real_test_trajectories = transform_normalized_gps_to_original(real_test_trajectories, lat_min, lat_max, lon_min, lon_max)
    real_train_trajectories = transform_normalized_gps_to_original(real_train_trajectories, lat_min, lat_max, lon_min, lon_max)

    # get the visiting distribution for POIs
    visiting_distributions = get_visiting_distribution(train_trajectories, len(gps_points))
    visiting_probabilities = get_visiting_probabilities(train_trajectories, len(gps_points))

    # get the discrete trajectories
    discrete_syn_trajectories = get_discrete_trajectories(original_gps, visiting_probabilities, visiting_distributions, syn_trajs)











