import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import itertools
import json
import pandas as pd
#from keras.utils import to_categorical


import re
import matplotlib.image as mpimg
from IPython.display import HTML, display
from tabulate import tabulate
import sqlite3


from sklearn.model_selection import train_test_split

import itertools
from scipy.stats import gaussian_kde
from tqdm import tqdm


# Function to move the cursor up by a specified number of lines
def move_cursor_up(lines):
    sys.stdout.write(f'\033[{lines}A')

# Function to clear the current line
def clear_line():
    sys.stdout.write('\033[K')

def calculate_kde(data):
    return gaussian_kde(data)

def calculate_kld(kde_p, kde_q, data_points, epsilon=1e-10):
    p = kde_p(data_points)
    q = kde_q(data_points)

    # Ensure no zero values
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    return np.sum(p * np.log(p / q))


def generate_class_distance(kdes, class1, class2, epsilon=1e-10):
    kl_distance = 0
    for kde1, kde2 in zip(kdes[class1], kdes[class2]):
        data_points = np.linspace(min(kde1.dataset.min(), kde2.dataset.min()), max(kde1.dataset.max(), kde2.dataset.max()), 100)
        kl_distance += calculate_kld(kde1, kde2, data_points, epsilon)
        kl_distance += calculate_kld(kde2, kde1, data_points, epsilon)
    return kl_distance

def generate_class_distance_matrix(kdes, num_classes, epsilon=1e-10):
    distance_matrix = np.zeros((num_classes, num_classes))
    for i in tqdm(range(num_classes), desc="Class Distance Matrix Calculation"):
        for j in range(i + 1, num_classes):
            distance = generate_class_distance(kdes, i, j, epsilon)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def calculate_separability_score(combination, distance_matrix):
    score = 0
    for (i, j) in itertools.combinations(combination, 2):
        score += distance_matrix[i, j]
    return score

def calculate_distance_matrix(data, labels, num_classes, epsilon=1e-10, show_matrix=False, save_matrix_name=None):

    """Use kullback-leibler (kl) distance to calculate the distance between the
    classes
    The KL distance is calculated for each pair of KDE.
    There is one KDE for each class.
    """

    kdes = {cls: [] for cls in range(num_classes)}

    print("Calculating KDEs for each class...")
    for cls in tqdm(range(num_classes), desc="Classes"):
        if len(data[labels == cls]) == 0:
            continue
        else:
            for i in range(data.shape[1]):
                kde = calculate_kde(data[labels == cls,i,:].reshape(-1))
                kdes[cls].append(kde)


    print("Generating class distance matrix using KLD...")
    distance_matrix = generate_class_distance_matrix(kdes, num_classes, epsilon)
    if save_matrix_name is not None:
        np.save(f"distance_matrix_{save_matrix_name}.npy", distance_matrix)
    if show_matrix:

        # Generate heatmap
        plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')

        # Add colorbar
        plt.colorbar()

        # Add title and labels
        plt.title('Distance Matrix Heatmap')
        plt.xlabel('Class')
        plt.ylabel('Class')

        # Show the plot
        plt.show()

    return distance_matrix

def calculate_separability_scores(distance_matrix, num_classes, n):

    class_combinations = itertools.combinations(range(num_classes), n)
    separability_scores = []
    print("Calculating final separability scores...")
    for combination in tqdm(class_combinations, desc="Combinations"):
        score = calculate_separability_score(combination, distance_matrix)
        separability_scores.append((combination, score))

    separability_scores.sort(key=lambda x: x[1], reverse=True)

    top_combinations = separability_scores[:10]
    for comb, score in top_combinations:
        print(f"Combination: {comb}, Score: {score}")

    return separability_scores

def distance_matrix_subset(distance_matrix, subset):
    subset_matrix = np.zeros((len(subset), len(subset)))
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            subset_matrix[i, j] = distance_matrix[subset[i], subset[j]]
            subset_matrix[j, i] = distance_matrix[subset[j], subset[i]]

    plt.imshow(subset_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Distance Matrix Heatmap')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.show()

    return subset_matrix
class regularization_loss(object):

    """Target Firing Rate Loss. Read more in the thesis

     We use a custom layer loss function as the L1 and L2 torch ones were
    are forcing aa fixed target firing rate."""

    def __init__(self, min_hz, max_hz ,time_window, pth = 0.99, device='cuda'):
        """
        Initializes the regularization loss function.

        Args:
            min_hz (float): The minimum desired spike frequency in Hz.
            max_hz (float): The maximum desired spike frequency in Hz.
            time_window (float): The length of the time window in seconds.
            pth (float, optional): The percentile value used to calculate the threshold spike frequency. Defaults to 0.99.
            device (str, optional): The device of the output loss. Defaults to 'cuda'.
        """
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.pth = pth
        self.time_window = time_window
        self.device = device
    # def __call0__(self, spike_count_array: list[torch.float32]) -> torch.float32:
    #     """
    #     Calculates the regularization loss.

    #     Args:
    #         spike_count_array (list[torch.float32]): A list of spike count arrays.

    #     Returns:
    #         torch.float32: The regularization loss.
    #     """
    #     loss = 0
    #     """ [time, batch, channels]"""

    #     for i in range(len(spike_count_array)):

    #         layer_loss = 0

    #         for j in range(spike_count_array[i].shape[2]):
    #             #print('spike_count_array[i].shape',spike_count_array[i].shape)
    #             frequency_list = []

    #             # for z in range(spike_count_array[i].shape[1]):

    #             #     frequency_list.append(torch.sum(spike_count_array[i][:,z,j])/self.time_window)

    #             frequency_list = torch.sum(spike_count_array[i][:,:,j], dim=(0)) / self.time_window
    #             #print('frequency list shape', len(frequency_list))
    #             frequency_matrix = torch.sum(spike_count_array[i], dim=(0, 2)) / self.time_window
    #             frequency_list.sort()

    #             Rpth = frequency_list[int(self.pth*len(frequency_list))]

    #             layer_loss += (F.relu(Rpth - self.max_hz) + F.relu(self.min_hz - Rpth))**2

    #         loss += layer_loss / spike_count_array[i].shape[1]

    #     return loss

    def __call__(self, spike_count_array: list[torch.float32]) -> torch.float32:
        """
        Calculates the regularization loss.

        Args:
            spike_count_array (list[torch.float32]): A list of spike count arrays.

        Returns:
            torch.float32: The regularization loss.
        """

        """ [time, batch, channels]"""
        loss = torch.tensor(0.0)

        for i in range(len(spike_count_array)):

            layer_loss = 0

            frequency_matrix = torch.sum(spike_count_array[i], dim=(0)) / self.time_window
            frequency_matrix = torch.sort(frequency_matrix, dim=-1).values
            Rpth = frequency_matrix[int(self.pth*frequency_matrix.shape[0])]

            for j in range(spike_count_array[i].shape[2]):

                layer_loss += (F.relu(Rpth[j] - self.max_hz) + F.relu(self.min_hz - Rpth[j]))**2

            loss += layer_loss / spike_count_array[i].shape[1]

        return loss.to(self.device)

def compute_output_labels(matrix):
    # Sum the elements along the last dimension
    #print(f'compute_output_labels matrix shape {matrix.shape}')
    summed_matrix = np.sum(matrix, axis=-1)

    # Divide the sum by the number of elements in the second dimension
    divided_matrix = summed_matrix / matrix.shape[1]

    # Find the maximum value along the second dimension and its index
    max_indices = np.argmax(divided_matrix, axis=1)

    return max_indices


def gen_confusion_matrix(predictions, labels, path):

    num_label = max(labels)
    #print(f'prediction shape {predictions.shape} and labels shape{len(labels)}')
    conf_matrix = confusion_matrix(predictions, labels)
    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_label),
                yticklabels=range(num_label))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save the plot
    file_path = os.path.join(path,'confusion_matrix.png')
    plt.savefig(file_path, bbox_inches='tight')





def spike_plot(data,spike_data,  label, save=None):


    line_size = 0.5
    colors_spike = []
    color = []
    handles = []

    if len(spike_data.shape) == 3:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            for d in range(spike_data.shape[1]):
                colors_spike.append([r, g, b])

        plot_data =  spike_data.reshape((spike_data.shape[0]*spike_data.shape[1], spike_data.shape[2])) * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]

    if len(spike_data.shape) == 2:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            colors_spike.append([r, g, b])

        plot_data =  spike_data * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]
        print(f'plot data {plot_data.shape}')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 30))
    fig.suptitle(f'Signlal label {label}')


    for i in range(data.shape[0]):
        ax1.plot(range(data.shape[-1]), data[i, :], color = color[i])
        handles.append(plt.Line2D([0], [0], color=color[i], lw=2, label=f'dimension {i}'))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Original signal dimensions')

    # print(f'colors spike {len(colors_spike)}')
    # print(f"plot data shape {plot_data.shape}")

    ax2.eventplot(plot_data, color=colors_spike, linelengths = line_size)
    ax2.set_xlabel('Spike')
    ax2.set_ylabel('Channels')
    ax2.set_title('Spike Train encoding')
    plt.xlabel('Spike')
    plt.ylabel('Channels')

    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.85))
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

class SearchSpaceUpdater(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def WisdmDf2Np(path,save_path, time_window=2, overlap =0, subset=0):
    phone = pd.read_pickle(path)

    ##### VARIABLES TO SET TIME AND ACTIVITIES SUBSET TO BE USED #####
    #time_window s of activity
    # 50% overlap has been used for the whole dataset (subset=0) to compare with baseline results of paper "A deep learning approach for human activities recognition from multimodal sensing devices"
    #overlap  overlapping fraction (FRACTION with respect to unity, NOT PERCENTAGE)
    #subset  use 0 to select all of the 18 activities
    ##################################################################  # activities re-ordered according to non-hand, hand general and hand eating subsets
    act_map = {
        'A': 'walking',
        'B': 'jogging',
        'C': 'stairs',
        'D': 'sitting',
        'E': 'standing',
        'M': 'kicking',
        'P': 'dribbling',
        'O': 'catch',
        'F': 'typing',
        'Q': 'writing',
        'R': 'clapping',
        'G': 'teeth',
        'S': 'folding',
        'J': 'pasta',
        'H': 'soup',
        'L': 'sandwich',
        'I': 'chips',
        'K': 'drinking',
    }

    LABEL_SUBSETS = {
        1: [0, 1, 2, 3, 4, 5], # for smartphone and smartwatch
        2: [6, 7, 8, 9, 10, 11, 12], # for smartwatch only
        3: [13, 14, 15, 16, 17], # for smartwatch only
    }

    # LABEL_SUBSETS = {
    #     0: [1,6,7,8,13,14,17], # for smartphone and smartwatch
    #     1: [11,12,13,14,15,16,17], # for smartwatch only
    # }

    window_size = int(20*time_window) # 20 Hz sampling times the temporal length of the window
    stride = int(window_size*(1-overlap))

    frames = []
    for i in range(0, len(phone)-window_size, stride):
        window = phone.iloc[i:i+window_size]
        if window['activity'].nunique() == 1:
            frames.append(window)

    #activities = sorted(act_map.keys())
    activities = act_map.keys()
    activity_encoding = {v: k for k, v in enumerate(activities)}

    X_list = []
    y_list = []

    #for each frame replace label with activity
    for frame in frames:
        X_list.append(frame[['watch_accel_x', 'watch_accel_y', 'watch_accel_z', 'watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z']].values)
        y_list.append(activity_encoding[frame.iloc[0]['activity']])

    if subset!=0:
        idx = [ii for ii,lbl in enumerate(y_list) if lbl in LABEL_SUBSETS[subset]] # list(ACTIVITIES_LABEL.keys())[:6]
        X_sub = [X_list[ii] for ii in idx] #[jj for jj in X[:,] if jj in idx]
        y_sub = [y_list[jj] for jj in idx] #[kk for kk in y if kk in idx]

        X = np.array(X_sub)
        #y = np.array(to_categorical(y_sub))[:,min(LABEL_SUBSETS[subset]):]
        y = np.array(y_sub)[:,min(LABEL_SUBSETS[subset]):]

    else:
        X = np.array(X_list)
        #y = np.array(to_categorical(y_list))
        y = np.array(y_list)
        print(f'y_list {y.shape}')
        y = one_hot_encode(y, np.max(y)+1)
        print(f'y_list {y.shape}')
    # 60/20/20 split
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    # if subset == 0:
    #     subset_name = '7BC'
    # elif subset == 1:
    #     subset_name = '7WC'

    savefile = save_path +'/data_watch'+'_subset_'+str(subset)+'_'+str(window_size)
    print (f'Saving data to {savefile}')
    np.savez(savefile, X_train, X_val, X_test, y_train, y_val, y_test)


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]



def merge_dicts(dict1, dict2)-> dict:
    merged_dict = {}
    for key in dict1.keys():
        merged_dict[key] = [dict1[key], dict2[key]]
    print(merged_dict)
    return merged_dict


def nni_query(tiral_sqlite_path, show=True) -> dict:

    # Connect to the SQLite database
    connection = sqlite3.connect(tiral_sqlite_path)
    cursor = connection.cursor()

    query = f"""
        SELECT trialjobId, sequence, data
        FROM MetricData
        WHERE trialjobId IN (
            SELECT trialjobId
            FROM MetricData
            WHERE type = 'FINAL'
            GROUP BY trialjobId
            ORDER BY data DESC
            LIMIT 5
        )
        AND type = 'PERIODICAL'
        ORDER BY trialjobId, sequence;
    """


    # Execute the query
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()
    print(results)
    score_dict = {}
    if len(results) == 0:
        print('NO METRICS FOUND')
        return
    if "{" not in results[0][2]:

        print("found single Score")
        query = """
                UPDATE MetricData
                SET data = replace(data, '"', '');
                """

        cursor.execute(query)
        connection.commit()

        query = f"""
                SELECT
                    trialjobId,
                    sequence,
                    CAST(data AS REAL) AS d,
                    MAX(CAST(data AS REAL)) OVER (PARTITION BY trialjobId) AS max_data
                FROM MetricData
                WHERE trialjobId IN (
                    SELECT trialjobId
                    FROM MetricData
                    WHERE type = 'FINAL'
                    GROUP BY trialjobId
                    ORDER BY CAST(data AS REAL) DESC
                    LIMIT 5
                ) AND type = 'PERIODICAL'
                ORDER BY  max_data DESC,trialjobId,sequence;
                """
            # Execute the query
        cursor.execute(query)
        results = cursor.fetchall()

        query = f"""
                UPDATE MetricData
                SET data = '"' || data || '"'
                WHERE data IS NOT NULL;
                """
        cursor.execute(query)
        connection.commit()
    else:
        print("found multiple metrics, tracking only default")
        query = f"""
                UPDATE MetricData
                SET data = replace(data, '\\"', '"');
                """
        cursor.execute(query)
        connection.commit()
        query = f"""
                UPDATE MetricData
                SET data = replace(data, '"{{', '{{');
                """
        cursor.execute(query)
        connection.commit()
        query = f"""
                UPDATE MetricData
                SET data = replace(data, '}}"', '}}');
            """

        cursor.execute(query)
        connection.commit()
        query = f"""
                SELECT trialjobId,
                        sequence,
                        CAST(JSON_EXTRACT(data, '$.default') AS REAL) as metrics,
                        MAX(CAST(JSON_EXTRACT(data, '$.default') AS REAL)) OVER (PARTITION BY trialjobId) AS max_data
                FROM MetricData
                WHERE trialjobId IN (
                    SELECT trialjobId
                    FROM MetricData
                    WHERE type = 'FINAL'
                    GROUP BY trialjobId
                    ORDER BY MAX(CAST(JSON_EXTRACT(data, '$.default') AS REAL)) DESC
                    LIMIT 5
                )
                AND type = 'PERIODICAL'
                ORDER BY max_data DESC, trialjobId, sequence;
            """

        cursor.execute(query)
        results = cursor.fetchall()

        query = f"""
                UPDATE MetricData
                SET data = replace(data, '"', '\\"');
                """
        cursor.execute(query)
        connection.commit()
        query = f"""
                UPDATE MetricData
                SET data = replace(data, '{{', '"{{');
                """
        cursor.execute(query)
        connection.commit()
        query = f"""
                UPDATE MetricData
                SET data = replace(data, '}}', '}}"');
            """

        cursor.execute(query)
        connection.commit()

                # Execute the query


    # Fetch the results
    #print(results)
    for row in results:
        trialjob_id = row[0]
        sequence = row[1]

    # If trialjob_id is not already in the dictionary, add it with an empty list
        if trialjob_id not in score_dict:
            score_dict[trialjob_id] = []

        # Append the data to the list, maintaining the order by sequence
        score_dict[trialjob_id].append(row[2])
    #print('score dict', score_dict)
    query2 = f"""
            SELECT trialjobId, data
            FROM TrialJobEvent
            WHERE event = 'RUNNING'
            AND trialjobId IN ({', '.join('?' for _ in score_dict.keys())});
            """
    cursor.execute(query2, list(score_dict.keys()))
    results2 = cursor.fetchall()
    connection.close()

    params_dict = {}
    for row in results2:
        trialjob_id = row[0]
        parameters = json.loads(row[1])
        # If trialjob_id is not already in the dictionary, add it with an empty list
        if trialjob_id not in params_dict:
        # Append the data to the list, maintaining the order by sequence
            params_dict[trialjob_id] = (parameters['parameters'])
    print('params dict', params_dict)
    print('score dict', score_dict)
    merged_dict = merge_dicts(score_dict, params_dict)

    if show == True:

        values = next(iter(merged_dict.values()))
        header = [['Trials Name'] + list(values[1].keys()) + ['Score']]
        plt.figure(figsize=(15, 10))

        for key, value in merged_dict.items():

            plt.plot(range(len(value[0])), value[0], label=f'{key}, best:{max(value[0]):.4f}')
            rows = [key] + list(value[1].values()) + [max(value[0])]
            header.append(rows)
        plt.legend()
        plt.show()

        print(tabulate(header, headers='firstrow', tablefmt='grid'))

    return merged_dict


def show_results(path, experiment_code=None, print_content=False):

    if os.path.isdir(path):
            # Ottieni la lista di tutti i file e le directory nel path
            if experiment_code is None:
                dir_content = os.listdir(path)
            else:
                dir_content = [experiment_code]

            # Scansiona tutti gli elementi nel path
            for element in dir_content:
                print(f'experiment codename {element}')
                db_path = os.path.join(path, element,'db/nni.sqlite')
                print('experiment folder', os.path.join(path, element))
                if os.path.isfile(db_path):
                    experimet_best_dict = nni_query(db_path, show=True)
                    if experimet_best_dict is not None and print_content == True:
                        for trial in experimet_best_dict.keys():
                            loss_trial_path = os.path.join(path, element,'environments/local-env/trials/', trial, 'Trained/loss.txt')
                            data = np.loadtxt(loss_trial_path, skiprows=1)
                            plt.figure(figsize=(15, 10))
                            plt.plot(range(len(data[:,0])), data[:,0], label='train')
                            plt.plot(range(len(data[:,1])), data[:,1], label='test')
                            plt.legend()
                            plt.title(f'{trial} loss graph')
                            plt.show()
                            image_path = os.path.join(path, element,'environments/local-env/trials', trial, 'Trained/confusion_matrix.png')
                            img = mpimg.imread(image_path)

                            # Visualizza l'immagine
                            plt.imshow(img)
                            plt.show()
                            gif_path = os.path.join(path, element,'environments/local-env/trials', trial, 'gifs')
                            if not os.path.exists(gif_path):
                                print("The gif_path does not exist.")
                            else:
                                gifs_names= os.listdir(gif_path)
                                if len(gifs_names) > 0 :
                                    end_char = '.gif'
                                    start_char = '_label'

                                    gif_td = lambda gif: f'<td> <img src="{gif}" alt="Drawing" style="height: 300px;"/> </td>'
                                    html = '<table>'
                                    html += '<tr><td align="center"><b>Input</b></td><td><b>Output</b></td></tr>'
                                    for i in range(5):
                                        pattern1 = re.compile(f'^inp{re.escape(str(i))}_.*\.gif$')
                                        pattern2 = re.compile(f'^out{re.escape(str(i))}_.*\.gif$')
                                        pattern3 = re.compile(f'{re.escape(start_char)}(.*?){re.escape(end_char)}')
                                        try:
                                            image_name1 =  [file for file in gifs_names if pattern1.match(file)][0]
                                            image_name2 =  [file for file in gifs_names if pattern2.match(file)][0]
                                            label = pattern3.search(image_name1).group(1)
                                            image_path1 = os.path.join(gif_path, image_name1)
                                            image_path2 = os.path.join(gif_path, image_name2)
                                        except:
                                            print("found error in gifs names")

                                        html += '<tr>'
                                        html += f'<td><b>Label: {label}</b></td>'
                                        html += gif_td(image_path1)
                                        html += gif_td(image_path2)
                                        html += '</tr>'
                                    html += '</table>'
                                    display(HTML(html))

                else:
                    print('no database file found in the experiment gfolder \n')
    else:
        print(f"given path is not a directory: {path}")

#### given the modularity of the saving, modular finction defined by the user


def extract_strings_at_indices(indices, strings, print_output=False):
    extracted_strings = [strings[idx] for idx in indices]
    if print_output:
        print(extracted_strings)
    return extracted_strings

def json_to_vector(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    vector = []
    for value in data:
        vector.append(value)
    return vector


def MseScore_plot(combination, distance_matrix, plot=False):
    distances = distance_matrix[np.ix_(combination, combination)]
    sum_score = sum(distances.flatten())/2
    distances_distribution = distances.flatten()
    distances_distribution /= np.sum(distances_distribution)  # Normalize the distribution
    uniform_distribution = np.full_like(distances_distribution, 1 / len(distances_distribution))
    mse = mean_squared_error(distances_distribution, uniform_distribution)
    distances_distribution = distances_distribution[distances_distribution != 0]  # Remove zeros
    uniform_distribution = np.full_like(distances_distribution, 1 / len(distances_distribution))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(distances_distribution, label='Distance Distribution')
        plt.plot(uniform_distribution, label='Uniform Distribution', linestyle='--')

        # Highlight the moving range with dotted lines
        plt.fill_between(range(len(distances_distribution)), distances_distribution, uniform_distribution, color='gray', alpha=0.2, linestyle=':')

        # Calculate and plot the range amplitude
        range_amplitude = distances_distribution.max() - distances_distribution.min()
        plt.axhline(y=distances_distribution.max(), color='r', linestyle=':', label='Max Amplitude')
        plt.axhline(y=distances_distribution.min(), color='b', linestyle=':', label='Min Amplitude')
        plt.title(f'Distance Distribution vs Uniform Distribution\nMSE: {mse:.3e}, Sum:{sum_score:.2f}, Amplitude: {range_amplitude:.6f}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.ylim(0, 0.055)  # Set y-axis range
        plt.legend()
        plt.show()


def calculate_separability_scores_v2(distance_matrix, num_classes, n):
    class_combinations = itertools.combinations(range(num_classes), n)
    separability_scores = []
    print("Calculating final separability scores...")
    for combination in tqdm(class_combinations, desc="Combinations"):
        distances = distance_matrix[np.ix_(combination, combination)]
        median_distance = np.median(distances.flatten())

        # Calculate the distances distributionsdsdsfdsfsdfsdfsdfdsfdsfdsewrewrwerewr234234234234234
        distances_distribution = distances.flatten()
        distances_distribution /= np.sum(distances_distribution)  # Normalize the distribution

        # Calculate the uniform distribution
        uniform_distribution = np.full_like(distances_distribution, 1 / len(distances_distribution))

        # Calculate the mse between the distributions
        mse = mean_squared_error(distances_distribution, uniform_distribution)
        score = calculate_separability_score(combination, distance_matrix)
        separability_scores.append((combination, score, median_distance, mse))

    separability_scores.sort(key=lambda x: x[1], reverse=True)

    top_combinations = separability_scores[:10]
    for comb, score, median_dist, mse in top_combinations:
        print(f"Combination: {comb}, Score: {score}, Median Distance: {median_dist}, mse: {mse}")

    return separability_scores

def search_combination(separability_scores, combination):

    for comb, score, median_dist, mse in separability_scores:
        if comb == combination:
            print(f"Combination: {comb}, Score: {score}, Median Distance: {median_dist}, mse: {mse}")
            break
    else:
        print("Combination not found in separability scores.")


def standardize_scores_range(separability_scores):
    score_sum = [x[1] for x in separability_scores]
    score_mse = [x[3] for x in separability_scores]
    separability_tmp = [list(x) for x in separability_scores]
    sum_max = max(score_sum)
    sum_min = min(score_sum)
    mse_max = max(score_mse)
    mse_min = min(score_mse)
    for i in range(len(separability_tmp)):
        sum_norm = (separability_tmp[i][1] - sum_min) / (sum_max - sum_min)
        mse_norm = (separability_tmp[i][3] - mse_min) / (mse_max - mse_min)
        separability_tmp[i][1] = sum_norm
        separability_tmp[i][3] = mse_norm
    return separability_tmp
