import pandas as pd
import simpsom as sps
import numpy as np
from sklearn import preprocessing
from simpsom.plots import scatter_on_map


df = pd.read_csv('../dusha/crowd_small.csv')
emotions = df[['emotion', 'label']]
pos_sad_df = df[(df['emotion'] == 'positive') | (df['emotion'] == 'sad')]
pos_sad_df = pos_sad_df.drop(columns=['label', 'emotion'])
df = df.drop(columns=['label', 'emotion'])
data = df.to_numpy(dtype='float')
pos_sad_data = pos_sad_df.to_numpy(dtype='float')

# scaler = preprocessing.StandardScaler().fit(dusha)
# scaled_data = scaler.transform(dusha)
scaler = preprocessing.StandardScaler().fit(pos_sad_data)
scaled_data = scaler.transform(pos_sad_data)

# quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100, random_state=0).fit(dusha)
# quantile_data_scaled = quantile_transformer.transform(dusha)

# scaler = preprocessing.MinMaxScaler().fit(dusha)
# scaled_data = scaler.transform(dusha)

net = sps.SOMNet(10, 10, scaled_data, topology='hexagonal',
                 init='random', metric='euclidean',
                 neighborhood_fun='gaussian', PBC=False,
                 random_seed=32, GPU=True, CUML=False,
                 debug=True)

net.train(train_algo='batch', start_learning_rate=0.01, epochs=-1, batch_size=10000,
          early_stop="mapdiff", early_stop_tolerance=1e-5)  # early_stop="mapdiff"
# net.save_map()

net.plot_map_by_difference(show=True, print_out=True)
net.plot_convergence(show=True, print_out=True)

# flat_data = scaled_data.reshape(scaled_data.shape[0], -1)
# projection = net.project_onto_map(flat_data)

# scatter_on_map([projection[train_y == i][:1000] for i in range(10)],
#                [[node.pos[0], node.pos[1]] for node in net.nodes_list],
#                net.polygons, color_val=None,
#                show=True, print_out=True, cmap=pylette)
