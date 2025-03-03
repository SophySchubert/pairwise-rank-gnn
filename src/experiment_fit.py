from datetime import datetime
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from itertools import combinations
from tqdm import tqdm
from scipy.stats import rankdata


if __name__ == '__main__':
    start_time = datetime.now()
    ######################################################################
    # SETUP
    # config = setup_experiment(sys.argv[1])
    # logger = setup_logger(config['folder_path'])
    #
    print(f"Starting at {start_time}")
    # logger.info(f"Experiment saved in {config['folder_path']}")
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    VALID_SPLIT = 0.8
    TEST_SPLIT = 0.1

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # shutil.copy('src/models/prgnn.py', config['folder_path']+'/model.py')
    ######################################################################
    class CustomDataLoader(torch.utils.data.DataLoader):
        def __init__(self, pairs_and_targets, dataset, batch_size=1, shuffle=False, **kwargs):
            super().__init__(pairs_and_targets, batch_size=batch_size, shuffle=shuffle, **kwargs)
            self.entire_dataset = dataset

        def __iter__(self):
            for batch in super().__iter__():
                batch = self.augment_batch(batch)
                yield batch

        def augment_batch(self, batch):
            # batch is a list of pairs and targets
            idx_a, idx_b, target = zip(*[(x[0], x[1], x[2]) for x in batch])
            data = self.get_data_from_indices(idx_a, idx_b)
            idx_a, idx_b = self.reindex_ids(idx_a, idx_b)

            # Create a DataBatch object
            data_batch = Batch.from_data_list(data)

            # Add idx_a and idx_b to the DataBatch object
            data_batch.idx_a = torch.tensor(idx_a)
            data_batch.idx_b = torch.tensor(idx_b)
            data_batch.y = torch.tensor(target)

            return data_batch

        def get_data_from_indices(self, idx_a, idx_b):
            ids = np.unique(np.concatenate((idx_a, idx_b)))
            required_data = [self.entire_dataset[int(i)] for i in ids]
            return required_data

        def reindex_ids(self, idx_a, idx_b):
            ids = np.unique(np.concatenate((idx_a, idx_b)))
            idx_a = np.array(idx_a)
            idx_b = np.array(idx_b)

            # Create a mapping from unique elements to the range [0, length)
            mapping = {element: idx for idx, element in enumerate(ids)}

            # Apply the mapping to both arrays
            mapped_a = np.array([mapping[element] for element in idx_a])
            mapped_b = np.array([mapping[element] for element in idx_b])

            return mapped_a, mapped_b
    #
    def sample_preference_pairs(graphs):
        c = [(a, b, check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
        return np.array(c)

    def check_util(data, index_a, index_b):
        a = data[index_a]
        b = data[index_b]
        util_a = a.y
        util_b = b.y
        if util_a >= util_b:
            return 1
        else:
            return 0

    def iterate_train_random(elements):
        objects = elements
        utilities = np.array([e.y.item() for e in elements])  # Convert tensor values to a numpy array
        sort_idx = np.argsort(utilities, axis=0)
        olen = len(objects)
        seed = SEED + olen
        pair_count = (olen * (olen - 1)) // 2
        sampling_ratio = 1
        sample_size = min(int(sampling_ratio * pair_count), pair_count)
        rng = np.random.default_rng(seed)

        sample = rng.choice(pair_count, sample_size, replace=False)
        sample_b = (np.sqrt(sample * 2 + 1 / 4) + 1 / 2).astype(int)  # Convert to integer type
        sample_a = (sample - (sample_b * (sample_b - 1)) // 2).astype(int)  # Convert to integer type
        idx_a = sort_idx[sample_a]
        idx_b = sort_idx[sample_b]

        return idx_a, idx_b

    def get_target(data, indices_a, indices_b):
        assert len(indices_a) == len(indices_b)
        util_a = np.array([data[idx].y.item() for idx in indices_a])
        util_b = np.array([data[idx].y.item() for idx in indices_b])
        target = (util_a > util_b).astype(int)
        return target

    # dataset = TUDataset(root='/tmp/aspirin', name='aspirin', use_node_attr=True)
    dataset = PygGraphPropPredDataset(name='ogbg-molesol') #ogbg-molesol, ogbg-molfreesolv, ogbg-mollipo

    # Split the dataset into training, validation, and test sets
    train_size = int(VALID_SPLIT * len(dataset))
    valid_size = int(TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    # Split the dataset
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # create pairs and targets
    train_prefs = sample_preference_pairs(train_dataset)
    # train_idx_a, train_idx_b = iterate_train_random(train_dataset)
    # # target = get_target(train_dataset, idx_a, idx_b)
    # train_target = np.zeros(len(train_idx_a))#due to argsort in iterate_train_randomB all targets are 0
    # train_prefs = np.array(list(zip(train_idx_a, train_idx_b, train_target)))

    valid_prefs = sample_preference_pairs(valid_dataset)
    # valid_idx_a, valid_idx_b = iterate_train_random(valid_dataset)
    # valid_target = np.zeros(len(valid_idx_a))
    # valid_prefs = np.array(list(zip(valid_idx_a, valid_idx_b, valid_target)))

    test_prefs = np.array(list(zip(range(len(test_dataset)), range(len(test_dataset)), range(len(test_dataset)))))
    # test_idx_a, test_idx_b = iterate_train_random(test_dataset)
    # test_target = np.zeros(len(test_idx_a))
    # test_prefs = np.array(list(zip(test_idx_a, test_idx_b, test_target)))
    test_ranking = rankdata([g.y.item() for g in test_dataset], method='dense')
    print(f'len test_dataset:{len(test_dataset)}')
    print(f'len test_ranking:{len(test_ranking)}')

    # Create data loaders
    train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_dataset), shuffle=False)


    class GCN(torch.nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 64)
            self.conv2 = GCNConv(64, 64)
            self.fc = torch.nn.Linear(64, 1)  # Output 1 for regression

        def pref_lookup(self, util, idx_a, idx_b):
            util = util.squeeze()
            pref_a = torch.gather(util, 0, idx_a)
            pref_b = torch.gather(util, 0, idx_b)
            return pref_a, pref_b

        def forward(self, data):
            x, edge_index, batch, idx_a, idx_b = data.x, data.edge_index, data.batch, data.idx_a, data.idx_b
            print(f'len idx_a:{len(idx_a)}')
            print(f'len x:{len(x)}')
            x = x.type(torch.FloatTensor).to(device)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = torch.nn.functional.relu(x)
            x_util = self.fc(x)
            print(f'len x_util:{len(x_util)}')
            x_a, x_b = self.pref_lookup(x_util, idx_a, idx_b)
            out = x_b - x_a
            return out, x_util



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()


    def train():
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            out = out[0]
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()


    def evaluate(loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                out = out[0]
                # print(f'out:{out}')
                out = out.float()
                error += criterion(out, data.y.float())  # Ensure target shape matches output shape
        return error / len(loader)

    def predict(loader):
        model.eval()
        with torch.no_grad():
            for data in loader:
                print(f'len data:{len(data)}')
                print(f'len data.y:{len(data.y)}')
                data = data.to(device)
                out = model(data)
                utils = out[1].detach().cpu().numpy()
                print(f'len utils:{len(utils)}')
                utils_ranked = rankdata(utils, method='dense')
        return utils_ranked



    print(f'Starting training loop')
    for epoch in range(2):
        train()
        train_error = evaluate(train_loader)
        valid_error = evaluate(valid_loader)
        print(f'Epoch: {epoch + 1}, Train Error: {train_error:.4f}, Valid Error: {valid_error:.4f}')

    test_error = evaluate(test_loader)
    print(f'Test Error: {test_error:.4f}')
    print(f'Training took {datetime.now() - start_time}')
    print(f'Starting Prediction of ranking')
    predicted_ranking = predict(test_loader)
    out = model









































































    # data_loader_train = CustomDisjointedLoader(train_graphs, pairs_and_targets_train, config, node_level=False, batch_size=config['batch_size'], epochs=config['epochs'], shuffle=True)
    #
    # pairs_and_targets_test = sample_preference_pairs2(test_graphs)
    # data_loader_test = CustomDisjointedLoader(test_graphs, pairs_and_targets_test, config, node_level=False, batch_size=config['batch_size'], epochs=1, shuffle=True)
    #
    # #########
    # # model #
    # #########
    # def pref_lookup(X, pref_a, pref_b):
    #     X_a = tf.gather(X, pref_a, axis=0)
    #     X_b = tf.gather(X, pref_b, axis=0)
    #
    #     return X_a, X_b
    #
    # def combine_model(config):
    #     from tensorflow.keras.layers import Input, Dense, Subtract, Activation
    #     from spektral.layers import ECCConv
    #
    #     x_in = Input(shape=(None,9))
    #     a_in = Input(shape=(None,None))
    #     e_in = Input(shape=(None,3))
    #     i_in = Input(shape=(None,1))#can be ignored
    #     idx_a = Input(shape=(None,), dtype=tf.int32)
    #     idx_b = Input(shape=(None,), dtype=tf.int32)
    #
    #     # x_in = tf.cast(x_in, tf.float32)
    #     # a_in = tf.cast(a_in, tf.float32) #a_in.with_values(tf.cast(a_in.values, tf.float32))
    #     # e_in = tf.cast(e_in, tf.float32)
    #
    #     outs = ECCConv(32, activation='relu')([x_in, a_in, e_in])
    #     outs = ECCConv(32, activation='relu')([outs, a_in, e_in])
    #     X_util = Dense(config['n_out'], activation=None)(outs)
    #
    #     X_a, X_b = pref_lookup(X_util, idx_a, idx_b)
    #     out = X_b - X_a
    #
    #     # Create the new model with the additional layers
    #     m_infer = tf.keras.Model(inputs=[x_in, a_in, e_in, i_in, idx_a, idx_b], outputs=X_util, name="InferenceModel")
    #     m = tf.keras.Model(inputs=[x_in, a_in, e_in, i_in, idx_a, idx_b], outputs=out, name="PairwiseModel")
    #
    #     return m, m_infer
    #
    #
    # model, model_infer = combine_model(config) #
    # # model = setup_model(config)
    #
    # model.compile(optimizer=Adam(config['learning_rate']),
    #               loss=BinaryCrossentropy(from_logits=True),
    #               metrics=[BinaryAccuracy(threshold=.5)],
    #               # run_eagerly=True
    #               )
    #
    #
    # ################################################################################
    # # Fit model
    # ################################################################################
    # # hs = model.fit(loader_tr.load(), epochs=config['epochs'], verbose=1)
    # hs = model.fit(data_loader_train.load(), epochs=config['epochs'], verbose=1)
    # ################################################################################
    # # Evaluate model
    # ################################################################################
    # # logger.info("Testing model")
    # # pred_utils = model_infer.predict(loader_te.load(), steps=loader_te.steps_per_epoch)
    # #TODO: ranking von scipy einbauen wie in load.py
    #
    # # print("end1")
    # # print(pred_utils.shape)
    # # print("end2")
    # # logger.info(f"Done. Test loss: {loss} - Test Accuracy: {acc}")
    ###############################################################################
    # df = pd.DataFrame({'loss': hs.history['loss'], 'binary_accuracy': hs.history['binary_accuracy']})
    # df.to_csv(config['folder_path'] + '/loss_acc.csv', index=False)
    ###############################################################################
    # import not matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    #
    # # Plot loss
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss', color='tab:red')
    # ax1.plot(range(1, len(epoch_loss) + 1), epoch_loss, color='tab:red', label='Loss')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    #
    # # Create a second y-axis for accuracy
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Accuracy', color='tab:blue')
    # ax2.plot(range(1, len(epoch_acc) + 1), epoch_acc, color='tab:blue', label='Accuracy')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    #
    # # Add a title and show the plot
    # fig.suptitle('Training Loss and Accuracy per Epoch')
    # fig.tight_layout()
    # plt.savefig(config['folder_path'] + '/loss_acc.png')
    # plt.show()

