import torch
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything
import numpy as np
from datetime import datetime
import sys
from shutil import copyfile
import pickle
import os.path
from random import shuffle

from misc import setup_experiment, setup_logger, config_add_nagsl, _read_config
from data.load import get_data
from data.loader import CustomDataLoader
from models.torch_gnn import RankGNN, RankGAT, PairRankGNN, PairRankGNN2, RANet
from models.NAGSL.NAGSL import NAGSLNet
from data.misc import compare_rankings_with_kendalltau, rank_data, train, evaluate, predict, preprocess_predictions, retrieve_preference_counts_from_predictions, check_trans, check_antisymmetry

if __name__ == '__main__':
    start_time = datetime.now()
    ######################################################################
    # CONFIG + SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(sys.argv) == 2:
        config = setup_experiment(sys.argv[1])
        config['start_epoch'] = 0
    else:
        torch.serialization.add_safe_globals([torch.nn.BCELoss])
        config = _read_config(sys.argv[1])
        config['folder_path'] = "./" + "/".join(sys.argv[2].split("/")[:-1])
        state_dict = torch.load(sys.argv[2], map_location=device)
        config['start_epoch'] = state_dict['epoch'] + 1

    config['device'] = device

    logger = setup_logger(config['folder_path'], lvl=config['logger']['level'])
    if len(sys.argv) == 2:
        # only save torch_gnn.py if it is a new experiment
        copyfile('src/models/torch_gnn.py', config['folder_path']+'/torch_gnn.py')
        logger.info(f'Starting at {start_time}')
    if len(sys.argv) == 4:
        logger.info(f'Evaluating Properties of Transitivity and Antisymmetry {start_time}')
    else:
        logger.info(f'Restarting training at {start_time} - Epoch number {config["start_epoch"]} ')

    seed_everything(config['seed']) # set seed for reproducibility
    ######################################################################
    # DATA PREPARATION
    # Load + prep data
    if config['mode'] == 'nagsl_attention':
        config = config_add_nagsl(config)
    if (config['mode'] == 'default' or config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention' or config['mode'] == 'rank_mask'):
        train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking, antisymmetry_prefs, transitivity_prefs = get_data(config)
    elif config['mode'] == 'fc' or config['mode'] == 'fc_extra':
        # Saving and loading of pickled data, to speed up the process if the same data is used
        if os.path.isfile(f"data/{config['data_name']}.pkl"):
            with open(f"data/{config['data_name']}.pkl", 'rb') as f:
                train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking, as_dataset, trans_dataset, config['num_node_features'], config['max_num_nodes'] = pickle.load(f)
        else:
            train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking, as_dataset, trans_dataset = get_data(config)
            with open(config['folder_path']+f"/{config['data_name']}.pkl", 'wb') as f:
                 pickle.dump((train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking, as_dataset, trans_dataset, config['num_node_features'], config['max_num_nodes']), f)
    else:
        raise ValueError(f'Unknown mode {config["mode"]}')
    logger.info(f'Config: {config}')

    data_prep_end_time = datetime.now()
    logger.info(f'Data prep took {data_prep_end_time - start_time}')

    # DataLoaders
    if config['mode'] == 'default':
        train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'],
                                        shuffle=True, mode=config['mode'], config=config)
        valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'],
                                        shuffle=False, mode=config['mode'], config=config)
        test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_dataset),
                                       shuffle=False, mode=config['mode'], config=config)
    elif config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention' or config['mode'] == 'rank_mask':
        train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'],
                                        shuffle=True, mode=config['mode'], config=config)
        valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'],
                                        shuffle=False, mode=config['mode'], config=config)
        test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_prefs),
                                       shuffle=False, mode=config['mode'], config=config)
    else:# fc, fc_extra
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    ######################################################################
    # MODEL + TRAINING
    # Create model, optimizer, and loss function
    if config['mode'] == 'default':
        model = RankGNN(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'gat_attention':
        model = RankGAT(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'fc':
        model = PairRankGNN(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'fc_extra':
        model = PairRankGNN2(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'nagsl_attention':
        model = NAGSLNet(config)
    elif config['mode'] == 'rank_mask':
        model = RANet(num_node_features=config['num_node_features'], device=device, config=config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCELoss()

    if len(sys.argv) > 2:
        # Load model state
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        criterion = state_dict['losslogger']

    model = model.to(device)
    torch.compile(model, backend="cudagraphs")

    if len(sys.argv) == 4:
        # evaluate model on ranking properties
        if config['mode'] == 'default':
            as_loader = CustomDataLoader(antisymmetry_prefs, test_dataset, batch_size=len(test_dataset),
                                         shuffle=False, mode=config['mode'], config=config)
            trans_loader = CustomDataLoader(transitivity_prefs, test_dataset, batch_size=len(test_dataset),
                                            shuffle=False, mode=config['mode'], config=config)
        elif config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention' or config['mode'] == 'rank_mask':
            as_loader = CustomDataLoader(antisymmetry_prefs, test_dataset, batch_size=len(antisymmetry_prefs),
                                         shuffle=False, mode=config['mode'], config=config)
            trans_loader = CustomDataLoader(transitivity_prefs, test_dataset, batch_size=len(transitivity_prefs),
                                            shuffle=False, mode=config['mode'], config=config)
        else:
            as_loader = DataLoader(as_dataset, batch_size=len(as_dataset), shuffle=False)
            trans_loader = DataLoader(trans_dataset, batch_size=len(trans_dataset), shuffle=False)

        trans_result = check_trans(model, trans_loader, device, config['mode'])
        as_result = check_antisymmetry(model, as_loader, device, config['mode'])
        logger.info(f' Antisymmetry-Score: {as_result:.4f}, Transitivity-Score: {trans_result:.4f}')
        sys.exit()


    # Cache the DataLoader to speed up training
    train_loader_cached = []
    valid_loader_cached = []
    test_loader_cached = []
    for data in train_loader:
        train_loader_cached.append(data)
    for data in valid_loader:
        valid_loader_cached.append(data)
    for data in test_loader:
        test_loader_cached.append(data)

    logger.info(f'Starting training loop')
    training_start_time = datetime.now()

    # Training loop
    for epoch in range(config['start_epoch'], config['epochs']):
        # shuffle cached dataloader
        shuffle(train_loader_cached)
        shuffle(valid_loader_cached)
        train(model, train_loader_cached, device, optimizer, criterion, config['mode'])
        torch.cuda.empty_cache() # empty cache to avoid no longer needed data in GPU memory
        train_error, test_acc = evaluate(model, train_loader_cached, device, criterion, config['mode'])
        torch.cuda.empty_cache()
        valid_error, valid_acc = evaluate(model, valid_loader_cached, device, criterion, config['mode'])
        torch.cuda.empty_cache()
        logger.info(f'Epoch: {epoch}, Train Error: {train_error:.4f}, Valid Error: {valid_error:.4f}, '
                    f'Train Acc: {test_acc:.4f}, Valid Acc: {valid_acc:.4f}')
        # save model and state every logging_interval epochs
        if epoch % config['logging_interval'] == 0:
            torch.save(model.state_dict(), config['folder_path'] + f'/epoch{epoch}_model.pt')
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'losslogger': criterion}
            torch.save(state, config['folder_path'] + f'/epoch{epoch}_state.pt')
            if config['mode'] == 'default' or config['mode'] == 'rank_mask':
                predicted_pref, predicted_util = predict(model, test_loader_cached, device)
                raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
                cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
                predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs,
                                                                             max_range=len(test_ranking))
            else:
                predicted_pref, _ = predict(model, test_loader_cached, device, config['mode'])
                raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
                cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
                predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs,
                                                                             max_range=len(test_ranking))

            predicted_ranking = rank_data(predicted_util)
            logger.info(f'length of rankings is the same: {len(predicted_ranking) == len(test_ranking)}')
            tau, p_value = compare_rankings_with_kendalltau(test_ranking, predicted_ranking)
            logger.info(f'Kendall`s Tau: {tau}, P-value: {p_value}')

    ######################################################################
    # TESTING and INFERENCE of MODEL
    test_error, test_acc = evaluate(model, test_loader_cached, device, criterion, config['mode'])
    logger.info(f'Test Error: {test_error:.4f}, Test Acc: {test_acc:.4f}')
    training_end_time = datetime.now()
    logger.info(f'Training took {training_end_time - training_start_time}')

    # test model with ranking prediction
    logger.info(f'Starting Prediction of ranking')
    if config['mode'] == 'default' or config['mode'] == 'rank_mask':
        predicted_pref, predicted_util = predict(model, test_loader_cached, device)
        raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
        cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
        predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs,
                                                                     max_range=len(test_ranking))
    else:
        predicted_pref, _ = predict(model, test_loader_cached, device, config['mode'])
        raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
        cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
        predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs,
                                                                     max_range=len(test_ranking))


    if config['mode'] == 'default':
        logger.info(f'raw prefs: {predicted_pref}')
        logger.info(f'raw utils: {predicted_util}')

    predicted_ranking = rank_data(predicted_util)
    logger.info(f'length of rankings is the same: {len(predicted_ranking) == len(test_ranking)}')
    tau, p_value = compare_rankings_with_kendalltau(test_ranking, predicted_ranking)
    logger.info(f'Kendall`s Tau: {tau}, P-value: {p_value}')

    logger.info(f'Overall experiment took: {datetime.now() - start_time}')
    torch.save(model.state_dict(), config['folder_path']+'/model.pt')

