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

from misc import setup_experiment, setup_logger, config_add_nagsl
from data.load import get_data
from data.loader import CustomDataLoader
from models.torch_gnn import RankGNN, RankGAN, PairRankGNN, PairRankGNN2, RANet
from models.NAGSL.NAGSL import NAGSLNet
from data.misc import compare_rankings_with_kendalltau, rank_data, train, evaluate, predict, preprocess_predictions, retrieve_preference_counts_from_predictions

if __name__ == '__main__':
    start_time = datetime.now()
    ######################################################################
    # CONFIG
    config = setup_experiment(sys.argv[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    logger = setup_logger(config['folder_path'], lvl=config['logger']['level'])
    copyfile('src/models/torch_gnn.py', config['folder_path']+'/torch_gnn.py')
    logger.info(f'Starting at {start_time}')
    # SETUP
    # np.random.seed(config['seed'])
    # torch.manual_seed(config['seed'])
    seed_everything(config['seed'])
    ######################################################################
    # Load + prep data
    if config['mode'] == 'nagsl_attention':
        config = config_add_nagsl(config)
    if config['mode'] == 'default' or config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention' or config['mode'] == 'my_attention':
        train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking = get_data(config)
    elif config['mode'] == 'fc_weight' or config['mode'] == 'fc_extra':
        if os.path.isfile(f"data/{config['data_name']}.pkl"):
            with open(f"data/{config['data_name']}.pkl", 'rb') as f:
                train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking, config['num_node_features'], config['max_num_nodes'] = pickle.load(f)
        else:
            train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking = get_data(config)
            with open(config['folder_path']+f"/{config['data_name']}.pkl", 'wb') as f:
                 pickle.dump((train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking, config['num_node_features'], config['max_num_nodes']), f)
    else:
        raise ValueError(f'Unknown mode {config["mode"]}')
    logger.info(f'Config: {config}')

    data_prep_end_time = datetime.now()
    logger.info(f'Data prep took {data_prep_end_time - start_time}')

    # Create data loaders
    if config['mode'] == 'default':
        train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'], shuffle=True, mode=config['mode'], config=config)
        valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'], shuffle=False, mode=config['mode'], config=config)
        test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_dataset), shuffle=False, mode=config['mode'], config=config)
    elif config['mode'] == 'gat_attention' or config['mode'] == 'nagsl_attention' or config['mode'] == 'my_attention':
        train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'], shuffle=True, mode=config['mode'], config=config)
        valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'], shuffle=False, mode=config['mode'], config=config)
        test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_prefs), shuffle=False, mode=config['mode'], config=config)
    else:# fc_weight, fc_extra or my_attention
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Create model, optimizer, and loss function
    if config['mode'] == 'default':
        model = RankGNN(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'gat_attention':
        model = RankGAN(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'fc_weight':
        model = PairRankGNN(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'fc_extra':
        model = PairRankGNN2(num_node_features=config['num_node_features'], device=device, config=config)
    elif config['mode'] == 'nagsl_attention':
        model = NAGSLNet(config)
    elif config['mode'] == 'my_attention':
        model = RANet(config=config)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.BCELoss()


    # Train and evaluate model
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

    for epoch in range(config['epochs']):
        shuffle(train_loader_cached)
        shuffle(valid_loader_cached)
        train(model, train_loader_cached, device, optimizer, criterion, config['mode'])
        torch.cuda.empty_cache()
        train_error, test_acc = evaluate(model, train_loader_cached, device, criterion, config['mode'])
        torch.cuda.empty_cache()
        valid_error, valid_acc = evaluate(model, valid_loader_cached, device, criterion, config['mode'])
        torch.cuda.empty_cache()
        logger.info(f'Epoch: {epoch}, Train Error: {train_error:.4f}, Valid Error: {valid_error:.4f}, Train Acc: {test_acc:.4f}, Valid Acc: {valid_acc:.4f}')
        if epoch % config['logging_inverval'] == 0:
            torch.save(model.state_dict(), config['folder_path'] + f'/epoch{epoch}_model.pt')
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'losslogger': criterion}
            torch.save(state, config['folder_path'] + f'/epoch{epoch}_state.pt')
            if config['mode'] == 'default':
                predicted_pref, predicted_util = predict(model, test_loader_cached, device)
                raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
                cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
                predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs, max_range=len(test_ranking))
            else:
                predicted_pref, _ = predict(model, test_loader_cached, device, config['mode'])
                raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
                cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
                predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs, max_range=len(test_ranking))

            predicted_ranking = rank_data(predicted_util)
            logger.info(f'length of rankings is the same: {len(predicted_ranking) == len(test_ranking)}')
            tau, p_value = compare_rankings_with_kendalltau(test_ranking, predicted_ranking)
            logger.info(f'Kendall`s Tau: {tau}, P-value: {p_value}')

    test_error, test_acc = evaluate(model, test_loader_cached, device, criterion, config['mode'])
    logger.info(f'Test Error: {test_error:.4f}, Test Acc: {test_acc:.4f}')
    training_end_time = datetime.now()
    logger.info(f'Training took {training_end_time - training_start_time}')

    # test model with ranking prediction
    logger.info(f'Starting Prediction of ranking')
    if config['mode'] == 'default':
        predicted_pref, predicted_util = predict(model, test_loader_cached, device)
        raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], predicted_pref))
        cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
        predicted_util = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs, max_range=len(test_ranking))
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

