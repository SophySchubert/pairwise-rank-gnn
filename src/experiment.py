import torch
from torch_geometric.loader import DataLoader
import numpy as np
from datetime import datetime
import sys
from shutil import copyfile
import pickle

from misc import setup_experiment, setup_logger
from data.load import get_data
from data.loader import CustomDataLoader
from models.torch_gnn import RGNN, PRGNN
from data.misc import compare_rankings_with_kendalltau, rank_data, train, evaluate, predict, preprocess_predictions, retrieve_preference_counts_from_predictions

if __name__ == '__main__':
    start_time = datetime.now()
    ######################################################################
    # CONFIG
    config = setup_experiment(sys.argv[1])
    logger = setup_logger(config['folder_path'], lvl=config['logger']['level'])
    copyfile('src/models/torch_gnn.py', config['folder_path']+'/torch_gnn.py')
    logger.info(f'Starting at {start_time}')
    # SETUP
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    ######################################################################
    # Load + prep data
    if config['mode'] == 'default':
        train_dataset, valid_dataset, test_dataset, train_prefs, valid_prefs, test_prefs, test_ranking = get_data(config)
    elif config['mode'] == 'fully-connected':
        with open('data/prepared_data.pkl', 'rb') as f:
            train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking = pickle.load(f)
        config['num_node_features'] = train_dataset[0].x.size(1) #should be 9 for ogbg-molesol
        # train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking = get_data(config)
        # with open(config['folder_path']+'/prepared_data.pkl', 'wb') as f:
        #      pickle.dump((train_dataset, valid_dataset, test_dataset, test_prefs, test_ranking), f)
        # print(f'test_prefs: {test_prefs}')
    else:
        raise ValueError(f'Unknown mode {config["mode"]}')
    data_prep_end_time = datetime.now()
    logger.info(f'Data prep took {data_prep_end_time - start_time}')

    # Create data loaders
    if config['mode'] == 'default':
        train_loader = CustomDataLoader(train_prefs,train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = CustomDataLoader(valid_prefs,valid_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = CustomDataLoader(test_prefs, test_dataset, batch_size=len(test_dataset), shuffle=False)
    elif config['mode'] == 'fully-connected':
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    # Create model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['mode'] == 'default':
        model = RGNN(num_node_features=config['num_node_features'], device=device)
    elif config['mode'] == 'fully-connected':
        model = PRGNN(num_node_features=config['num_node_features'], device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()

    # Train and evaluate model
    logger.info(f'Starting training loop')
    training_start_time = datetime.now()
    for epoch in range(config['epochs']):
        train(model, train_loader, device, optimizer, criterion, config['mode'])
        train_error = evaluate(model, train_loader, device, criterion, config['mode'])
        valid_error = evaluate(model, valid_loader, device, criterion, config['mode'])
        logger.info(f'Epoch: {epoch + 1}, Train Error: {train_error:.4f}, Valid Error: {valid_error:.4f}')
        if epoch % 50 == 0 and epoch != config['epochs']:
            torch.save(model.state_dict(), config['folder_path'] + f'/epoch{epoch}_model.pt')
            if config['mode'] == 'default':
                predicted_utils = predict(model, test_loader, device)
            elif config['mode'] == 'fully-connected':
                raw_predictions = predict(model, test_loader, device, config['mode'])
                # https://discuss.pytorch.org/t/softmax-outputing-0-or-1-instead-of-probabilities/101564
                # raw_predictions[0] = 1000.
                # raw_predictions = torch.nn.functional.softmax(raw_predictions, dim=0)
                raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], raw_predictions))
                cleaned_predictions = preprocess_predictions(raw_predictions_and_prefs)
                predicted_utils = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs, max_range=len(test_ranking))

            predicted_ranking = rank_data(predicted_utils)
            tau, p_value = compare_rankings_with_kendalltau(test_ranking, predicted_ranking)
            logger.info(f'Kendall`s Tau: {tau}, P-value: {p_value}')

    training_end_time = datetime.now()
    logger.info(f'Training took {training_end_time - training_start_time}')

    # test model with ranking prediction
    logger.info(f'Starting Prediction of ranking')
    if config['mode'] == 'default':
        predicted_utils = predict(model, test_loader, device)
    elif config['mode'] == 'fully-connected':
        raw_predictions = predict(model, test_loader, device, config['mode'])
        raw_predictions_and_prefs = np.column_stack((test_prefs[:, :2], raw_predictions))
        cleaned_predictions =  preprocess_predictions(raw_predictions_and_prefs)
        predicted_utils = retrieve_preference_counts_from_predictions(raw_predictions_and_prefs, max_range=len(test_ranking))

    predicted_ranking = rank_data(predicted_utils)
    tau, p_value = compare_rankings_with_kendalltau(test_ranking, predicted_ranking)
    logger.info(f'Kendall`s Tau: {tau}, P-value: {p_value}')
    # logger.info(f'test ranking: {test_ranking}')
    # logger.info(f'test utils: {[g.y.item() for g in test_dataset]}')
    # logger.info(f'Predicted ranking: {predicted_ranking}')
    # logger.info(f'Predicted utils: {predicted_utils}')

    logger.info(f'Overall experiment took: {datetime.now() - start_time}')
    torch.save(model.state_dict(), config['folder_path']+'/model.pt')

