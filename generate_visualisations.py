import torch
import numpy as np

from typing import Union

from virtual_nodes.visualisation import visualise_neighbourhood_label_dist, visualise_graph, plot_smooth_curve, plot_smooth_curves, plot_heatmap
from Heterophily_and_oversmoothing.process import full_load_data
from virtual_nodes.augment import _binarize_tensor


def are_there_only_2_unqiue_values_in_dataset(dataset_str, features):
    """
    Returns True if in every row in `features`, there are only 2 unique values.
    """
    for row_idx in range(features.shape[0]):
        row = features[row_idx]
        unique_vals_row, unique_vals_idxs, unique_vals_counts = torch.unique(row, return_inverse=True, return_counts=True)
        # unique_vals.append(unique_vals_row)
        if unique_vals_row.shape[0] != 2 and not torch.all(unique_vals_row == 0):
            print(f'There are not only 2 unique values in {dataset_str}.')
            return False
    
    print(f'There are only 2 unique values in every row of {dataset_str}.')
    return True


def are_the_features_binary_in_dataset(dataset_str, features):
    '''
    Returns True if in every row of `features`, there are only 2 unique values
    and they are exactly 0 or 1.
    '''
    for row_idx in range(features.shape[0]):
        row = features[row_idx]
        unique_vals_row, val_idxs, val_counts = torch.unique(row, return_inverse=True, return_counts=True)

        # If the there are more unique values than just 0 and 1 then return False.
        if not torch.all(torch.logical_or(unique_vals_row == 0, unique_vals_row == 1)):
            print(f'There are not only 0s and 1s in {dataset_str}.')
            return False


def how_many_classes_are_there_in_dataset(dataset_str, labels):
    '''
    Returns the number of unique values in `labels`.
    '''
    unique_vals_labels, _, _ = torch.unique(labels, return_inverse=True, return_counts=True)
    print(f'There are {unique_vals_labels.shape[0]} classes in {dataset_str}.')
    return unique_vals_labels.shape[0]


def count_most_common_feature(g: torch.tensor,
                              features: torch.tensor,
                              labels: torch.tensor,
                              target_label: int):
    '''
    For a feature matrix features and a label vector labels, for a target label
    target_label, count the number of times the feature value is the most common
    among the features of nodes with label target_label and return the count.

    NOTE: This method assumes that each row of features has only 2 unique values.

    For given features and labels and a given target label, compute the number of times
    the feature value is the the most common among the features of neighbours of nodes
    with label target_label, and return the count.
    '''
    # labels for each node
    label_mask = labels == target_label
    # list of node ids with label
    label_indices = torch.argwhere(label_mask)
    label_indices = label_indices.squeeze(dim=-1)
    # go through each of the nodes in label_indices neighbours and add their features to a tensor
    neigh_features = torch.tensor([])
    for node_id in label_indices:
        neigh_mask = g[node_id, :]
        neigh_indices = torch.argwhere(neigh_mask)
        neigh_indices = neigh_indices.squeeze(dim=-1)
        neigh_features = torch.cat(
            (neigh_features, features[neigh_indices, :]), dim=0)
    
    # # take standard devation over tensor (dim=1)
    # label_feat_std[l] = torch.std(neigh_features, dim=0)

    # Binarize the features
    neigh_features = torch.where(neigh_features > 0, torch.ones_like(neigh_features), torch.zeros_like(neigh_features))

    # rows_with_target_label = torch.where(labels == target_label)[0]
    # rows_with_target_label_fetures = features[rows_with_target_label]
    
    # For each element in the row, find out which value is the most common
    # across all rows and count their occurrences.
    most_common_val_counts = torch.zeros(neigh_features.shape[1])
    for col_idx in range(neigh_features.shape[1]):
        col = neigh_features[:, col_idx]
        unique_vals_col, val_idxs, val_counts = torch.unique(col, return_inverse=True, return_counts=True)

        # At this point, the unique values should always be 0 and 1
        assert torch.all(torch.logical_or(unique_vals_col == 0, unique_vals_col == 1)), 'There are not only 0s and 1s in the features.'

        # Find the most common value in the column
        most_common_val = unique_vals_col[torch.argmax(val_counts)]
        most_common_val_counts[col_idx] = torch.sum(col == most_common_val)

    return most_common_val_counts, neigh_features


def label_feature_similarity(g,
                             features: torch.tensor,
                             labels: torch.tensor, 
                             target_label: int, 
                             return_mean: bool = False):
    
    most_common_val_counts, neigh_features = count_most_common_feature(g, features, labels, target_label)
    
    # Divide most_common_vals by the number of rows 
    rows_of_target_label = neigh_features
    most_common_val_counts = most_common_val_counts / rows_of_target_label.shape[0]

    # All of the elements in most_common_val_counts should be between 0 or 1.
    assert torch.all(torch.logical_and(most_common_val_counts >= 0, most_common_val_counts <= 1)), 'The most common value counts are not between 0 and 1.'

    # The last dimension of most_common_val_counts should be the number of features.
    assert most_common_val_counts.shape[-1] == features.shape[1], 'The number of features is not the same as the number of elements in most_common_val_counts.'

    if return_mean:
        return most_common_val_counts, torch.mean(most_common_val_counts)

    return most_common_val_counts


def label_mean_features(g, features: torch.tensor, labels: torch.tensor, label: Union[int, torch.tensor]):
    '''
    Returns the mean of the features for a given label.
    '''
    if isinstance(label, int):
        label = torch.tensor([label])
    
    # labels for each node
    label_mask = labels == label
    # list of node ids with label
    label_indices = torch.argwhere(label_mask)
    label_indices = label_indices.squeeze(dim=-1)
    # go through each of the nodes in label_indices neighbours and add their features to a tensor
    neigh_features = torch.tensor([])
    for node_id in label_indices:
        neigh_mask = g[node_id, :]
        neigh_indices = torch.argwhere(neigh_mask)
        neigh_indices = neigh_indices.squeeze(dim=-1)
        neigh_features = torch.cat(
            (neigh_features, features[neigh_indices, :]), dim=0)

    mean = torch.mean(neigh_features, dim=0)

    # The resultant tensor should havethe same number of columns as the features.
    assert mean.shape[-1] == features.shape[1], 'The number of features is not the same as the number of elements in the mean.'

    return mean


def compute_cosine_similarity_matrix(g, features: torch.tensor, labels: torch.tensor):
    '''
    Returns a torch.tensor matrix of dimensions (num_labels, num_labels) where
    the element at (i, j) is the cosine similarity between the mean of the features
    for label i and the mean of the features for label j.
    '''
    features = torch.nan_to_num(features)

    num_labels = torch.unique(labels).max().item() + 1
    cosine_similarity_matrix = torch.zeros((num_labels, num_labels))
    mean_features = torch.zeros((num_labels, features.shape[1]))

    for i in range(num_labels):
        mean_features[i] = label_mean_features(g, features, labels, i)
    
    for i in range(num_labels):
        for j in range(num_labels):
            similarity = torch.nn.functional.cosine_similarity(mean_features[i], mean_features[j], dim=0)
            cosine_similarity_matrix[i, j] = similarity
    
    return cosine_similarity_matrix


def dist_of_number_of_nodes_with_unexpected_feats(g,
                                                  features: torch.tensor,
                                                  labels: torch.tensor,
                                                  target_label: int):
    
    most_common_feat_counts, neigh_features = count_most_common_feature(g, features, labels, target_label)
    # Get the features of nodes with label target_label
    # rows_with_target_label = torch.where(labels == target_label)[0]
    rows_of_target_label = neigh_features
    most_common_feat_counts = most_common_feat_counts / rows_of_target_label.shape[0]

    proportion_of_nodes_with_unexpected_feat = 1 - most_common_feat_counts
    return torch.mean(proportion_of_nodes_with_unexpected_feat), torch.std(proportion_of_nodes_with_unexpected_feat)


if __name__ == "__main__":
    assert False

    datasets = ['wisconsin' , 'texas', 'cora', 'citeseer', 'chameleon', 'cornell', 'texas', 'squirrel', 'film'] # 'pubmed', 
    binary_datasets = []
    
    augment = True
    combined_test = True
    directed = False
    clip = True
    augstr = 'aug' if augment else 'noaug'

    ps = [1.0]
    # ps = [i * 0.05 for i in range(0, 10)]
    # ps.extend([i * 0.1 for i in range(6, 15)])

    best_p_mean_improvement = 0.0
    best_p = 0.0

    # Test that whether datasets are originally binary or not.
    for p in ps:
        aug_means = []
        aug_stds = []
        aug_sums = []
        noaug_means = []
        noaug_stds = []
        noaug_sums = []
        for dataset in datasets:
            if dataset == 'pubmed':
                continue
            # if dataset != 'cora':
            #     continue

            splitstr = 'splits/'+dataset+'_split_0.6_0.2_'+str(0)+'.npz'

            if combined_test:
                    # Load and view reults data with augmentation
                    g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj, num_vnodes, _ = full_load_data(dataset,
                                                                                                                                splitstr,
                                                                                                                                directed=directed,
                                                                                                                                clip=clip,
                                                                                                                                augment=augment,
                                                                                                                                p=p,
                                                                                                                                include_vnode_labels=True)
                    g = _binarize_tensor(g)
                    # TODO Add for loop
                    line, mean = label_feature_similarity(g, features, labels, 0, return_mean=True)
                    print(f'Augmented mean: {mean}, augmented std: {torch.std(line)}, augmented mode: {torch.mode(line)}, augmented median: {torch.median(line)}, augmented sum: {torch.sum(line)}')
                    aug_means.append(mean.cpu().detach().numpy())
                    aug_stds.append(torch.std(line).cpu().detach().numpy())
                    aug_sums.append(torch.sum(line).cpu().detach().numpy())
                    # Plot the similarity between the average features of each class.
                    similarity_matrix = compute_cosine_similarity_matrix(g, features, labels)
                    plot_heatmap(similarity_matrix,
                                xlabel="Label", 
                                ylabel= "Label", 
                                filename=f'plots/{dataset}-aug-label-feature-similarity.png')

                    plot_smooth_curve(line, 'Dataset: '+dataset+' '+augstr+' p='+str(p))

                    # Load and view reults without augmentation
                    g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj, num_vnodes, _ = full_load_data(dataset,
                                                                                                                                    splitstr,
                                                                                                                                    clip=False)

                    g = _binarize_tensor(g)
                    # TODO Add for loop
                    line, mean = label_feature_similarity(g, features, labels, 0, return_mean=True)
                    print(f'Unaugmented mean: {mean}, unaugmented std: {torch.std(line)}, unaugmented mode: {torch.mode(line)}, unaugmented median: {torch.median(line)}, unaugmented sum: {torch.sum(line)}')
                    noaug_means.append(mean.cpu().detach().numpy())
                    noaug_stds.append(torch.std(line).cpu().detach().numpy())
                    noaug_sums.append(torch.sum(line).cpu().detach().numpy())
                    plot_smooth_curve(line, 'Dataset: '+dataset+' '+'Unaugmented'+' p='+str(p))
                    # Plot the similarity between the average features of each class.
                    similarity_matrix = compute_cosine_similarity_matrix(g, features, labels)
                    plot_heatmap(similarity_matrix,
                                xlabel="Label", 
                                ylabel= "Label", 
                                filename=f'plots/{dataset}-unaug-label-feature-similarity.png')

                    if aug_means[-1] - noaug_means[-1] > best_p_mean_improvement:
                        best_p_mean_improvement = aug_means[-1] - noaug_means[-1]
                        best_p = p

            elif augment:
                for p in ps:
                    g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj, num_vnodes, _ = full_load_data(dataset,
                                                                                                                                splitstr,
                                                                                                                                directed=directed,
                                                                                                                                clip=clip,
                                                                                                                                augment=augment,
                                                                                                                                p=p,
                                                                                                                                include_vnode_labels=True)
                    
                    means, stds = torch.zeros((num_labels, len(ps))), torch.zeros((num_labels, len(ps)))

                    for target_label in range(num_labels):
                        mean, std = dist_of_number_of_nodes_with_unexpected_feats(g, features, labels, target_label)
                        means[target_label, ps.index(p)] = mean
                        stds[target_label, ps.index(p)] = std

                directed_str = 'directed' if directed else 'undirected'
                clip_str = 'clip' if clip else 'noclip'

                plot_smooth_curves(means,
                                torch.tensor(ps),
                                'Number of VNodes Added to the Graph, Proportional to its Masked Size',
                                'Mean % of Nodes with Unexpected Features',
                                filename=f'plots/{dataset}-{augstr}-{directed_str}-{clip_str}-mean-unexpected-percentage.png')
                
                plot_smooth_curves(stds,
                                torch.tensor(ps),
                                'Number of VNodes Added to the Graph, Proportional to its Masked Size',
                                'Std % of Nodes with Unexpected Features',
                                filename=f'plots/{dataset}-{augstr}-{directed_str}-{clip_str}-std-unexpected-percentage.png')

            else:
                # Load and view results without augmentation
                g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels, deg_vec, raw_adj, num_vnodes, _ = full_load_data(dataset,
                                                                                                                                splitstr,
                                                                                                                                clip=False)

                line, mean = label_feature_similarity(g, features, labels, 0, return_mean=True)
                print(mean)
                plot_smooth_curve(line)

        # # Plot the similarity between the average features of each class.
        # similarity_matrix = compute_cosine_similarity_matrix(g, features, labels)
        # plot_heatmap(similarity_matrix,
        #              xlabel="Label", 
        #              ylabel= "Label", 
        #              filename=f'plots/{dataset}-{augstr}-label-feature-similarity.png')
        
        # Plot the distribution of the number of nodes that do not share the same features.
        # mean, std = dist_of_number_of_nodes_with_unexpected_feats(features, labels, 0)

        np.savetxt(f'average_neighbourhood_consistency_{p}.csv',
                   np.column_stack((np.array(datasets, dtype='<U32'), np.array(aug_means, dtype=str), 
                                    np.array(aug_stds, dtype=str), np.array(aug_sums, dtype=str), np.array(noaug_means, dtype=str),
                                    np.array(noaug_stds, dtype=str), np.array(noaug_sums, dtype=str))),
                   delimiter=',',
                   fmt = '%.32s')
        print(f'Best p so far: {best_p}, best mean improvement: {best_p_mean_improvement}')

    print(f'BEST OVERALL p: {best_p}, best mean improvement: {best_p_mean_improvement}')

    """
    NOTE: If our method works, then ideally the similarity between classes is going to decrease. Also,
    the mean label feature similarity should be as close to 1 as possible (i.e. classes are becoming)
    more discernible on average. Ideally the standard deviation of the number of nodes that do not
    share the same features should decrease.
    """