from __future__ import absolute_import, division, print_function
import numpy as np
import os
import json
from math import log
import wandb
from collections import Counter
import argparse
from math import log
from tqdm import tqdm
from easydict import EasyDict as edict
import torch
from functools import reduce
from kg_env import BatchKGEnvironment
from actor_critic import ActorCritic
from utils import *

def get_all_items(train, test, validation):
    all_items = set()
    for dataset in [train, test, validation]:
        for items in dataset.values():
            all_items.update(items)
    return list(all_items)


def evaluate(
    topk_matches,
    test_user_products,
    use_wandb,
    tmp_dir,
    result_file_dir,
    result_file_name,
    min_courses=10,
    compute_all=True,
    Ks=[5, 10],
):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
        Ks: list of K values to evaluate (e.g., [5, 10]).
    """
    invalid_users = []
    metrics = {k: {'precisions': [], 'recalls': [], 'ndcgs': [], 'hits': [], 
                   'hits_at_1': [], 'hits_at_3': [], 'maps': [], 'f1_scores': []} 
               for k in Ks}
    metrics_all = {k: {'precisions': [], 'recalls': [], 'ndcgs': [], 'hits': [], 
                       'hits_at_1': [], 'hits_at_3': [], 'maps': [], 'f1_scores': []} 
                   for k in Ks}
    
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        is_invalid = False
        if uid not in topk_matches or len(topk_matches[uid]) < min_courses:
            invalid_users.append(uid)
            is_invalid = True
        pred_list, rel_set = topk_matches.get(uid, [])[::-1], test_user_products[uid]
        
        if len(pred_list) == 0:
            for k in Ks:
                metrics_all[k]['ndcgs'].append(0.0)
                metrics_all[k]['recalls'].append(0.0)
                metrics_all[k]['precisions'].append(0.0)
                metrics_all[k]['hits'].append(0.0)
                metrics_all[k]['hits_at_1'].append(0.0)
                metrics_all[k]['hits_at_3'].append(0.0)
                metrics_all[k]['maps'].append(0.0)
                metrics_all[k]['f1_scores'].append(0.0)
            continue

        for k in Ks:
            dcg = 0.0
            hit_num = 0.0
            hit_at_1 = 0.0
            hit_at_3 = 0.0
            ap = 0.0  # For MAP
            relevant_at_k = 0  # For MAP calculation
            
            # Calculate metrics for top-k predictions
            for i in range(min(k, len(pred_list))):
                if pred_list[i] in rel_set:
                    dcg += 1.0 / (log(i + 2) / log(2))
                    hit_num += 1
                    relevant_at_k += 1
                    ap += relevant_at_k / (i + 1)  # Precision at position i+1
                    if i < 1:
                        hit_at_1 += 1
                    if i < 3:
                        hit_at_3 += 1
            
            # IDCG for NDCG
            idcg = 0.0
            for i in range(min(len(rel_set), k)):
                idcg += 1.0 / (log(i + 2) / log(2))
            
            # Calculate metrics
            ndcg = dcg / idcg if idcg > 0 else 0.0
            recall = hit_num / len(rel_set) if len(rel_set) > 0 else 0.0
            precision = hit_num / k if k > 0 else 0.0
            hit = 1.0 if hit_num > 0.0 else 0.0
            hit_at_1 = 1.0 if hit_at_1 > 0.0 else 0.0
            hit_at_3 = 1.0 if hit_at_3 > 0.0 else 0.0
            map_score = ap / min(len(rel_set), k) if len(rel_set) > 0 else 0.0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if not is_invalid:
                metrics[k]['ndcgs'].append(ndcg)
                metrics[k]['recalls'].append(recall)
                metrics[k]['precisions'].append(precision)
                metrics[k]['hits'].append(hit)
                metrics[k]['hits_at_1'].append(hit_at_1)
                metrics[k]['hits_at_3'].append(hit_at_3)
                metrics[k]['maps'].append(map_score)
                metrics[k]['f1_scores'].append(f1_score)

            if compute_all or not is_invalid:
                metrics_all[k]['ndcgs'].append(ndcg)
                metrics_all[k]['recalls'].append(recall)
                metrics_all[k]['precisions'].append(precision)
                metrics_all[k]['hits'].append(hit)
                metrics_all[k]['hits_at_1'].append(hit_at_1)
                metrics_all[k]['hits_at_3'].append(hit_at_3)
                metrics_all[k]['maps'].append(map_score)
                metrics_all[k]['f1_scores'].append(f1_score)
            else:
                metrics_all[k]['ndcgs'].append(0.0)
                metrics_all[k]['recalls'].append(0.0)
                metrics_all[k]['precisions'].append(0.0)
                metrics_all[k]['hits'].append(0.0)
                metrics_all[k]['hits_at_1'].append(0.0)
                metrics_all[k]['hits_at_3'].append(0.0)
                metrics_all[k]['maps'].append(0.0)
                metrics_all[k]['f1_scores'].append(0.0)

    # Compute average metrics
    avg_metrics = {}
    avg_metrics_all = {}
    for k in Ks:
        avg_metrics[k] = {
            'precision': np.mean(metrics[k]['precisions']) * 100,
            'recall': np.mean(metrics[k]['recalls']) * 100,
            'ndcg': np.mean(metrics[k]['ndcgs']) * 100,
            'hit': np.mean(metrics[k]['hits']) * 100,
            'hit_at_1': np.mean(metrics[k]['hits_at_1']) * 100,
            'hit_at_3': np.mean(metrics[k]['hits_at_3']) * 100,
            'map': np.mean(metrics[k]['maps']) * 100,
            'f1_score': np.mean(metrics[k]['f1_scores']) * 100,
        }
        avg_metrics_all[k] = {
            'precision': np.mean(metrics_all[k]['precisions']) * 100,
            'recall': np.mean(metrics_all[k]['recalls']) * 100,
            'ndcg': np.mean(metrics_all[k]['ndcgs']) * 100,
            'hit': np.mean(metrics_all[k]['hits']) * 100,
            'hit_at_1': np.mean(metrics_all[k]['hits_at_1']) * 100,
            'hit_at_3': np.mean(metrics_all[k]['hits_at_3']) * 100,
            'map': np.mean(metrics_all[k]['maps']) * 100,
            'f1_score': np.mean(metrics_all[k]['f1_scores']) * 100,
        }

    # Print results
    print(f"Min courses to consider user valid={min_courses} | Compute metrics for all users={compute_all}\n")
    for k in Ks:
        print(f"Metrics @ K={k} (Valid users):")
        print(
            f"NDCG={avg_metrics[k]['ndcg']:.3f} | Recall={avg_metrics[k]['recall']:.3f} | "
            f"HR={avg_metrics[k]['hit']:.3f} | Precision={avg_metrics[k]['precision']:.3f} | "
            f"HR@1={avg_metrics[k]['hit_at_1']:.3f} | HR@3={avg_metrics[k]['hit_at_3']:.3f} | "
            f"MAP={avg_metrics[k]['map']:.3f} | F1={avg_metrics[k]['f1_score']:.3f} | "
            f"Invalid users={len(invalid_users)}\n"
        )
        print(f"Metrics @ K={k} (All users):")
        print(
            f"NDCG={avg_metrics_all[k]['ndcg']:.3f} | Recall={avg_metrics_all[k]['recall']:.3f} | "
            f"HR={avg_metrics_all[k]['hit']:.3f} | Precision={avg_metrics_all[k]['precision']:.3f} | "
            f"HR@1={avg_metrics_all[k]['hit_at_1']:.3f} | HR@3={avg_metrics_all[k]['hit_at_3']:.3f} | "
            f"MAP={avg_metrics_all[k]['map']:.3f} | F1={avg_metrics_all[k]['f1_score']:.3f}\n"
        )

    # Save results
    filename = os.path.join(tmp_dir, result_file_dir, result_file_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_to_save = {
        f"K={k}": {
            'ndcg': avg_metrics_all[k]['ndcg'],
            'recall': avg_metrics_all[k]['recall'],
            'hit': avg_metrics_all[k]['hit'],
            'precision': avg_metrics_all[k]['precision'],
            'map': avg_metrics_all[k]['map'],
            'f1_score': avg_metrics_all[k]['f1_score'],
        } for k in Ks
    }
    json.dump(metrics_to_save, open(filename, 'w'))

    if use_wandb:
        wandb.save(filename)

    return {k: (avg_metrics[k]['precision'], avg_metrics[k]['recall'], 
                avg_metrics[k]['ndcg'], avg_metrics[k]['hit'], 
                avg_metrics[k]['map'], avg_metrics[k]['f1_score']) for k in Ks}

def evaluate_neg_sampling(
    topk_matches,
    test_user_products,
    train_user_products,
    validation_user_products,
    use_wandb,
    tmp_dir,
    result_file_dir,
    result_file_name,
    num_neg=100,
    Ks=[5, 10],
):
    """Evaluate using negative sampling with 100 negatives and 1 positive.
    Args:
        topk_matches: dict of predicted product ids.
        test_user_products: dict of actual user products.
        all_items: list of all possible item IDs.
        num_neg: number of negative samples (default: 100).
        Ks: list of K values to evaluate.
    """
    all_items=get_all_items(
        train_user_products, 
        test_user_products, 
        validation_user_products)
    invalid_users = []
    metrics = {k: {'precisions': [], 'recalls': [], 'ndcgs': [], 'hits': [], 
                   'hits_at_1': [], 'hits_at_3': [], 'maps': [], 'f1_scores': []} 
               for k in Ks}
    
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            for k in Ks:
                metrics[k]['ndcgs'].append(0.0)
                metrics[k]['recalls'].append(0.0)
                metrics[k]['precisions'].append(0.0)
                metrics[k]['hits'].append(0.0)
                metrics[k]['hits_at_1'].append(0.0)
                metrics[k]['hits_at_3'].append(0.0)
                metrics[k]['maps'].append(0.0)
                metrics[k]['f1_scores'].append(0.0)
            continue
        
        # Get positive item
        rel_set = test_user_products[uid]
        if len(rel_set) == 0:
            continue
        positive_item = np.random.choice(list(rel_set))  # Randomly select one positive
        
        # Sample negative items
        negative_items = np.random.choice(
            [item for item in all_items if item not in rel_set], 
            size=num_neg, replace=False
        ).tolist()
        
        # Combine positive and negative items
        candidate_items = [positive_item] + negative_items
        pred_list = topk_matches.get(uid, [])[::-1]
        
        # Filter predictions to only include candidate items
        filtered_pred = [pid for pid in pred_list if pid in candidate_items]
        
        if len(filtered_pred) == 0:
            for k in Ks:
                metrics[k]['ndcgs'].append(0.0)
                metrics[k]['recalls'].append(0.0)
                metrics[k]['precisions'].append(0.0)
                metrics[k]['hits'].append(0.0)
                metrics[k]['hits_at_1'].append(0.0)
                metrics[k]['hits_at_3'].append(0.0)
                metrics[k]['maps'].append(0.0)
                metrics[k]['f1_scores'].append(0.0)
            continue

        for k in Ks:
            dcg = 0.0
            hit_num = 0.0
            hit_at_1 = 0.0
            hit_at_3 = 0.0
            ap = 0.0
            relevant_at_k = 0
            
            # Calculate metrics for top-k predictions
            for i in range(min(k, len(filtered_pred))):
                if filtered_pred[i] == positive_item:
                    dcg += 1.0 / (log(i + 2) / log(2))
                    hit_num += 1
                    relevant_at_k += 1
                    ap += relevant_at_k / (i + 1)
                    if i < 1:
                        hit_at_1 += 1
                    if i < 3:
                        hit_at_3 += 1
            
            # IDCG for NDCG (only one positive item)
            idcg = 1.0 / (log(2) / log(2))  # Since we have only 1 positive
            
            # Calculate metrics
            ndcg = dcg / idcg if idcg > 0 else 0.0
            recall = hit_num / 1.0  # Only one positive item
            precision = hit_num / k if k > 0 else 0.0
            hit = 1.0 if hit_num > 0.0 else 0.0
            hit_at_1 = 1.0 if hit_at_1 > 0.0 else 0.0
            hit_at_3 = 1.0 if hit_at_3 > 0.0 else 0.0
            map_score = ap / 1.0  # Only one positive item
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[k]['ndcgs'].append(ndcg)
            metrics[k]['recalls'].append(recall)
            metrics[k]['precisions'].append(precision)
            metrics[k]['hits'].append(hit)
            metrics[k]['hits_at_1'].append(hit_at_1)
            metrics[k]['hits_at_3'].append(hit_at_3)
            metrics[k]['maps'].append(map_score)
            metrics[k]['f1_scores'].append(f1_score)

    # Compute average metrics
    avg_metrics = {}
    for k in Ks:
        avg_metrics[k] = {
            'precision': np.mean(metrics[k]['precisions']) * 100,
            'recall': np.mean(metrics[k]['recalls']) * 100,
            'ndcg': np.mean(metrics[k]['ndcgs']) * 100,
            'hit': np.mean(metrics[k]['hits']) * 100,
            'hit_at_1': np.mean(metrics[k]['hits_at_1']) * 100,
            'hit_at_3': np.mean(metrics[k]['hits_at_3']) * 100,
            'map': np.mean(metrics[k]['maps']) * 100,
            'f1_score': np.mean(metrics[k]['f1_scores']) * 100,
        }

    # Print results
    print(f"Negative Sampling Evaluation (100 negatives) | Invalid users={len(invalid_users)}\n")
    for k in Ks:
        print(f"Metrics @ K={k}:")
        print(
            f"NDCG={avg_metrics[k]['ndcg']:.3f} | Recall={avg_metrics[k]['recall']:.3f} | "
            f"HR={avg_metrics[k]['hit']:.3f} | Precision={avg_metrics[k]['precision']:.3f} | "
            f"HR@1={avg_metrics[k]['hit_at_1']:.3f} | HR@3={avg_metrics[k]['hit_at_3']:.3f} | "
            f"MAP={avg_metrics[k]['map']:.3f} | F1={avg_metrics[k]['f1_score']:.3f}\n"
        )

    # Save results
    filename = os.path.join(tmp_dir, result_file_dir, f"neg_sampling_{result_file_name}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_to_save = {
        f"K={k}": {
            'ndcg': avg_metrics[k]['ndcg'],
            'recall': avg_metrics[k]['recall'],
            'hit': avg_metrics[k]['hit'],
            'precision': avg_metrics[k]['precision'],
            'map': avg_metrics[k]['map'],
            'f1_score': avg_metrics[k]['f1_score'],
        } for k in Ks
    }
    json.dump(metrics_to_save, open(filename, 'w'))

    if use_wandb:
        wandb.save(filename)

    return {k: (avg_metrics[k]['precision'], avg_metrics[k]['recall'], 
                avg_metrics[k]['ndcg'], avg_metrics[k]['hit'], 
                avg_metrics[k]['map'], avg_metrics[k]['f1_score']) for k in Ks}




def evaluate_validation(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    hits = []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 1:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                hit_num += 1

        hit = 1.0 if hit_num > 0.0 else 0.0
        hits.append(hit)

    avg_hit = np.mean(hits) * 100

    print(" HR={:.3f} | Invalid users={}\n".format(avg_hit, len(invalid_users)))

    return avg_hit


def batch_beam_search(env, model, kg_args, uids, device, topk=[10, 3, 1], policy=0):
    """
    Performs batch beam search over a knowledge graph environment using a given model.
    Args:
        env: The environment object that provides state, action, and transition functions.
        model: The neural network model used to predict action probabilities.
        kg_args: Arguments or configuration related to the knowledge graph, including relation mappings and self-loop identifier.
        uids (list): List of user IDs for which to perform the search.
        device: The device (CPU or GPU) on which to run the model computations.
        topk (list, optional): List specifying the beam width (number of top actions to keep) at each hop. Default is [10, 3, 1].
        policy (int, optional): Policy indicator (unused in this function, but may be used for future extensions). Default is 0.
    Returns:
        tuple:
            - path_pool (list): List of paths found for each user, where each path is a list of (relation, node_type, node_id) tuples.
            - probs_pool (list): List of probability sequences corresponding to each path in path_pool.
    """
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(env.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(len(topk)):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        batch_act_embeddings = env.batch_action_embeddings(
            path_pool, acts_pool
        )  # numpy array of size [bs, 2*embed_size, act_dim]
        embeddings = torch.ByteTensor(batch_act_embeddings).to(device)
        probs, _ = model(
            (state_tensor, actmask_tensor, embeddings)
        )  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(
            probs, topk[hop], dim=1
        )  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == kg_args.self_loop:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = kg_args.kg_relation[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < len(topk) - 1:  # no need to update state at the last hop
            state_pool = env._batch_get_state(path_pool)
    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args, kg_args, data="test"):
    print("Predicting paths...")
    env = BatchKGEnvironment(
        args.tmp_dir,
        kg_args,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
        reward_function=args.reward,
        use_pattern=args.use_pattern,
    )
    pretrain_sd = torch.load(policy_file, map_location=torch.device("cpu"))
    model = ActorCritic(
        env.state_dim,
        env.act_dim,
        gamma=args.gamma,
        hidden_sizes=args.hidden,
        modified_policy=args.modified_policy,
        embed_size=env.embed_size,
    ).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.tmp_dir, data)
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(
            env,
            model,
            kg_args,
            batch_uids,
            args.device,
            topk=args.topk,
            policy=args.modified_policy,
        )
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {"paths": all_paths, "probs": all_probs}
    pickle.dump(predicts, open(path_file, "wb"))
    if args.use_wandb:
        wandb.save(path_file)


def evaluate_paths(
    dir_path,
    path_file,
    train_labels,
    test_labels,
    validation_labels,
    kg_args,
    use_wandb,
    result_file_dir,
    result_file_name,
    validation=False,
    sum_prob=False,
):
    embeds = load_embed(dir_path)
    user_embeds = embeds["user"]
    enroll_embeds = embeds[kg_args.interaction][0]
    course_embeds = embeds["item"]
    scores = np.dot(user_embeds + enroll_embeds, course_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, "rb"))
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results["paths"], results["probs"]):
        if path[-1][1] != "item":
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    # 2) Compute the sum of probabilities for each user-course pair
    if sum_prob == True:
        user_course_probs = {}
        for uid in pred_paths:
            user_course_probs[uid] = Counter()
            for pid in pred_paths[uid]:
                prob_sum = sum(path[1] for path in pred_paths[uid][pid])
                user_course_probs[uid][pid] = prob_sum
                user_course_probs[uid].most_common(10)

        topk_matches = {}
        for uid, prob_dict in user_course_probs.items():
            topk_matches[uid] = [pid for pid, _ in prob_dict.most_common(10)]

        with open("json_data.json", "w") as f:
            json.dump(user_course_probs, f)

        if validation:
            return evaluate_validation(topk_matches, test_labels)

        else:
            for min_courses in [1, 10]:
                for compute_all in [True, False]:
                    evaluate(
                        topk_matches,
                        test_labels,
                        use_wandb,
                        args.tmp_dir,
                        result_file_dir=result_file_dir,
                        result_file_name=result_file_name,
                        min_courses=min_courses,
                        compute_all=compute_all,
                        sum_prob=sum_prob,
                    )

    # 3) Pick best path for each user-product pair, also remove pid if it is in train set.
    if sum_prob == False:
        best_pred_paths = {}
        for uid in pred_paths:
            train_pids = set(train_labels.get(uid, []))
            if len(train_pids) == 0:
                continue
            best_pred_paths[uid] = []
            for pid in pred_paths[uid]:
                if pid in train_pids:
                    continue
                # Get the path with highest probability
                sorted_path = sorted(
                    pred_paths[uid][pid], key=lambda x: x[1], reverse=True
                )
                best_pred_paths[uid].append(sorted_path[0])

        path_patterns = {}
        for uid in best_pred_paths:
            for path in best_pred_paths[uid]:
                path_pattern = path[2]
                pattern_key = ""
                for node in path_pattern:
                    pattern_key += node[0] + "_" + node[1] + "-->"
                path_patterns[pattern_key] = path_patterns.get(pattern_key, 0) + 1

        print(path_patterns)

        # 3) Compute top 10 recommended products for each user.
        sort_by = "score"
        pred_labels = {}
        pred_full_rankings = {}
        for uid in best_pred_paths:
            if sort_by == "score":
                sorted_path = sorted(
                    best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True
                )
            elif sort_by == "prob":
                sorted_path = sorted(
                    best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True
                )
            top10_pids = [
                p[-1][2] for _, _, p in sorted_path[:10]
            ]  # from largest to smallest

            all_pids = [
                p[-1][2] for _, _, p in sorted_path
            ]  

            pred_labels[uid] = top10_pids[
                ::-1
            ]  # change order to from smallest to largest!
            pred_full_rankings[uid] = all_pids[::-1]

        if validation == True:
            return evaluate_validation(pred_labels, test_labels)

        else:
            for min_courses in [10]:
                for compute_all in [True]:
                    evaluate(
                        pred_labels,
                        test_labels,
                        use_wandb,
                        args.tmp_dir,
                        result_file_dir=result_file_dir,
                        result_file_name=result_file_name,
                        min_courses=10,
                        compute_all=compute_all,
                        Ks=[5, 10],
                    )
                    evaluate_neg_sampling(
                        pred_full_rankings,
                        test_labels,
                        train_labels,
                        validation_labels,
                        use_wandb,
                        args.tmp_dir,
                        result_file_dir=result_file_dir,
                        result_file_name=result_file_name,
                        num_neg=100,
                        Ks=[5, 10],
                    )


def test(args, kg_args):
    policy_file = args.log_dir + "/tmp_policy_model_epoch_{}.ckpt".format(args.epochs)
    path_file = args.log_dir + "/policy_paths_epoch_{}.pkl".format(args.epochs)

    train_labels = load_labels(args.tmp_dir, "train")
    test_labels = load_labels(args.tmp_dir, "test")
    validation_labels = load_labels(args.tmp_dir, "validation")
     
    all_courses = get_all_items(train_labels, test_labels, validation_labels)

    if args.run_path:
        predict_paths(policy_file, path_file, args, kg_args)
    if args.run_eval:
        evaluate_paths(
            args.tmp_dir,
            path_file,
            train_labels,
            test_labels,
            validation_labels,
            kg_args,
            args.use_wandb,
            args.result_file_dir,
            args.result_file_name,
            validation=False,
            sum_prob=args.sum_prob,
        )


if __name__ == "__main__":
    boolean = lambda x: (str(x).lower() == "true")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/UPGPR/mooc.json", help="Config file."
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.TEST_AGENT

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name, name=args.wandb_run_name, config=args
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    if args.early_stopping == True:
        with open("early_stopping.txt", "r") as f:
            args.epochs = int(f.read())

    args.log_dir = args.tmp_dir + "/" + args.name
    test(args, config.KG_ARGS)

    if args.use_wandb:
        wandb.finish()
