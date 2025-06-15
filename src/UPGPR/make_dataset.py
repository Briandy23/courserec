from __future__ import absolute_import, division, print_function

import os
import argparse
import random
import wandb
import pickle
import random
from sklearn.model_selection import train_test_split
import json

from utils import *
from data_utils import Dataset
from knowledge_graph import KnowledgeGraph
from easydict import EasyDict as edict


def generate_labels(data_dir:str, filename:str)-> dict:
    enrolment_file = f"{data_dir}/{filename}"
    user_courses = {}  # {uid: [cid,...], ...}
    with open(enrolment_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            arr = line.split(" ")
            user_idx = int(arr[0])
            course_idx = int(arr[1])
            if user_idx not in user_courses:
                user_courses[user_idx] = []
            user_courses[user_idx].append(course_idx)
    return user_courses


def split_train_test_data_by_user_by_time(
    data_dir, data_file="enrolments.txt"
):
    learner_courses = generate_labels(data_dir, data_file)
    train_data = []
    test_data = []
    validation_data = []

    for learner in learner_courses:
        # Mỗi learner sẽ có danh sách các khóa học dạng (course_id, timestamp)
        courses = learner_courses[learner]

        # Sắp xếp theo thời gian tăng dần (giả sử tuple (course_id, timestamp))


        # Chia tập
        l_train_data = courses[:-2]  # Tất cả trừ 2 khóa học cuối
        l_validation_data = [courses[-2]]  # Khóa học gần nhất
        l_test_data = [courses[-1]]  # Khóa học cuối cùng

        # Lưu vào danh sách kết quả
        for c in l_train_data:
            train_data.append(f"{learner} {c[0]}\n")  # c[0] là course_id
        for c in l_validation_data:
            validation_data.append(f"{learner} {c[0]}\n")
        for c in l_test_data:
            test_data.append(f"{learner} {c[0]}\n")


    create_data_file(data_dir, train_data, "train.txt")
    create_data_file(data_dir, validation_data, "validation.txt")
    create_data_file(data_dir, test_data, "test.txt")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/UPGPR/mooc.json", help="Config file."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    args = config.PREPROCESS

    set_random_seed(args.seed)
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config=config.PREPROCESS,
        )

    # Create train and test data
    # split_train_test_data(args.dataset, ratio=args.ratio, ratio_validation=args.ratio_validation, data_file=args.data_file)

    print(args)
    split_train_test_data_by_user_by_time(
        args.data_dir,
        data_file=args.data_file,
    )
    # Create MoocDataset instance for dataset.
    # ========== BEGIN ========== #
    print(f"Loading dataset from folder: {args.data_dir}")
    if not os.path.isdir(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    dataset = Dataset(args.data_dir, config.KG_ARGS)
    save_dataset(args.tmp_dir, dataset, args.use_wandb)

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print("Creating knowledge graph from dataset...")
    # dataset = load_dataset(args.tmp_dir)
    kg = KnowledgeGraph(
        dataset,
        config.KG_ARGS,
        use_user_relations=args.use_user_relations,
        use_entity_relations=args.use_entity_relations,
    )
    kg.compute_degrees()
    save_kg(args.tmp_dir, kg, args.use_wandb)
    # =========== END =========== #

    # Genereate train/test labels.
    # ========== BEGIN ========== #
    print("Generate train/test labels.")
    train_labels = generate_labels(args.data_dir, "train.txt")
    test_labels = generate_labels(args.data_dir, "test.txt")
    validation_labels = generate_labels(args.data_dir, "validation.txt")

    save_labels(args.tmp_dir, train_labels, mode="train", use_wandb=args.use_wandb)
    save_labels(args.tmp_dir, test_labels, mode="test", use_wandb=args.use_wandb)
    save_labels(
        args.tmp_dir, validation_labels, mode="validation", use_wandb=args.use_wandb
    )

    # =========== END =========== #
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
