import os
import argparse
import pickle

import pandas as pd


def read_relations(dataset:pd.DataFrame, column_names):
    if column_names[0] == "user":
        df = dataset[[column_names[0]+"_id", column_names[1]+"_id"]].copy()
        df.columns = column_names
        return df
    elif column_names[0] == "course":
        new_column_names = []
        course_df = dataset.filter(regex="^(course|concept|video|exercise|school|teacher)").groupby("course_id").first().reset_index()
        if column_names[1]== "video":
            new_column_names = [f"{column_names[0]}_id", f"{column_names[1]}_ccid"]
        elif column_names[1] == "field":
            new_column_names = [f"{column_names[0]}_id", f"course_{column_names[1]}"]
        else:
            new_column_names = [f"{column_names[0]}_id", f"{column_names[1]}_id"]
        df = course_df[new_column_names].explode(new_column_names[1]).drop_duplicates().dropna().copy()
        df.columns = column_names
        return df
    else:
        raise ValueError(f"Unknown relation type: {column_names[0]}-{column_names[1]}")


def read_all_relations(dataset, relations, min_concept_count):
    dataframes = {}
    print(f"Reading relations from {dataset}")
    # Read the combined dataframe
    combine_df = pd.read_hdf(dataset, key="df")
    for relation in relations:
        print(f"Reading relation: {relation}")
        df = read_relations(
            combine_df,
            relation.split("-"),
        )
        dataframes[relation] = df

    print(f"Removing concepts in less than {min_concept_count} courses")
    dataframes["course-concept"] = dataframes["course-concept"][
        dataframes["course-concept"].groupby("concept")["concept"].transform("size")
        > min_concept_count
    ]

    return dataframes


def get_enrolments(dataframes, min_user_count):
    print(f"Removing users enrolled in less than {min_user_count} courses")
    enrolments = dataframes["user-course"][
        dataframes["user-course"].course.isin(dataframes["course-school"].course)
    ]
    enrolments = enrolments[
        enrolments.groupby("user")["user"].transform("size") >= min_user_count
    ]
    return enrolments


def get_all_entities(dataframes, enrolments):
    print(f"Extracting entities")
    entities = {}
    entities["users"] = enrolments.user.unique()
    entities["courses"] = enrolments.course.unique()
    entities["teachers"] = dataframes["course-teacher"][
        dataframes["course-teacher"].course.isin(entities["courses"])
    ].teacher.unique()

    for i, t in enumerate(entities["teachers"]):
        entities["teachers"][i] = t.replace(" ", "_")

    entities["schools"] = dataframes["course-school"][
        dataframes["course-school"].course.isin(entities["courses"])
    ].school.unique()

    entities["concepts"] = dataframes["course-concept"].concept.unique()
    entities["video"] = dataframes["course-video"].video.unique()
    entities["exercise"] = dataframes["course-exercise"].exercise.unique()
    entities["field"] = dataframes["course-field"].field.unique()

    for entity in entities:
        print(f"Number of {entity}: {len(entities[entity])}")

    return entities


def save_entity(entity, file_name):
    print(f"Saving {file_name}")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(entity)
        f.write(out)


def save_entities(save_dir, entities):
    for entity in entities:
        file_name = os.path.join(save_dir, f"{entity}.txt")
        save_entity(entities[entity], file_name)


def get_entity_to_idx(entity):
    entity_to_idx = {}
    for i, e in enumerate(entity):
        entity_to_idx[e] = i
    return entity_to_idx


def get_all_entities_to_idx(entities):
    entities_to_idx = {}
    for entity in entities:
        entities_to_idx[entity] = get_entity_to_idx(entities[entity])
    return entities_to_idx


def save_enrolments(save_dir, enrolments, entities_to_idx):
    # enr_by_user = {}
    print("Saving enrolments")
    out = []
    for idx in enrolments.index:
        u = enrolments.user[idx]
        c = enrolments.course[idx]
        u_idx = entities_to_idx["users"][u]
        c_idx = entities_to_idx["courses"][c]
        # enr_by_user[u_idx] = enr_by_user.get(u_idx, []) + [c_idx]
        out.append(f"{u_idx} {c_idx}")

    file_name = os.path.join(save_dir, "enrolments.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # pkl_file_name = os.path.join(save_dir, "enrolments.pkl")
    # with open(pkl_file_name, "wb") as f:
    #     pickle.dump(enr_by_user, f)


def save_all_relations(save_dir, dataframes, entities, entities_to_idx):
    course_to_school = {}
    course_to_teachers = {}
    course_to_concepts = {}
    course_to_videos = {}
    course_to_exercises = {}
    course_to_fields = {}

    # save course-school relations
    print("Saving course-school relations")
    for idx in dataframes["course-school"].index:
        s = dataframes["course-school"].school[idx]
        c = dataframes["course-school"].course[idx]
        course_to_school[c] = s

    out = []
    for course in entities["courses"]:
        out.append(str(entities_to_idx["schools"][course_to_school[course]]))

    file_name = os.path.join(save_dir, "course_school.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # save course-teacher relations
    print("Saving course-teacher relations")
    for idx in dataframes["course-teacher"].index:
        t = dataframes["course-teacher"].teacher[idx]
        c = dataframes["course-teacher"].course[idx]
        t = t.replace(" ", "_")
        course_to_teachers[c] = course_to_teachers.get(c, []) + [t]

    out = []
    for course in entities["courses"]:
        ts = course_to_teachers.get(course, [])
        ts = [str(entities_to_idx["teachers"][t]) for t in ts]
        out.append(" ".join(ts))

    file_name = os.path.join(save_dir, "course_teachers.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # save course-concept relations
    print("Saving course-concept relations")
    for idx in dataframes["course-concept"].index:
        k = dataframes["course-concept"].concept[idx]
        c = dataframes["course-concept"].course[idx]
        course_to_concepts[c] = course_to_concepts.get(c, []) + [k]

    out = []

    for course in entities["courses"]:
        cs = course_to_concepts.get(course, [])
        cs = [str(entities_to_idx["concepts"][c]) for c in cs]
        out.append(" ".join(cs))

    file_name = os.path.join(save_dir, "course_concepts.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # save course-video relations
    print("Saving course-video relations")
    for idx in dataframes["course-video"].index:
        v = dataframes["course-video"].concept[idx]
        c = dataframes["course-video"].course[idx]
        course_to_videos = course_to_concepts.get(c, [])
        course_to_videos.append(v)

    out = []
    for course in entities["courses"]:
        vs = course_to_concepts.get(course, [])
        vs = [str(entities_to_idx["video"][v]) for v in vs]
        out.append(" ".join(vs))
    
    file_name = os.path.join(save_dir, "course_video.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)

    # save course-exercise relations
    print("Saving course-exercise relations")
    for idx in dataframes["course-exercise"].index: 
        e = dataframes["course-exercise"].concept[idx]
        c = dataframes["course-exercise"].course[idx]
        course_to_exercises = course_to_concepts.get(c, [])
        course_to_exercises.append(e)

    out = []
    for course in entities["courses"]:
        es = course_to_concepts.get(course, [])
        es = [str(entities_to_idx["exercise"][e]) for e in es]
        out.append(" ".join(es))
    file_name = os.path.join(save_dir, "course_exercise.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        out = "\n".join(out)
        f.write(out)
    
    # save course-field relations
    print("Saving course-field relations")
    for idx in dataframes["course-field"].index:
        f = dataframes["course-field"].field[idx]
        c = dataframes["course-field"].course[idx]
        course_to_fields = course_to_concepts.get(c, [])
        course_to_fields.append(f)
    
    out = []
    for course in entities["courses"]:
        fs = course_to_concepts.get(course, [])
        fs = [str(entities_to_idx["field"][f]) for f in fs]
        out.append(" ".join(fs))
    file_name = os.path.join(save_dir, "course_field.txt")
    with open(file_name, "w", encoding="utf-8") as f:  
        out = "\n".join(out)
        f.write(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/")
    parser.add_argument("--save_dir", type=str, default="/preprocessed_files")
    parser.add_argument("--min_concept_count", type=int, default=10)
    parser.add_argument("--min_user_count", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    relations = [
        "course-video",
        "course-exercise",
        "course-field",
        "course-concept",
        "course-school",
        "course-teacher",
        "user-course",
    ]

    dataframes = read_all_relations(args.dataset, relations, args.min_concept_count)

    enrolments = get_enrolments(dataframes, args.min_user_count)

    entities = get_all_entities(dataframes, enrolments)

    save_entities(args.save_dir, entities)

    entities_to_idx = get_all_entities_to_idx(entities)

    save_enrolments(args.save_dir, enrolments, entities_to_idx)

    save_all_relations(args.save_dir, dataframes, entities, entities_to_idx)


if __name__ == "__main__":
    main()
