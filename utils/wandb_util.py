import logging
import os

import pandas as pd 
import wandb

from utils.meter import AverageMeter


def wandb_log(prefix, sp_values, com_values, update_summary=False, wandb_summary_dict={}):
    """
        prefix + tags.values is the name of sp_values;
        values should include information like:
        {"Acc": 0.9, "Loss":}
        com_values should include information like:
        {"epoch": epoch, }
    """
    new_values = {}
    for k, _ in sp_values.items():
        # new_values[prefix+"/" + k] = sp_values[k]
        new_key = prefix+"/" + k
        new_values[new_key] = sp_values[k]
        if update_summary:
            if new_key not in wandb_summary_dict:
                wandb_summary_dict[new_key] = AverageMeter()
            wandb_summary_dict[new_key].update(new_values[new_key], n=1)
            summary = wandb_summary_dict[new_key].make_summary(new_key)
            for key, valaue in summary.items():
                wandb.run.summary[key] = valaue

    new_values.update(com_values)
    wandb.log(new_values)



# def upload_metric_info(str_pre, train_metric_info, test_metric_info, metrics, comm_values):
#     logging.info(str_pre + 'Train: ' + metrics.str_fn(train_metric_info))
#     logging.info(str_pre + 'Test: ' + metrics.str_fn(test_metric_info))

#     wandb_log(prefix='Train', sp_values=train_metric_info, com_values=comm_values)
#     wandb_log(prefix='Test', sp_values=test_metric_info, com_values=comm_values)


def delete_output_log(path=""):
    api = wandb.Api()
    runs = api.runs(path)
    for run in runs:
        log = run.file("output.log")
        if log and log.size > 0:
            print("Log: {}, size: {}, executing delete....".format(log, log.size))
            log.delete()
        else:
            print("Log: {}, pass....".format(log))




def get_project_runs_from_wandb(entity, project, filters={}, order="-created_at", per_page=50):
    """
        path="", filters={}, order="-created_at", per_page=50
        run: A single run associated with an entity and project.
        Attributes:
            tags ([str]): a list of tags associated with the run
            url (str): the url of this run
            id (str): unique identifier for the run (defaults to eight characters)
            name (str): the name of the run
            state (str): one of: running, finished, crashed, aborted
            config (dict): a dict of hyperparameters associated with the run
            created_at (str): ISO timestamp when the run was started
            system_metrics (dict): the latest system metrics recorded for the run
            summary (dict): A mutable dict-like property that holds the current summary.
                        Calling update will persist any changes.
            project (str): the project associated with the run
            entity (str): the name of the entity associated with the run
            user (str): the name of the user who created the run
            path (str): Unique identifier [entity]/[project]/[run_id]
            notes (str): Notes about the run
            read_only (boolean): Whether the run is editable
            history_keys (str): Keys of the history metrics that have been logged
                with `wandb.log({key: value})`
    """

    api = wandb.Api()
    # Project is specified by <entity/project-name>
    # usage: path="", filters={}, order="-created_at", per_page=50
    path = entity + "/" + project
    runs = api.runs(path, filters, order, per_page)
    summary_list = []
    config_list = []
    name_list = []
    id_list = []
    project_list = []
    uid_list = []
    state_list = []
    url_list = []

    username_list = []
    entity_list = []
    created_at_list = []

    runs_dict = {}
    for run in runs: 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 
        id_list.append(run.id)
        project_list.append(run.project)

        uid = run.entity + '/' + run.project + '/' + run.id
        uid_list.append(uid)
        runs_dict[uid] = run
        state_list.append(run.state)
        url_list.append(run.url)

        username_list.append(run._attrs['user']['username'])
        entity_list.append(run.entity)
        created_at_list.append(run.created_at)
        # run.config is the input metrics.
        # We remove special values that start with _.
        # config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config = {}
        for k, v in run.config.items():
            if not k.startswith('_'):
                if type(v) == list:
                    config[k] = str(v)
                else:
                    config[k] = v
        config_list.append(config) 

        # run.name is the name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list})
    id_df = pd.DataFrame({'id': id_list})
    project_df = pd.DataFrame({'project': project_list})
    uid_df = pd.DataFrame({'uid': uid_list})
    state_df = pd.DataFrame({'state': state_list})
    url_df = pd.DataFrame({'url': url_list})

    username_df = pd.DataFrame({'Username': username_list})
    entity_df = pd.DataFrame({'entity': entity_list})
    created_at_df = pd.DataFrame({'created_at': created_at_list})

    all_df = pd.concat([name_df, config_df, summary_df, id_df, project_df, uid_df,
                        state_df, url_df, username_df, entity_df, created_at_df], axis=1)
    # all_df.to_csv("project.csv")
    return all_df, runs_dict


# def filter_pd_data(pd_frame, x_name, y_name):
#     x_list = list(pd_frame[x_name])
#     y_list = list(pd_frame[y_name])
#     return x_list, y_list


def get_project_path(entity, project):
    return os.path.join(entity, project)


def get_run_folder_name(created_at_number_str, id):
    name = "run" + "-" + created_at_number_str + "-" + id
    return name


def get_run_path(entity, project, created_at_number_str, id):
    folder_name = get_run_folder_name(created_at_number_str, id)
    project_path = get_project_path(entity, project)
    return os.path.join(project_path, folder_name)



def time_to_number_str(time):
    day = time.split('T')[0]
    time = time.split('T')[1]
    day = day.replace("-", "")
    time = time.replace(":", "")
    new_time = day + "_" + time
    return new_time

def number_str_to_time(number_str):
    day = number_str.split('_')[0]
    time = number_str.split('_')[1]
    day = day[0:3] + "-" + day[4:6] + "-" + day[6:]
    time = time[0:2] + ":" + time[2:4] + ":" + time[4:]
    new_time = day + "T" + time
    return new_time












