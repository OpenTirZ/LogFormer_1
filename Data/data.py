import pandas as pd

def GettingData() :
    url = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/HDFS/HDFS_2k.log_structured.csv"
    df = pd.read_csv(url)

    data = df[["EventId" , "EventTemplate"]]
    all_event_ids_list = data["EventId"].tolist()
    print(f"Number of Event IDs: {len(all_event_ids_list)}")
    print(all_event_ids_list) 
    return all_event_ids_list


def makingEventToeventId() :
    url12 = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/HDFS/HDFS_2k.log_templates.csv"
    df1 = pd.read_csv(url12)

    event_template_to_id = df1.set_index('EventTemplate')['EventId'].to_dict()
    print(event_template_to_id)
    return event_template_to_id


def makingVocab() :
    url1 = "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/HDFS/HDFS_2k.log_templates.csv"
    tokens = pd.read_csv(url1)
    tokens = tokens[["EventId"]]
    vocab = {event_id: i for i, event_id in enumerate(tokens['EventId'])}
    return vocab