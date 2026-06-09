import torch
from torch.utils.data import Dataset , DataLoader

class TirZ_LogFormer_dataSet(Dataset) :
    def __init__(self, event_ids_list, vocab, max_lenght , stride) :
        self.input_ids = []
        self.target_ids = []

        token_ids = [vocab[event_id] for event_id in event_ids_list]

        for i in range(0, len(token_ids) - max_lenght, stride) :
            input_chunk = token_ids[i : i + max_lenght]
            target_token = token_ids[i + max_lenght]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_token))

    def __len__(self) :
        return len(self.input_ids)

    def __getitem__(self, idx) :
        return self.input_ids[idx] , self.target_ids[idx]

eventList= ["E1","E2","E3","E4", "E13", "E5","E5","E6","E7","E8","E9","E10"]


def create_dataloader(event_ids_list, vocab, batch_size =4,max_length = 4 , stride = 1 ,
                      shuffle = True , drop_last = True , num_workers = 0) :

    dataset = TirZ_LogFormer_dataSet(event_ids_list, vocab, max_lenght=max_length , stride=stride)

    dataloader = DataLoader(
        dataset=dataset,

        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader