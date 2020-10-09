import os
import torch

OUT_DIR = 'aistats'


def read_dump(path):
    filepath = f'{OUT_DIR}/{path}/raw_data.pkl'
    raw = torch.load(open(filepath, 'rb'), map_location='cpu')
    return raw


for beta in [1, 5, 10]:
    dump1 = read_dump(f'psj_b_{beta}_bayesian_ns_2')
    dump2 = read_dump(f'psj_b_{beta}_bayesian_ns_3')
    merged_dump = dump1 + dump2
    out_path = f'{OUT_DIR}/psj_b_{beta}_bayesian_ns_5'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(merged_dump, open('{}/raw_data.pkl'.format(out_path), 'wb'))
