path = '../data/model_dumps/logs.txt'
filtered_models_path = '../data/model_dumps/filtered_models.txt'
f = open(path, 'r')
lines = f.readlines()
f.close()

filtered_models = []
i = 0
count = 0
while i+10 < len(lines):
    acc = float(lines[i+10].split('test_acc: ')[1].split(',')[0])
    if acc >= 98:
        optim = lines[i].split('optim: ')[1].split(',')[0]
        lr = lines[i].split('lr: ')[1].split(',')[0]
        momentum = lines[i].strip().split('momentum: ')[1].split(',')[0]
        filtered_models.append(f'mnist_{optim}_{lr}_{momentum}')
    i += 11

f = open(filtered_models_path, 'w')
for fm in filtered_models:
    f.write(fm+'\n')
f.close()