import matplotlib.pyplot as plt
import os
import pandas
import random
import time
import torch

import bad_or_not_domain.charnets as charnets

_MALICIOUS_THRESHOLD = 3

_NUM_EPOCHS = 100
_BATCH_SIZE = 32
_START_LR = 0.001
_DECAY_EVERY = 40
_DECAY_MULT = 0.5

_RES_FOLDER = "results"


def load_train_data_from_single_csv(csv_path, domain_col_idx, detect_col_idx, has_header):
    if has_header:
        df = pandas.read_csv(csv_path)
    else:
        df = pandas.read_csv(csv_path, header=None)
    csv_data = []
    for idx, row in df.iterrows():
        domain_name = row[domain_col_idx]
        count = int(row[detect_col_idx])
        cls = 0
        if count >= _MALICIOUS_THRESHOLD:
            cls = 1
        csv_data.append((domain_name, cls))
    return csv_data


def load_train_data(data_folder):
    data = []
    data.extend(load_train_data_from_single_csv(
        os.path.join(data_folder, "host_detections.csv"),
        0,
        2,
        True
    ))
    data.extend(load_train_data_from_single_csv(
        os.path.join(data_folder, "mal_domains.csv"),
        0,
        1,
        False
    ))
    print("Total data samples: {:d}".format(len(data)))

    # Some interesting stat.
    num_pos = 0
    num_neg = 0
    for _, t in data:
        if t == 0:
            num_pos += 1
        if t == 1:
            num_neg += 1
    print("Number of good domains = {:d}".format(num_pos))
    print("Number of bad domains  = {:d}".format(num_neg))
    return data


def separate_train_and_val(data, val_fraction=0.2):
    random.shuffle(data)
    num_samples = len(data)
    val_size = int(num_samples * val_fraction + 0.5)
    train_data = data[val_size:]
    val_data = data[:val_size]
    return train_data, val_data


def get_lr(epoch_idx):
    return _START_LR * (_DECAY_MULT ** (epoch_idx // _DECAY_EVERY))


def sample_to_tensors(sample, device):
    domain_name, cls = sample
    x = charnets.domain_name_to_tensor(domain_name).to(device)
    t = torch.LongTensor([cls])
    t = t.to(device)
    return x, t


def get_batches(train_data, batch_size, device):
    batches = []
    random.shuffle(train_data)
    accumulator = []
    for sample in train_data:
        x, t = sample_to_tensors(sample, device)
        accumulator.append((x, t))
        if len(accumulator) >= batch_size:
            batches.append(accumulator)
            accumulator = []
    return batches


def train(train_data, net, optimizer, device, global_stat):
    net.train()

    batches = get_batches(train_data, _BATCH_SIZE, device)

    loss_func = torch.nn.CrossEntropyLoss().to(device)

    num_samples = 0
    total_loss = 0.0
    total_acc = 0.0

    for i, batch in enumerate(batches):
        #print("  Processing batch {:d} of {:d}".format(i, len(batches)))

        optimizer.zero_grad()
        batch_loss = 0.0
        batch_acc = 0.0
        for sample in batch:
            x, t = sample
            y = net.predict(x)
            sample_loss = loss_func(y, t)
            sample_loss.backward()

            num_samples += 1
            batch_loss += sample_loss.item()
            if torch.argmax(y) == t.item():
                batch_acc += 1.0

        optimizer.step()

        total_loss += batch_loss
        total_acc += batch_acc

        #batch_loss /= len(batch)
        #batch_acc /= len(batch)
        #print("    Train batch loss = {:.4f} , accuracy = {:.4f}".format(batch_loss, batch_acc))

    total_loss /= num_samples
    total_acc /= num_samples
    print("    Train total loss = {:.4f} , accuracy = {:.4f}".format(total_loss, total_acc))

    global_stat.append(('train', total_loss, total_acc))


def validate(val_data, net, device, global_stat):
    net.eval()

    loss_func = torch.nn.CrossEntropyLoss().to(device)

    total_loss = 0.0
    total_acc = 0.0
    for sample in val_data:
        x, t = sample_to_tensors(sample, device)
        y = net.predict(x)
        sample_loss = loss_func(y, t)

        total_loss += sample_loss.item()
        if torch.argmax(y) == t.item():
            total_acc += 1.0

    total_loss /= len(val_data)
    total_acc /= len(val_data)

    print("    Test  total loss = {:.4f} , accuracy = {:.4f}".format(total_loss, total_acc))

    global_stat.append(('test', total_loss, total_acc))


def plot_stat(global_stat):
    _, (ax1, ax2) = plt.subplots(2, 1, sharex="col")

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for t, loss, acc in global_stat:
        if t == "train":
            train_loss.append(loss)
            train_acc.append(acc)
        else:
            test_loss.append(loss)
            test_acc.append(acc)
    x = list(range(1, len(train_loss) + 1))

    ax1.plot(x, train_loss, 'r-', x, test_loss, 'b-', linewidth=2)
    ax1.set_title('Losses (train is red)')
    ax1.axis("on")

    ax2.plot(x, train_acc, 'r-', x, test_acc, 'b-', linewidth=2)
    ax2.set_title('Accuracies (train is red)')

    plt.savefig("./results/plot.png")


def main():
    data_folder = os.path.abspath("../data")
    data = load_train_data(data_folder)

    train_data, val_data = separate_train_and_val(data)

    #device = torch.device("cpu")
    device = torch.device("cuda:0")

    net = charnets.CharGRU()
    net.to(device)

    current_lr = _START_LR
    optimizer = torch.optim.Adam(net.parameters(), lr=current_lr)

    global_stat = []
    for epoch_idx in range(_NUM_EPOCHS):
        print("Epoch #{:d}".format(epoch_idx))
        start = time.time()

        current_lr = get_lr(epoch_idx)
        print("  Learning rate = {:f}".format(current_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        train(train_data, net, optimizer, device, global_stat)
        validate(val_data, net, device, global_stat)

        print("  Saving model...")
        #path = os.path.abspath("./{}/epoch_{:04d}.pth".format(_RES_FOLDER, epoch_idx))
        path = os.path.abspath("./{}/epoch_last.pth".format(_RES_FOLDER, epoch_idx))
        torch.save(net.state_dict(), path)
        print("  Saved to {}".format(path))

        duration_sec = float(time.time() - start)
        print("  Epoch time = {:.2f} seconds".format(duration_sec))

        plot_stat(global_stat)

    print("Training done.")


if __name__ == "__main__":
    random.seed(0)  # Make debugging a little easier.
    main()

