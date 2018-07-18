import argparse
import os
import torch
import sys

import bad_or_not_domain.charnets as charnets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play around with the domain predictions.')
    parser.add_argument('model', type=str, help='Path to checkpoint')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Cannot find checkpoint at {}".format(args.model))
        sys.exit(1)

    net = charnets.CharGRU()
    net.load_state_dict(torch.load(args.model))

    while True:
        print("Input domain name: ", sep='', end='')
        dn = input()
        if dn == "q":
            break

        x = charnets.domain_name_to_tensor(dn)
        y = net.predict(x).detach().numpy().flatten().tolist()
        print("Log-probs:", y)

        if y[0] >= y[1]:
            print("Domain is GOOD!")
        else:
            print("Domain is BAD!")

    print("Bye!")



