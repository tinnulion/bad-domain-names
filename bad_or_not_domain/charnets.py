import string
import torch
import torch.nn

_DOMAIN_LETTERS = "".join([string.ascii_lowercase, string.digits, "._-"])
_HIDDEN_SIZE = 32
_OUTPUT_SIZE = 2


def domain_name_to_tensor(dn):
    dn = dn.lower()
    tensor = torch.zeros(1, len(dn), len(_DOMAIN_LETTERS))
    for i, c in enumerate(dn):
        char_idx = _DOMAIN_LETTERS.find(c)
        if char_idx < 0:
            print("Unknown char '{}' in domain name {}".format(c, dn))
        tensor[0][i][char_idx] = 1
    return tensor


class CharGRU(torch.nn.Module):

    def __init__(self):
        super(CharGRU, self).__init__()

        self.gru = torch.nn.GRU(
            input_size=len(_DOMAIN_LETTERS),
            hidden_size=_HIDDEN_SIZE,
            batch_first=True,
            num_layers=1,
            bias=True
        )
        self.drop = torch.nn.Dropout(p=0.75)
        self.cls_fc = torch.nn.Linear(_HIDDEN_SIZE, _OUTPUT_SIZE)

    def get_zero_hidden(self):
        return torch.zeros(1, 1, _HIDDEN_SIZE).to(self.gru.all_weights[0][0].device)

    def forward(self, x0, h0):
        xt, ht = self.gru(x0, h0)
        v = xt[:, -1, :]
        w = self.drop(v)
        y = self.cls_fc(v)
        return y

    def predict(self, x):
        h0 = self.get_zero_hidden()
        y = self.forward(x, h0)
        return y


def main():
    dn = "google.com"

    net = CharGRU()
    x = domain_name_to_tensor(dn)
    y = net.predict(x).detach().numpy().flatten().tolist()

    print("Log-probs:", y)


if __name__ == "__main__":
    main()
