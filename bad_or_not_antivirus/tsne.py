import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_antiviruses(line):
    if DATA_PATH.endswith("mal_domains.csv"):
        return line.split(",")[2:]
    assert DATA_PATH.endswith("host_detections.csv")
    antiviruses = [name for name in line.split(",")[1:-1] if name != "None" and name != "0"]
    return antiviruses


def read_data(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            if "listOfDetections" in line:
                continue
            line = line.\
                replace("\"", "").\
                replace("\n", "").\
                replace("u'", "").\
                replace("'", "").\
                replace("[", "").\
                replace("]", "").\
                replace(" ", "")
            antiviruses = get_antiviruses(line)
            data.append(antiviruses)
    return data


def get_distance_matrix_and_av_names(data):
    av_to_i = {}
    av_names = []
    index = 0
    for sample in data:
        for antivirus in sample:
            if antivirus not in av_to_i:
                av_names.append(antivirus)
                av_to_i[antivirus] = index
                index += 1
    distance_matrix = np.zeros((index, index))
    for sample in data:
        av = set(av_to_i.keys())
        for i in range(len(sample)):
            av.remove(sample[i])
            for j in range(len(sample)):
                distance_matrix[av_to_i[sample[i]], av_to_i[sample[j]]] += 1
        av = list(av)
        for i in range(len(av)):
            for j in range(len(av)):
                distance_matrix[av_to_i[av[i]], av_to_i[av[j]]] += 1
    distance_matrix = 1 - distance_matrix / len(data)
    return distance_matrix, av_names


def get_outliers():
    # outliers have been found by human using TSNE plots
    # outliers can mean two things:
    # 1) outliers use some novel good approach which almost none use
    # 2) or outliers use some not valid approach
    # There must be a check by human expert for defining which case is the truth.
    # If it's case number 1 then let's use outliers for predictions otherwise drop them.
    if "mal_domains.csv" in DATA_PATH:
        # equal values in map below means belonging to same outliers cluster
        return {
            "CMC": 6,
            "Rising": 6,
            "ESET-NOD32": 2,
            "NANO-Antivirus": 2,
            "TrendMicro": 3,
            "TrendMicro-HouseCall": 3,
            "ViRobot": 4,
            "F-Prot": 4,
            "TotalDefense": 4,
            "Commtouch": 4,
            "VIPRE": 5
        }
        # Besides outliers there are 4 large clusters:
        # 1) Avast5 or NOD32
        # 2) MicroWorld-eScan
        # 3) K7AntiVirus
        # 4) MacAfee or Kaspersky
        # Probably we can use several (two for example) AVs from a single cluster and got predictions with
        # same accuracy as with using all AVs.

    assert "host_detections.csv" in DATA_PATH
    # equal values in map below means belonging to same outliers cluster
    return {
        "MyWOT": 2,
        "SURBL": 2,
        "DNS-BH": 3,
        "DShield": 3,
        "SCUMWARE": 4,
        "hpHosts": 4,
        "urlQuery": 5,
        "GoogleSafeBrowsing": 5,
        "DrWeb": 6,
    }


def save_plot(embedded, av_names):
    outliers = get_outliers()

    plt.subplots_adjust(bottom=0.1)
    plt.scatter(
        embedded[:, 0],
        embedded[:, 1],
        marker="o",
        s=2,
        c=[outliers[name] if name in outliers else 0 for name in av_names]
    )
    for label, x, y in zip(av_names, embedded[:, 0], embedded[:, 1]):
        plt.annotate(
            label,
            xy=(x, y),
            ha="right",
            va="bottom",
            size=4,
        )

    plt.savefig("tsne.png", dpi=MAC_PRO_DPI)


def main():
    data = read_data(DATA_PATH)
    distance_matrix, av_names = get_distance_matrix_and_av_names(data)
    print("distance_matrix :")
    [print(list(row)) for row in distance_matrix]
    tsne = TSNE(
        metric="precomputed",
        # random_state=0,
        #perplexity=4,
        #perplexity=5,
        perplexity=10,
        learning_rate=10
    )
    embedded = tsne.fit_transform(distance_matrix)
    print("embedded :")
    print(embedded)
    print("save_plot")
    save_plot(embedded, av_names)


MAC_PRO_DPI = 192
DATA_PATH = "../data/mal_domains.csv"
#DATA_PATH = "../data/host_detections.csv"


if __name__ == "__main__":
    main()
