import pandas
import numpy as np
import matplotlib.pyplot as plt
import UNIlib as lb

filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def Fterstellung(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    CA_0 = 0
    CA_inf = 0.05
    W_0 = 0.001
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]
    plt.figure(figsize=(10, 6))
    for i in range(3):
        # plt.figure(figsize=(10, 6))
        plt.plot(tK1, Ft[i], label=f"Reaktor {i+1}", color=f"C{i}")
        plt.xlabel("Zeit [min]")
        plt.ylabel("F(t) = (W(t) - W₀) / (W(max) - W₀)")
        plt.title("F(t)-Kurven der Reaktor-Kaskade")
        plt.legend()
        plt.grid(True)
        plt.ylim(
            -0.1, 1.1
        )  # Y-Achse von -0.1 bis 1.1 für bessere Sichtbarkeit von 0 und 1
        plt.savefig(f"/Users/jordihohmann/Desktop/V3 Auswertung/PicAll.png")


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def Ftübereinanderlegung(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    tau = (1.8 * 2) / (19.97 / 60)
    CA_0 = 0
    CA_inf = 0.05
    W_0 = 0.09
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]
    Ftref = [
        [1 - (np.e) ** (-t / tau[0]) for t in tK1],
        [1 - (np.e) ** (-2 * t / tau[1]) * (1 + 2 * (t / tau[1])) for t in tK1],
        [
            1
            - (np.e) ** (-3 * t / tau[2])
            * (1 + 3 * (t / tau[2]) + (9 / 2) * (t / tau[2]) ** 2)
            for t in tK1
        ],
    ]

    # plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.plot(tK1, Ftref[i], label=f"Referenzkurve Reaktor {i+1}")
        plt.plot(tK1, Ft[i], label=f"Kurve Reaktor {i+1}")
        plt.xlabel("Zeit [min]")
        plt.ylabel("F(t) = (W(t) - W₀) / (W(max) - W₀)")
        plt.title("F(t)-Kurven der Reaktor-Kaskade")
        plt.legend()
        plt.grid(True)
        plt.ylim(
            -0.1, 1.1
        )  # Y-Achse von -0.1 bis 1.1 für bessere Sichtbarkeit von 0 und 1
        plt.savefig(
            f"/Users/jordihohmann/Desktop/V3 Auswertung/Picübermitvermeindlichentauaufreaktor2bezogen{i+1}.png"
        )


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def Fttauberechnung(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    tau = [(1.8 * (i + 1)) / (19.97 / 60) for i in range(3)]

    W_0 = 0.00
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]

    Ftref = [
        [1 - (np.e) ** (-t / tau[0]) for t in tK1],
        [1 - (np.e) ** (-2 * t / tau[1]) * (1 + 2 * (t / tau[1])) for t in tK1],
        [
            1
            - (np.e) ** (-3 * t / tau[2])
            * (1 + 3 * (t / tau[2]) + (9 / 2) * (t / tau[2]) ** 2)
            for t in tK1
        ],
    ]
    Rrest = []

    for listindex, a in enumerate(Ft):
        Rrest.append([])
        for i in a:
            Rrest[listindex].append(1 - i)

    return Rrest, tK1


def numericintegration(Rrest, tK1):
    Rrestbalken = [
        [(i + j) / 2 for i, j in zip(Rrest[0][:-1], Rrest[0][1:])],
        [(i + j) / 2 for i, j in zip(Rrest[1][:-1], Rrest[1][1:])],
        [(i + j) / 2 for i, j in zip(Rrest[2][:-1], Rrest[2][1:])],
    ]
    Rrestlängen = [[(j - i) for i, j in zip(tK1[:-1], tK1[1:])]]

    intval = [
        [i * j for i, j in zip(Rrestbalken[0], Rrestlängen[0])],
        [i * j for i, j in zip(Rrestbalken[1], Rrestlängen[0])],
        [i * j for i, j in zip(Rrestbalken[2], Rrestlängen[0])],
    ]

    unseretaussind = [sum(intval[0]), sum(intval[1]), sum(intval[2])]

    return unseretaussind


Rrest, tk1 = Fttauberechnung(data)


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def TaukurvenzuBerechnetenvergleiche(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    tau = [(1.8 * (i + 1)) / (19.97 / 60) for i in range(3)]
    tauexp = [5.29665543792108, 10.375577478344564, 15.67981231953802]
    W_0 = 0.00
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]

    FtExp = [
        [1 - (np.e) ** (-t / 5.29665543792108) for t in tK1],
        [
            1
            - (np.e) ** (-2 * t / 10.375577478344564)
            * (1 + 2 * (t / 10.375577478344564))
            for t in tK1
        ],
        [
            1
            - (np.e) ** (-3 * t / 15.67981231953802)
            * (1 + 3 * (t / 15.67981231953802) + (9 / 2) * (t / 15.67981231953802) ** 2)
            for t in tK1
        ],
    ]
    Ftref = [
        [1 - (np.e) ** (-t / tau[0]) for t in tK1],
        [1 - (np.e) ** (-2 * t / tau[1]) * (1 + 2 * (t / tau[1])) for t in tK1],
        [
            1
            - (np.e) ** (-3 * t / tau[2])
            * (1 + 3 * (t / tau[2]) + (9 / 2) * (t / tau[2]) ** 2)
            for t in tK1
        ],
    ]

    # plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.plot(
            tK1, Ftref[i], label=f"Referenzkurve Reaktor {i+1}, Tau = {tau[i]} [min]"
        )
        plt.plot(
            tK1,
            FtExp[i],
            label=f"Referenzkurve mit Berechnetem Tau {i+1}, Tau = {tauexp[i]} [min]",
        )
        plt.xlabel("Zeit [min]")
        plt.ylabel("F(t)")
        plt.title(
            "Ber. F(t)-Kurven der Reaktor-Kaskade, Vergleich mit Experimentell ermitteltem Tau"
        )
        plt.legend()
        plt.grid(True)
        plt.ylim(
            -0.1, 1.1
        )  # Y-Achse von -0.1 bis 1.1 für bessere Sichtbarkeit von 0 und 1
        plt.savefig(
            f"/Users/jordihohmann/Desktop/V3 Auswertung/PicTauVerglTauberechnetzuTauexperimentell{i+1}.png"
        )


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def TauKurvenzuexpKurvenvergleiche(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    tau = [(1.8 * (i + 1)) / (19.97 / 60) for i in range(3)]
    tauexp = [5.29665543792108, 10.375577478344564, 15.67981231953802]
    W_0 = 0.00
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]

    FtExp = [
        [1 - (np.e) ** (-t / 5.29665543792108) for t in tK1],
        [
            1
            - (np.e) ** (-2 * t / 10.375577478344564)
            * (1 + 2 * (t / 10.375577478344564))
            for t in tK1
        ],
        [
            1
            - (np.e) ** (-3 * t / 15.67981231953802)
            * (1 + 3 * (t / 15.67981231953802) + (9 / 2) * (t / 15.67981231953802) ** 2)
            for t in tK1
        ],
    ]
    Ftref = [
        [1 - (np.e) ** (-t / tau[0]) for t in tK1],
        [1 - (np.e) ** (-2 * t / tau[1]) * (1 + 2 * (t / tau[1])) for t in tK1],
        [
            1
            - (np.e) ** (-3 * t / tau[2])
            * (1 + 3 * (t / tau[2]) + (9 / 2) * (t / tau[2]) ** 2)
            for t in tK1
        ],
    ]

    # plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.plot(tK1, Ft[i], label=f"Kurve Reaktor {i+1}")
        plt.plot(tK1, FtExp[i], label=f"Referenzkurve mit Berechnetem Tau {i+1}")
        plt.xlabel("Zeit [min]")
        plt.ylabel("F(t)")
        plt.title(
            "F(t)-Kurven der Reaktor-Kaskade, Vergleich mit Experimentell ermitteltem Tau"
        )
        plt.legend()
        plt.grid(True)
        plt.ylim(
            -0.1, 1.1
        )  # Y-Achse von -0.1 bis 1.1 für bessere Sichtbarkeit von 0 und 1
        plt.savefig(
            f"/Users/jordihohmann/Desktop/V3 Auswertung/PicTauVerglMitExp(stimmt das tau mit dem überein?){i+1}.png"
        )


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def Umsatzauswertung(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]
    tau = [(1.8 * (i + 1)) / (19.97 / 60) for i in range(3)]
    tauexp = [5.29665543792108, 10.375577478344564, 15.67981231953802]
    F_i = [0]
    C_0i = [0.05]
    R_0 = 8.314
    k = 5  # 8423000*(np.e)**(-44976/(R_0*21))
    ii = 3
    for i in range(ii):
        a = tauexp[i] * k * C_0i[i]
        F = (1 + 2 * a - np.sqrt(1 + 4 * a * (1 - F_i[i]))) / (2 * a)
        F_i.append(F)
        C_0i.append(F * C_0i[i])
    for i in range(ii):
        print(F_i[i + 1])


filepath = "/Users/jordihohmann/Desktop/V3 Auswertung/Versuch3.csv"
data = pandas.read_csv(filepath, sep=";", skiprows=3, header=None)


def Farbigedarstellung(data):
    ll = [[], [], [], [], [], [], [], []]
    for i in range(8):
        ll[i].extend(data.iloc[:, i].astype(float).tolist())

    tK1 = ll[1]

    KK1 = ll[3]
    KK2 = ll[5]
    KK3 = ll[7]

    tau = [(1.8 * (i + 1)) / (19.97 / 60) for i in range(3)]
    tauexp = [5.29665543792108, 10.375577478344564, 15.67981231953802]
    W_0 = 0.00
    W_inf = 10.39

    Ft = [
        [(W - W_0) / (W_inf - W_0) for W in KK1],
        [(W - W_0) / (W_inf - W_0) for W in KK2],
        [(W - W_0) / (W_inf - W_0) for W in KK3],
    ]

    FtExp = [
        [1 - (np.e) ** (-t / 5.29665543792108) for t in tK1],
        [
            1
            - (np.e) ** (-2 * t / 10.375577478344564)
            * (1 + 2 * (t / 10.375577478344564))
            for t in tK1
        ],
        [
            1
            - (np.e) ** (-3 * t / 15.67981231953802)
            * (1 + 3 * (t / 15.67981231953802) + (9 / 2) * (t / 15.67981231953802) ** 2)
            for t in tK1
        ],
    ]
    Ftref = [
        [1 - (np.e) ** (-t / tau[0]) for t in tK1],
        [1 - (np.e) ** (-2 * t / tau[1]) * (1 + 2 * (t / tau[1])) for t in tK1],
        [
            1
            - (np.e) ** (-3 * t / tau[2])
            * (1 + 3 * (t / tau[2]) + (9 / 2) * (t / tau[2]) ** 2)
            for t in tK1
        ],
    ]

    # plt.figure(figsize=(10, 6))
    for i in range(1):
        plt.figure(figsize=(10, 6))
        plt.plot(
            tK1,
            Ftref[i],
            label=f"Referenzkurve Reaktor {i+1}, Tau = {tau[i]} (Flächeninhalt Blauer Bereich) [min]",
        )
        # plt.plot(tK1, FtExp[i], label=f"Referenzkurve mit Berechnetem Tau {i+1}, Tau = {tauexp[i]} [min]")
        plt.fill_between(tK1, Ftref[i], 1, color="blue")
        plt.xlabel("Zeit [min]")
        plt.ylabel("F(t)")
        plt.title("Darstellung von der Graphischen Bedeutung von Tau in F(t)")
        plt.legend()
        plt.grid(True)
        plt.ylim(
            -0.1, 1.1
        )  # Y-Achse von -0.1 bis 1.1 für bessere Sichtbarkeit von 0 und 1
        plt.savefig(
            f"/Users/jordihohmann/Desktop/V3 Auswertung/PicTauVerglTauberechnetzuTauexperimentell{i+1}.png"
        )

    # CA_t = [(W - W_inf) / (W_0 - W_inf) * (CA_0 - CA_inf) + CA_inf for W in KK1]
