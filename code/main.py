from time import strptime
from eiapy import Series, Category
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from dateutil import parser
from datetime import datetime
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
# from pyramid.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from pyramid.arima import autoarima


def ts(timescale):
    if (timescale == 'monthly'):
        return 'M'
    if (timescale == 'annual'):
        return 'A'
    if (timescale == 'quarterly'):
        return 'Q'


def preprocess(data):
    x = []
    y = []
    print(data)
    raw_data = data['series'][0]['data']
    name = data['series'][0]['name'].split(" : ")
    name = name[0] + " " + "(" + name[-1] + ")"
    units = data['series'][0]['units']
    for datapoint in raw_data:
        if len(datapoint[0]) > 4:
            datapoint[0] = datetime.strptime(
                datapoint[0], '%Y%m').strftime('%Y-%m')
        else:
            datapoint[0] = int(datapoint[0])
        x.append(datapoint[0])
        y.append(datapoint[1])
    return x, y, name, units


def query(str, timescale, num):
    return Series(str + ts(timescale)).last(num)


def net_gen_all_fuels(timescale, num):
    data = query('ELEC.GEN.ALL-NY-99.', timescale, num)
    return preprocess(data)


def net_gen_per_fuel(fuel, timescale, num):
    data = query('ELEC.GEN.' + fuel + '-NY-99.', timescale, num)
    return preprocess(data)


def net_demand_all_fuels(timescale, num):
    data = query('ELEC.GEN.ALL-NY-99.', timescale, num)
    return preprocess(data)


def net_demand_per_fuel(fuel, timescale, num):
    data = query('ELEC.GEN.' + fuel + '-NY-99.', timescale, num)
    return preprocess(data)


def total_consumption(timescale, num, btu=False):
    if btu:
        return Category(33)
    return Category(36)


def net_consumption_per_fuel(fuel, timescale, num):
    data = query('ELEC.CONS_EG.' + fuel + '-NY-99.', timescale, num)
    return preprocess(data)


def net_consumption(timescale, num):
    data = query('ELEC.CONS_EG.ALL-NY-99.', timescale, num)
    return preprocess(data)


def net_consumption_by_sector():
    pass


def avg_retail_price(timescale, num):
    data = query('ELEC.PRICE.NY-ALL.', timescale, num)
    return preprocess(data)


data_funcs = {
    "gen": net_gen_all_fuels,
    "price": avg_retail_price,
    "cons": net_consumption,
    "cons_t": net_consumption_per_fuel,
    "dem_per": net_demand_per_fuel,
    "gen_per": net_gen_per_fuel,
}

fuel_types = [
    "COW",
    "PEL",
    "PC",
    "NG",
    "OOG",
    "NUC",
    "HYC",
    "AOR",
    "WND",
    "SUN",
    "GEO",
    "BIO",
    "WWW",
    "WAS",
    "HPS",
    "OTH",
    "TSN",
    "DPV",
    "SUN"]

cons_fuel_types = [
    ""
]


def plot(category, timescale, num, type=None):
    dir = ""
    data_func = data_funcs[category]
    if type != None:
        data_x, data_y, name, units = data_func(type, timescale, num)
        dir = "type/"
    data_x, data_y, name, units = data_func(timescale, num)
    fig, ax = plt.subplots()
    tick_spacing = 2
    if timescale == 'annual':
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    if timescale == 'monthly':
        tick_spacing = 12
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title(name)
    plt.xlabel('year')
    plt.xticks(rotation=45)
    plt.ylabel(units)
    plt.plot(data_x, data_y)
    plt.tight_layout()
    plt.savefig('fig/' + dir + category + "_" + timescale + ".png")


def arima(train, n_periods):
    model = auto_arima(train, trace=True, error_action='ignore',
                       suppress_warnings=True, seasonal=True, m=6, stepwise=True)
    model.fit(train)

    return model.predict(n_periods=n_periods)


def save():
    pass


if __name__ == "__main__":

    plt.style.use('seaborn')

    # Data Extraction and Charts

    # Generation
    # plot("gen", 'monthly', 12*10)
    # plot('gen', 'annual', 20)

    # # Generation per fuel type
    # for type in fuel_types:
    #     plot("gen", 'monthly', 12*10, type)
    #     plot('gen', 'annual', 20, type)

    # # Consumption
    # plot('cons', 'monthly', 12*10)
    # plot('cons', 'annual', 20)

    # # Demand per fuel type
    # for type in fuel_types:
    #     plot("cons_t", 'monthly', 20 * 12, type)
    #     plot('cons_t', 'annual', 12, type)

    # Retail Price
    # plot('price', 'monthly', 20 * 12)
    # plot('price', 'annual', 20)

    # # Consumption for Electricity Generation
    # plot('cons', 'monthly', 20*12)
    # plot('cons', 'annual', 20)

#    # Consumption for Electricity Generation per type
#     plot('cons', 'monthly', 20*12)
#     plot('cons', 'annual', 20)

    # Electricity Generation by Source

    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    labels = ["Petroleum-Fired", "Natural Gas-Fired", "Nuclear",
              "Hydroelectric", "Nonhydroelectric\nRenewables"]
    vals = [1122, 4475, 2477, 2377, 704]

    import seaborn as sns

    sns.set(style="whitegrid")
    # sns.set_color_codes("Spectral")

    plt.figure(2, figsize=(15, 7))

    sns.barplot(x=vals, y=labels, palette='Spectral')

    plt.xlabel("thousand MWh")
    plt.ylabel("Source")

    plt.suptitle('NY State Net Electricity Generation by Source', fontsize=20)
    plt.savefig('fig/elec_gen_by_source.png')
