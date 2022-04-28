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


def net_demand_all_fuels(timescale, num):
    data = query('ELEC.GEN.ALL-NY-99.', timescale, num)
    return preprocess(data)


def net_demand_per_fuel(fuel, timescale, num):
    data = query('ELEC.GEN.ALL-NY-99.', timescale, num)
    return preprocess(data)


def total_consumption(timescale, num, btu=False):
    if btu:
        return Category(33)
    return Category(36)


def net_consumption(timescale, num):
    data = query('ELEC.CONS_EG.ALL-NY-98.', timescale, data)
    return preprocess(data)


def net_consumption(timescale, num):
    pass


def avg_retail_price(timescale, num):
    data = query('ELEC.PRICE.NY-ALL.', timescale, num)
    return preprocess(data)


data_funcs = {
    "gen": net_gen_all_fuels,
    "dem": net_demand_all_fuels,
    "price": avg_retail_price,
    "cons": net_consumption,
    "dem_per": net_demand_per_fuel
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
    plt.savefig('fig/' + type + category + "_" + timescale + ".png")


def save():
    pass


if __name__ == "__main__":

    plt.style.use('ggplot')

    # Data Extraction and Charts

    # Generation
    plot("gen", 'monthly')
    plot('gen', 'annual')

    # Generation per fuel type
    for type in fuel_types:
        plot("gen", 'monthly', type)
        plot('gen', 'annual', type)

    # # Demand
    plot('dem', 'monthly', 12*10)
    plot('dem', 'annual', 20)

    # Demand per fuel type
    for type in fuel_types:
        plot("dem", 'monthly', 20 * 12, type)
        plot('dem', 'annual', 12, type)

    # Retail Price
    plot('price', 'monthly', 20 * 12)
    plot('price', 'annual', 20)

    # Consumption for Electricity Generation
    plot('cons', 'monthly', 20*12)
    plot('cons', 'annual', 20)

   # Consumption for Electricity Generation per type
    plot('cons', 'monthly', 20*12)
    plot('cons', 'annual', 20)
