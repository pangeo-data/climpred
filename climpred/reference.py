def persistence(verif, inits, verif_dates, lead):
    a = verif.where(verif.time.isin(inits[lead]), drop=True)
    b = verif.sel(time=verif_dates[lead])
    a['time'] = b['time']
    return a, b


def historical(hist, verif, inits, verif_dates, lead):
    a = hist.sel(time=verif_dates[lead])
    b = verif.sel(time=verif_dates[lead])
    a['time'] = b['time']
    return a, b
