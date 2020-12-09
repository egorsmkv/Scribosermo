def seconds_to_hours(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    t = "{:d}:{:02d}:{:02d}".format(h, m, s)
    return t
