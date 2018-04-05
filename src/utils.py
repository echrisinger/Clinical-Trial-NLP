import datetime
import dateutil
import re
import string

# scrub the punctuation from a string
def scrub_punctuation(s):
    exclude = set('!"#$%&\'()*,.:;<=>?@[]^_`{|}~')
    return ''.join(ch for ch in s if ch not in exclude)

# If it gets a match, return that datetime object, otherwise return None
# TODO: stop ignoring records without found dates (find all the dates)
def _get_date_from_str(s):
    m = re.search(r'\d{4}-\d{2}-\d{2}', s)
    if m == None:
        return None
    date = datetime.datetime.strptime(m.group(), '%Y-%m-%d').date()
    return date

# returns a list of dates for a list of entries, corresponding to the date of the 
# entry at the start of each string. Returns none if no date found.
def get_relative_dates(p_file):
    dates = [_get_date_from_str(e.raw_text) for e in p_file.entries]
    m_i, m_d = 0, dates[0]
    for i, d in enumerate(dates):
        if d == None:
            continue
        if m_d < d:
            m_i, m_d = i, d
    rel_dates = []
    for date in dates:
        if date == None:
            rel_dates.append(date)
        else:
            rel_dates.append(m_d-date)
    return rel_dates

# calls metadata helpers on the strings to collect the data from the strings, usually using regular expressions
# Ignores entries without dates currently
def get_metadata(patient_files):
    pfiles_meta = []
    for j, p_file in enumerate(patient_files):
        r_dates = get_relative_dates(p_file)
        for i, _ in enumerate(p_file.entries):
            p_file.entries[i].date = r_dates[i]
        pfiles_meta.append(p_file)
    return pfiles_meta
