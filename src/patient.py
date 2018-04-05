class Entry(object):
    def __init__(self, raw_text, date=None):
        self.date = date
        self.raw_text = raw_text

class PatientFile(object):
    def __init__(self, raw_text, entries=None):
        self.raw_text = raw_text
        self.entries  = entries

