class Entry(object):
    """
    Has the following members:
    date: date of the entry
    raw_text: the raw_text of the entry
    text: the scrubbed text of the entry (without punctuation, de-stemmed, no stopwords)
    """
    def __init__(self, raw_text, date=None):
        self.date = date
        self.raw_text = raw_text

class PatientFile(object):
    def __init__(self, raw_text, entries=None):
        self.raw_text = raw_text
        self.entries  = entries