from enum import Enum

class TraceTuple:
    def __init__(self, traceID, influenceFactor):
        self.traceID = traceID
        self.influenceFactor = influenceFactor
        return

class TraceList:
    def __init__(self, traceList=None):
        if traceList is None:
            self.traceList = []
        else:
            self.traceList = traceList
    
    def len(self):
        return len(self.traceList)

    def get_all(self):
        return self.traceList

    def get(self, index):
        return self.traceList[index]
    
    def append(self, traceTuple):
        self.traceList.append(traceTuple)

class TracingTypes(Enum):
    NO_TRACING = 0
    TRACE_ID = 1
    TRACE_LIST = 2
    TRACE_VECTOR = 3