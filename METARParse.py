import re
class METAR:
    def __init__(self,obsStr):
        self.obsStr=obsStr
        self.points={'NOVFR': 1 , 'LowOVC': 0.5}
        self.preexp = "(\w{3})([0-9]+)\s*"
        self.points = 0.0
    def severity(self):
        self.prefixes=re.findall(self.preexp,self.obsStr)
        for i in self.prefixes:
            self.pfx = self.prefixes[i][0]
            self.alt = int(self.prefixes[i][1])
            if self.pfx == 'OVC':
                if self.alt < 10:
                    self.points = 1.0
                if self.alt >= 10 and self.alt < 50:
                    self.points = 0.5
                if self.alt >= 50 and self.alt < 100:
                    self.points = 0.25
                else:
                    self.points = 0.10
            if self.pfx == 'SCT':
                if self.alt < 10:
                    self.points = 0.3
                else:
                    self.points = 0
            else:
                self.points = 0
        return self.points


