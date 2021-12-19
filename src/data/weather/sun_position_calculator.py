from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

from math import cos, sin, acos, asin, tan
from math import degrees as deg, radians as rad

'''
Code adapted from but it seems that:
https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html
'''


class DayTime:
    """
    represents a day time object calculated from a raw day value

    the raw_day_value is a decimal day between 0.0 and 1.0, e.g. noon = 0.5
    """

    def __init__(self, raw_day_value: float) -> None:
        if raw_day_value < 0.0 or 1.0 <= raw_day_value:
            raise ValueError("The raw value for a day time has to be inside the half open interval [0.0, 1.0)")
        self.raw_value = raw_day_value
        hours = 24.0 * raw_day_value
        self.hour = int(hours)
        minutes = (hours - self.hour) * 60
        self.minute = int(minutes)
        seconds = (minutes - self.minute) * 60
        self.second = int(seconds)

    @staticmethod
    def from_time(hour: int = 0, minute: int = 0, second: int = 0) -> DayTime:
        if not(0 <= hour <= 24) or not(0 <= minute <= 59) or not(0 <= second <= 59):
            raise ValueError(f"Cannot create a daytime value for the following time {hour}:{minute}:{second}")

        return DayTime((60*60*hour+60*minute+second) / 86400)

    @staticmethod
    def from_datetime(date_time: datetime):
        return DayTime.from_time(date_time.hour, date_time.minute, date_time.second)

    def to_datetime_time(self) -> datetime.time:
        return time(hour=self.hour, minute=self.minute, second=self.second)

    def __assert_valid_other_type(self, other) -> None:
        if not isinstance(other, DayTime):
            raise ValueError(f"Cannot compare this type {type(self)} with other type {type(other)}")

    def compare_to(self, other: DayTime, epsilon: Optional[float] = None) -> float:
        if epsilon is None:
            return self == other
        # lower_bound = other.raw_value * (1 - epsilon)
        # upper_bound = other.raw_value * (1 + epsilon)
        # print(f"lower:{DayTime(lower_bound)}, upper:{DayTime(upper_bound)}")

        return abs(other.raw_value - self.raw_value) < epsilon

    def __eq__(self, other: DayTime):
        self.__assert_valid_other_type(other)
        return self.raw_value == other.raw_value

    def __lt__(self, other: DayTime):
        self.__assert_valid_other_type(other)
        return self.raw_value < other.raw_value

    def __le__(self, other: DayTime):
        self.__assert_valid_other_type(other)
        return self.raw_value <= other.raw_value

    def __str__(self) -> str:
        return f"{self.hour:0>2}:{self.minute:0>2}:{self.second:0>2}"

    def __repr__(self) -> str:
        return f"{(type(self)).__name__}({', '.join([f'{k}={v}' for k,v in vars(self).items()])})"


@dataclass
class SunPositionTimings:
    sunrise: DayTime
    solarnoon: DayTime
    sunset: DayTime


class SunPositionCalculator:
    """
    Calculate sunrise and sunset based on equations from NOAA
    https://www.srrb.noaa.gov/highlights/sunrise/calcdetails.html

    typical use, calculating the sunrise at the present day:

    import datetime
    import sunrise
    s = sun(lat=49,long=3)
    print('sunrise at ',s.sunrise(when=datetime.datetime.now())
    """

    def __init__(self, lat, long):
        self.lat = lat
        self.long = long

    def calc(self, when) -> SunPositionTimings:
        """
        return the time of sunrise as a datetime.time object
        when is a datetime.datetime object. If none is given
        a local time zone is assumed (including daylight saving
        if present)
        """
        self.__prep_time(when)
        self.__calc()
        return SunPositionTimings(
            sunrise=DayTime(self.sunrise_t),
            sunset=DayTime(self.sunset_t),
            solarnoon=DayTime(self.solarnoon_t),
        )

    def __prep_time(self, when: datetime):
        """
        Extract information in a suitable format from when,
        a datetime.datetime object.
        """
        # datetime days are numbered in the Gregorian calendar
        # while the calculations from NOAA are distibuted as
        # OpenOffice spreadsheets with days numbered from
        # 1/1/1900. The difference are those numbers taken for
        # 18/12/2010
        self.day = when.toordinal() - (734124 - 40529)
        t = when.time()
        self.time = (t.hour + t.minute / 60.0 + t.second / 3600.0) / 24.0

        self.timezone = 0
        offset = when.utcoffset()
        if offset is not None:
            self.timezone = offset.seconds / 3600.0

    def __calc(self):
        """
        Perform the actual calculations for sunrise, sunset and
        a number of related quantities.

        The results are stored in the instance variables
        sunrise_t, sunset_t and solarnoon_t
        """
        timezone = self.timezone  # in hours, east is positive
        longitude = self.long  # in decimal degrees, east is positive
        latitude = self.lat  # in decimal degrees, north is positive

        time = self.time  # percentage past midnight, i.e. noon  is 0.5
        day = self.day  # daynumber 1=1/1/1900

        Jday = day + 2415018.5 + time - timezone / 24  # Julian day
        Jcent = (Jday - 2451545) / 36525  # Julian century

        Manom = 357.52911 + Jcent * (35999.05029 - 0.0001537 * Jcent)
        Mlong = 280.46646 + Jcent * (36000.76983 + Jcent * 0.0003032) % 360
        Eccent = 0.016708634 - Jcent * (0.000042037 + 0.0001537 * Jcent)
        Mobliq = 23 + (26 + (21.448 - Jcent * (46.815 + Jcent * (0.00059 - Jcent * 0.001813))) / 60) / 60
        obliq = Mobliq + 0.00256 * cos(rad(125.04 - 1934.136 * Jcent))
        vary = tan(rad(obliq / 2)) * tan(rad(obliq / 2))
        Seqcent = sin(rad(Manom)) * (1.914602 - Jcent * (0.004817 + 0.000014 * Jcent)) + sin(rad(2 * Manom)) * (0.019993 - 0.000101 * Jcent) + sin(
            rad(3 * Manom)) * 0.000289
        Struelong = Mlong + Seqcent
        Sapplong = Struelong - 0.00569 - 0.00478 * sin(rad(125.04 - 1934.136 * Jcent))
        declination = deg(asin(sin(rad(obliq)) * sin(rad(Sapplong))))

        eqtime = 4 * deg(
            vary * sin(2 * rad(Mlong)) - 2 * Eccent * sin(rad(Manom)) + 4 * Eccent * vary * sin(rad(Manom)) * cos(2 * rad(Mlong)) - 0.5 * vary * vary * sin(
                4 * rad(Mlong)) - 1.25 * Eccent * Eccent * sin(2 * rad(Manom)))

        hourangle = deg(acos(cos(rad(90.833)) / (cos(rad(latitude)) * cos(rad(declination))) - tan(rad(latitude)) * tan(rad(declination))))

        self.solarnoon_t = (720 - 4 * longitude - eqtime + timezone * 60) / 1440
        self.sunrise_t = self.solarnoon_t - hourangle * 4 / 1440
        self.sunset_t = self.solarnoon_t + hourangle * 4 / 1440

