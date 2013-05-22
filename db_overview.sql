CREATE TABLE `OnTime` (
  `Date` date DEFAULT NULL,
  `Origin` varchar(255) DEFAULT NULL,
  `Destination` varchar(255) DEFAULT NULL,
  `DepartureTime` time DEFAULT NULL,
  `Delay` int(11) DEFAULT NULL,
  `Cancelled` int(11) DEFAULT NULL,
  `Distance` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `AirportOperations` (
  `Airport` text,
  `Date` date DEFAULT NULL,
  `Operations` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE `Weather` (
  `Airport` text,
  `Date` date DEFAULT NULL,
  `Time` time DEFAULT NULL,
  `SkyCondition` varchar(255) DEFAULT NULL,
  `Visibility` decimal(5,2) DEFAULT '0.00',
  `WetBulbCelsius` decimal(5,2) DEFAULT '0.00',
  `DewPointCelsius` double(5,2) DEFAULT NULL,
  `RelativeHumidity` int(3) DEFAULT NULL,
  `WindSpeed` int(4) DEFAULT '0',
  `StationPressure` double(10,4) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*
Training Generation Sample for ORD (O'Hare)
*/
SELECT
OnTime.Distance,Operations.Operations,DailyWeather.Visibility,DailyWeather.DryBulbCelsius,DailyWeather.StationPressure,DailyWeather.WindSpeed,OnTime.Delay
FROM OnTime(
	JOIN DailyWeather
		On OnTime.Date = DailyWeather.Date
	JOIN Operations
		On OnTime.Date = Operations.Date
		)
WHERE OnTime.Origin = "ORD"
LIMIT 100000;