from flight import query_flights
from attraction import query_attractions
from accommodation import query_accommodations

print(query_flights("Houston", "Wichita", "2022-04-04"))
print(query_attractions("Houston"))
print(query_accommodations("Houston"))
