# 导入航班查询函数，用于获取两地之间的航班信息
from flight import query_flights
# 导入景点查询函数，用于获取指定地点的景点信息
from attraction import query_attractions
# 导入住宿查询函数，用于获取指定地点的住宿信息
from accommodation import query_accommodations

# 查询2022年4月4日从休斯顿到威奇托的航班信息并打印结果
print(query_flights("Houston", "Wichita", "2022-04-04"))
# 查询休斯顿的景点信息并打印结果
print(query_attractions("Houston"))
# 查询休斯顿的住宿信息并打印结果
print(query_accommodations("Houston"))