import redis

NUM_EMULATORS = 8
BASE_TELNET_PORT = 5554

r = redis.Redis(host="localhost", db=0, decode_responses=True)


r.delete("android:idle_ports")
r.delete("android:busy_map")

ports = [BASE_TELNET_PORT + 2 * i for i in range(NUM_EMULATORS)]
r.lpush("android:idle_ports", *ports)

print(" Redis idle port list reset:", ports)
